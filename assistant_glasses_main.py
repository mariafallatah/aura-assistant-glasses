# assistant_glasses_main.py

import io
import os
import re
import time
import threading
from flask import Flask, Response, jsonify, request

# =========================
#  HARDWARE IMPORTS
# =========================

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("MISSING LIBRARY: picamera2 — camera disabled. Install with: pip install picamera2")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("MISSING LIBRARY: opencv-python — YOLO frame conversion disabled. Install with: pip install opencv-python")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("MISSING LIBRARY: RPi.GPIO — ultrasonic sensors and panic button disabled. Install with: pip install RPi.GPIO")

try:
    import ncnn
    import numpy as np
    NCNN_AVAILABLE = True
except ImportError:
    NCNN_AVAILABLE = False
    print("MISSING LIBRARY: ncnn — object detection disabled. Install with: pip install ncnn")

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("MISSING LIBRARY: pyserial — UWB navigation disabled. Install with: pip install pyserial")

try:
    import pyaudio
    import json
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("MISSING LIBRARY: pyaudio or vosk — voice recognition disabled. Install with: pip install pyaudio vosk")


# =========================
# CONFIGURATION
# =========================

app = Flask(__name__)

# GPIO pins
PANIC_BUTTON_PIN   = 17
LEFT_TRIG_PIN      = 23
LEFT_ECHO_PIN      = 24
RIGHT_TRIG_PIN     = 27
RIGHT_ECHO_PIN     = 22

# Thresholds
OBSTACLE_DISTANCE_LIMIT_CM = 80

# Paths
MODEL_PARAM      = "/home/pi/Desktop/model_ncnn_model/model.ncnn.param"
MODEL_BIN        = "/home/pi/Desktop/model_ncnn_model/model.ncnn.bin"
VOSK_MODEL_PATH  = "/home/pi/ag/models/vosk-model-small-en-us-0.15"

# UWB serial
UWB_PORT         = "/dev/serial0"
UWB_BAUD         = 115200

# Camera
CAMERA_RESOLUTION = (640, 480)

# =========================
# GLOBAL STATE
# =========================

picamera        = None          # Picamera2 instance (panic stream + sign reading)
yolo_model      = None
vosk_recognizer = None
vosk_stream     = None
uwb_serial      = None

camera_lock     = threading.Lock()  # one operation on the camera at a time
panic_streaming = False             # True while panic stream is active
yolo_paused     = False             # True while panic stream holds the camera

# Current active mode — set by voice commands
# Modes: None | "navigation" | "find_seat"
current_mode    = None

# Cooldown tracker — prevents espeak from firing every second for the same detection
_last_spoken    = {}   # { label: timestamp }
SPEECH_COOLDOWN = 5.0  # seconds before repeating the same phrase

system_state = {
    "obstacle":               None,
    "objects":                [],
    "current_position":       None,
    "destination":            None,
    "navigation_instruction": None,
    "speech_text":            None,
    "voice_command":          None,
    "panic_active":           False,
    "panic_streaming":        False,
    "last_alert":             None,
    "mode":                   None,   # "navigation" | "find_seat" | None
}


# =========================
# CAMERA MODULE
# (uses Picamera2 for everything — one library, no conflict)
# =========================

def setup_camera():
    global picamera

    if not PICAMERA_AVAILABLE:
        print("Picamera2 not installed. Camera disabled.")
        return

    picamera = Picamera2()
    config = picamera.create_video_configuration(
        main={"size": CAMERA_RESOLUTION}
    )
    picamera.configure(config)
    picamera.start()
    print("Camera started (Picamera2).")


def get_camera_frame_numpy():
    """
    Returns a numpy (BGR) frame for YOLO, or None.
    Only called when not panic-streaming.
    """
    if picamera is None or yolo_paused:
        return None

    with camera_lock:
        frame = picamera.capture_array()   # RGB numpy array

    # YOLO / OpenCV expect BGR
    if CV2_AVAILABLE:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


def get_camera_frame_jpeg():
    """
    Returns raw JPEG bytes for the panic stream.
    """
    if picamera is None:
        return None

    with camera_lock:
        buf = io.BytesIO()
        picamera.capture_file(buf, format="jpeg")
        return buf.getvalue()


# --- Panic stream generator ---

def generate_panic_frames():
    while panic_streaming:
        frame = get_camera_frame_jpeg()
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )
        time.sleep(0.05)   # ~20 fps


@app.route("/stream")
def panic_stream():
    if not panic_streaming:
        return jsonify({"error": "Stream not active. Trigger /panic first."}), 404
    return Response(
        generate_panic_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =========================
# OBSTACLE DETECTION MODULE
# (dual sensors from od.py)
# =========================

def setup_ultrasonic():
    if not GPIO_AVAILABLE:
        print("GPIO not available. Ultrasonic disabled.")
        return

    for pin in (LEFT_TRIG_PIN, RIGHT_TRIG_PIN):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, False)

    for pin in (LEFT_ECHO_PIN, RIGHT_ECHO_PIN):
        GPIO.setup(pin, GPIO.IN)

    time.sleep(0.5)
    print("Ultrasonic sensors ready (left + right).")


def read_distance(trig, echo):
    """
    Returns distance in cm from one HC-SR04 sensor, or None on timeout.
    """
    if not GPIO_AVAILABLE:
        return None

    GPIO.output(trig, False)
    time.sleep(0.05)

    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    timeout    = time.time() + 0.04
    pulse_start = None
    pulse_end   = None

    while GPIO.input(echo) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            return None

    timeout = time.time() + 0.04

    while GPIO.input(echo) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            return None

    if pulse_start is None or pulse_end is None:
        return None

    return round((pulse_end - pulse_start) * 17150, 2)


def obstacle_detection():
    left  = read_distance(LEFT_TRIG_PIN,  LEFT_ECHO_PIN)
    right = read_distance(RIGHT_TRIG_PIN, RIGHT_ECHO_PIN)

    if left is None or right is None:
        return {
            "obstacle_detected": False,
            "left_cm":  left,
            "right_cm": right,
            "message":  "Sensor error",
            "timestamp": time.time(),
        }

    lim = OBSTACLE_DISTANCE_LIMIT_CM

    if left < lim and right < lim:
        message = "Obstacle ahead"
    elif left < lim:
        message = "Obstacle on the left"
    elif right < lim:
        message = "Obstacle on the right"
    else:
        message = None

    return {
        "obstacle_detected": message is not None,
        "left_cm":  left,
        "right_cm": right,
        "message":  message or "Path is clear",
        "timestamp": time.time(),
    }


def obstacle_loop():
    while True:
        result = obstacle_detection()
        system_state["obstacle"] = result

        if result["obstacle_detected"]:
            system_state["last_alert"] = result["message"]
            text_to_speech(result["message"])

        time.sleep(0.3)


@app.route("/obstacle/status")
def obstacle_status():
    return jsonify(system_state["obstacle"])


# =========================
# OBJECT DETECTION MODULE
# =========================

def setup_ncnn():
    if not NCNN_AVAILABLE:
        print("ncnn not installed. Object detection disabled.")
        return
    print("ncnn ready. Model will load per detection call.")


def object_detection(frame):
    if not NCNN_AVAILABLE or frame is None:
        return {
            "human_detected": False,
            "door_detected":  False,
            "chair_detected": False,
            "detections":     [],
            "message":        "ncnn not available",
        }

    CLASS_NAMES    = {0: "chair", 1: "door", 2: "person"}
    CONF_THRESHOLD = 0.5
    NMS_THRESHOLD  = 0.45
    INPUT_SIZE     = 640

    detections     = []
    human_detected = False
    door_detected  = False
    chair_detected = False

    with ncnn.Net() as net:
        net.load_param(MODEL_PARAM)
        net.load_model(MODEL_BIN)
        with net.create_extractor() as ex:
            mat_in = ncnn.Mat.from_pixels_resize(
                frame,
                ncnn.Mat.PixelType.PIXEL_RGB,
                frame.shape[1], frame.shape[0],
                INPUT_SIZE, INPUT_SIZE
            )
            mat_in.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])
            ex.input("in0", mat_in)
            _, out = ex.extract("out0")
            out = np.array(out).T

    boxes      = []
    scores     = []
    class_ids  = []

    for det in out:
        x, y, w, h    = det[0], det[1], det[2], det[3]
        class_scores  = det[4:]
        cls           = int(np.argmax(class_scores))
        conf          = float(class_scores[cls])
        if conf < CONF_THRESHOLD:
            continue
        boxes.append([float(x - w / 2), float(y - h / 2), float(w), float(h)])
        scores.append(conf)
        class_ids.append(cls)

    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            for i in indices.flatten():
                label = CLASS_NAMES.get(class_ids[i], "unknown")
                detections.append({
                    "label":      label,
                    "confidence": round(scores[i] * 100, 1)
                })
                if label == "person":
                    human_detected = True
                elif label == "door":
                    door_detected  = True
                elif label == "chair":
                    chair_detected = True

    speak_if_ready("human", "Human ahead") if human_detected else None
    speak_if_ready("door",  "Door ahead")  if door_detected  else None

    if chair_detected and current_mode == "find_seat":
        speak_if_ready("chair", "Seat detected")

    return {
        "human_detected": human_detected,
        "door_detected":  door_detected,
        "chair_detected": chair_detected,
        "detections":     detections,
        "timestamp":      time.time(),
    }


def object_detection_loop():
    while True:
        if not yolo_paused:
            frame  = get_camera_frame_numpy()
            result = object_detection(frame)
            system_state["objects"] = result
        time.sleep(1)


@app.route("/objects/status")
def objects_status():
    return jsonify(system_state["objects"])


# =========================
# SIGN READING MODULE
# (from frs1.py)
# =========================

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("MISSING LIBRARY: pytesseract — sign reading disabled. Install with: pip install pytesseract")

IMAGE_PATH     = "/home/pi/captured_sign.jpg"
PROCESSED_PATH = "/home/pi/processed_sign.jpg"


def read_sign():
    """
    Captures a photo, runs OCR on it, speaks the result.
    Triggered by voice command "read sign".
    """
    if not PICAMERA_AVAILABLE:
        text_to_speech("Camera not available")
        return {"text": None, "error": "Camera not available"}

    if not TESSERACT_AVAILABLE:
        text_to_speech("Sign reading not available")
        return {"text": None, "error": "Tesseract not available"}

    text_to_speech("Reading sign")

    # --- CAPTURE ---
    with camera_lock:
        picamera.capture_file(IMAGE_PATH)

    # --- LOAD ---
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        text_to_speech("Could not read image")
        return {"text": None, "error": "Image not found"}

    # --- CROP CENTER ---
    h, w, _ = img.shape
    img = img[int(h * 0.20):int(h * 0.80), int(w * 0.10):int(w * 0.90)]

    # --- PREPROCESS ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    processed_images = []

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(otsu)

    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed_images.append(otsu_inv)

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    processed_images.append(adaptive)

    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    adaptive_norm = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 10)
    processed_images.append(adaptive_norm)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed_images = [cv2.morphologyEx(p, cv2.MORPH_CLOSE, kernel) for p in processed_images]

    # --- OCR ---
    configs = [
        "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ]

    candidates = []
    for image in processed_images:
        for config in configs:
            raw   = pytesseract.image_to_string(image, config=config).strip().upper()
            clean = re.sub(r"[^A-Z0-9]", "", raw)
            if clean:
                candidates.append(clean)

    # --- SCORE ---
    def normalize_candidate(text):
        text  = re.sub(r"[^A-Z0-9]", "", text.upper())
        match = re.search(r"[A-Z][0-9]{2,4}", text)
        return match.group(0) if match else text

    def score_text(text):
        score        = 0
        common_signs = ["EXIT", "ENTRANCE", "ROOM", "WC", "OFFICE", "LAB"]
        if text in common_signs:
            score += 100
        if re.fullmatch(r"[A-Z][0-9]{2,4}", text):
            score += 80
        score -= len(text)
        return score

    clean_candidates = list(set(
        normalize_candidate(c) for c in candidates
        if 2 <= len(normalize_candidate(c)) <= 8
    ))
    clean_candidates.sort(key=score_text, reverse=True)

    best_text = clean_candidates[0] if clean_candidates else ""

    # --- SPEAK ---
    if best_text:
        text_to_speech(f"The sign says {best_text}")
    else:
        text_to_speech("No text detected")

    return {"text": best_text or None, "candidates": clean_candidates}


@app.route("/sign/read", methods=["POST"])
def sign_read_route():
    result = read_sign()
    return jsonify(result)


# =========================
# NAVIGATION MODULE
# (from uwb.py)
# =========================

def send_uwb_cmd(cmd):
    print(">>", cmd)
    uwb_serial.write((cmd + "\r\n").encode())
    time.sleep(0.5)


def setup_uwb():
    global uwb_serial

    if not SERIAL_AVAILABLE:
        print("pyserial not installed. UWB disabled.")
        return

    try:
        uwb_serial = serial.Serial(UWB_PORT, UWB_BAUD, timeout=1)
        time.sleep(2)

        send_uwb_cmd("AT")
        send_uwb_cmd("AT+version?")
        send_uwb_cmd("AT+RST")
        send_uwb_cmd("AT+anchor_tag=0")
        send_uwb_cmd("AT+interval=5")
        send_uwb_cmd("AT+switchdis=1")

        print("UWB ready. Listening...")
    except Exception as e:
        print(f"UWB setup failed: {e}")
        uwb_serial = None


def get_current_position():
    """
    Reads latest raw data from UWB module.
    Returns raw serial output. Coordinate parsing to be added
    once UWB output format is confirmed.
    """
    if uwb_serial is None or not uwb_serial.in_waiting:
        return {"raw": None, "timestamp": time.time()}

    data = uwb_serial.read(uwb_serial.in_waiting).decode(errors="ignore")
    print(data, end="")
    return {"raw": data.strip(), "timestamp": time.time()}


def navigation(current_position, destination):
    if destination is None:
        return {
            "instruction":      "No destination selected",
            "current_position": current_position,
            "destination":      None,
        }

    return {
        "current_position": current_position,
        "destination":      destination,
        "instruction":      f"Navigate to {destination}",
        "timestamp":        time.time(),
    }


def navigation_loop():
    while True:
        pos = get_current_position()
        system_state["current_position"] = pos
        system_state["navigation_instruction"] = navigation(
            pos, system_state["destination"]
        )
        time.sleep(1)


@app.route("/navigation/set_destination", methods=["POST"])
def set_destination():
    destination = request.get_json().get("destination")
    system_state["destination"] = destination
    return jsonify({"status": "success", "destination": destination})


@app.route("/navigation/status")
def navigation_status():
    return jsonify(system_state["navigation_instruction"])


# =========================
# SPEECH TO TEXT MODULE
# (Vosk from mic_test.py)
# =========================

def setup_vosk():
    global vosk_recognizer, vosk_stream

    if not VOSK_AVAILABLE:
        print("Vosk not installed. Voice recognition disabled.")
        return

    try:
        model          = Model(VOSK_MODEL_PATH)
        vosk_recognizer = KaldiRecognizer(model, 16000)

        p = pyaudio.PyAudio()
        vosk_stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000
        )
        vosk_stream.start_stream()
        print("Vosk voice recognition ready.")
    except Exception as e:
        print(f"Vosk setup failed: {e}")


def speech_to_text():
    if vosk_recognizer is None or vosk_stream is None:
        return {"text": None, "confidence": 0.0, "timestamp": time.time()}

    data = vosk_stream.read(4000, exception_on_overflow=False)

    if vosk_recognizer.AcceptWaveform(data):
        result = json.loads(vosk_recognizer.Result())
        text   = result.get("text", "").strip()
        if text:
            return {"text": text, "confidence": 1.0, "timestamp": time.time()}

    return {"text": None, "confidence": 0.0, "timestamp": time.time()}


# =========================
# VOICE COMMAND MODULE
# =========================

def voice_command(text):
    global current_mode

    if text is None:
        return {"command": None, "action": None}

    text = text.lower()

    if text == "help" or text == "emergency":
        trigger_panic_alert(source="voice")
        return {"command": text, "action": "panic_alert"}

    if "read sign" in text or "what does it say" in text:
        threading.Thread(target=read_sign, daemon=True).start()
        return {"command": text, "action": "read_sign"}

    if "chair" in text or "seat" in text:
        current_mode = "find_seat"
        text_to_speech("Finding seat. Looking for chairs.")
        return {"command": text, "action": "find_seat"}

    if "door" in text:
        return {"command": text, "action": "detect_door"}

    if "take me to" in text or "go to" in text:
        destination = text.replace("take me to", "").replace("go to", "").strip()
        system_state["destination"] = destination
        current_mode = "navigation"
        text_to_speech(f"Navigating to {destination}")
        return {"command": text, "action": "navigation", "destination": destination}

    if "stop" in text or "cancel" in text:
        current_mode = None
        text_to_speech("Stopped")
        return {"command": text, "action": "stop"}

    return {"command": text, "action": "unknown"}


def voice_loop():
    while True:
        speech_result = speech_to_text()
        system_state["speech_text"] = speech_result

        text = speech_result["text"]
        if text:
            system_state["voice_command"] = voice_command(text)
            system_state["mode"] = current_mode

        time.sleep(0.5)


@app.route("/voice/status")
def voice_status():
    return jsonify({
        "speech_text":  system_state["speech_text"],
        "voice_command": system_state["voice_command"],
    })


# =========================
# TEXT TO SPEECH MODULE
# =========================

def text_to_speech(text):
    print(f"TTS: {text}")
    os.system(f'espeak "{text}" 2>/dev/null')
    return {"spoken_text": text, "status": "spoken", "timestamp": time.time()}


@app.route("/tts/speak", methods=["POST"])
def tts_from_app():
    text = request.get_json().get("text", "")
    return jsonify(text_to_speech(text))


def speak_if_ready(label, phrase):
    """
    Speaks phrase only if SPEECH_COOLDOWN seconds have passed since the last
    time this label was spoken. Prevents espeak from firing every second.
    """
    now = time.time()
    if now - _last_spoken.get(label, 0) >= SPEECH_COOLDOWN:
        _last_spoken[label] = now
        text_to_speech(phrase)


# =========================
# PANIC BUTTON MODULE
# =========================

def setup_panic_button():
    if not GPIO_AVAILABLE:
        print("GPIO not available. Panic button disabled.")
        return

    GPIO.setup(PANIC_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    threading.Thread(target=panic_button_loop, daemon=True).start()
    print("Panic button ready.")


def panic_button_loop():
    last_state = GPIO.input(PANIC_BUTTON_PIN)
    while True:
        current_state = GPIO.input(PANIC_BUTTON_PIN)
        if last_state == GPIO.HIGH and current_state == GPIO.LOW:
            trigger_panic_alert(source="physical_button")
        last_state = current_state
        time.sleep(0.05)


def trigger_panic_alert(source="unknown"):
    global panic_streaming, yolo_paused

    location = get_current_position()

    system_state["panic_active"]    = True
    system_state["panic_streaming"] = True
    system_state["last_alert"]      = "Panic alert triggered"

    # Pause YOLO so the camera is free for the stream
    yolo_paused     = True
    panic_streaming = True

    text_to_speech("Emergency alert activated")

    panic_data = {
        "panic":     True,
        "source":    source,
        "location":  location,
        "message":   "Emergency alert sent. Stream available at /stream",
        "timestamp": time.time(),
    }

    print("PANIC ALERT:", panic_data)
    return panic_data


def reset_panic_alert():
    global panic_streaming, yolo_paused

    panic_streaming = False
    yolo_paused     = False

    system_state["panic_active"]    = False
    system_state["panic_streaming"] = False

    return {"panic": False, "message": "Panic alert reset. YOLO resumed."}


@app.route("/panic", methods=["POST"])
def panic_from_app():
    return jsonify(trigger_panic_alert(source="app"))


@app.route("/panic/reset", methods=["POST"])
def reset_panic_from_app():
    return jsonify(reset_panic_alert())


@app.route("/panic/status")
def panic_status():
    return jsonify({
        "panic_active":    system_state["panic_active"],
        "panic_streaming": system_state["panic_streaming"],
        "stream_url":      "/stream" if panic_streaming else None,
        "location":        system_state["current_position"],
        "last_alert":      system_state["last_alert"],
    })


# =========================
# FULL SYSTEM STATUS
# =========================

@app.route("/status")
def full_status():
    return jsonify(system_state)


# =========================
# SETUP AND MAIN
# =========================

def print_status_loop():
    while True:
        print("\n--- SYSTEM STATUS ---")
        print(f"Mode:         {current_mode}")
        print(f"Panic:        {system_state['panic_active']}")
        print(f"Obstacle:     {system_state['obstacle']}")
        print(f"Objects:      {system_state['objects']}")
        print(f"Position:     {system_state['current_position']}")
        print(f"Last alert:   {system_state['last_alert']}")
        print("---------------------\n")
        time.sleep(10)


def setup_system():
    print("Starting Assistant Glasses System...")

    if GPIO_AVAILABLE:
        GPIO.setmode(GPIO.BCM)

    setup_camera()
    setup_ultrasonic()
    setup_panic_button()
    setup_ncnn()
    setup_uwb()
    setup_vosk()

    print("System setup complete.")


def start_background_threads():
    threads = [
        threading.Thread(target=obstacle_loop,         daemon=True),
        threading.Thread(target=object_detection_loop, daemon=True),
        threading.Thread(target=navigation_loop,       daemon=True),
        threading.Thread(target=voice_loop,            daemon=True),
        threading.Thread(target=print_status_loop,     daemon=True),
    ]
    for t in threads:
        t.start()


def main():
    setup_system()
    start_background_threads()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    try:
        main()
    finally:
        if picamera is not None:
            picamera.stop()
        if uwb_serial is not None:
            uwb_serial.close()
        if GPIO_AVAILABLE:
            GPIO.cleanup()
