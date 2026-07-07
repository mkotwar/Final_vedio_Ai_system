import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
from PIL import Image
import os
import re
import csv
import threading
import queue
import numpy as np
import math
import time
import json
from datetime import datetime
from ultralytics import YOLO
import supervision as sv

# GLOBAL CONFIGURATION
VIDEO_PATH = r"D:\FrogCeLL\Videos\ANPR + speed\RAW1.mp4"
# VIDEO_PATH = r"rtsp://admin:abcd1234@10.10.3.112:554/Streaming/Channels/1"
CAMERA_ID = "CAM_01_Main_Gate"
VEHICLE_COUNT = 0
ANALYTICS_NAME = "ANPR + SPEED"

# PATHS
VEHICLE_MODEL_PATH = r"best_old.pt"
PLATE_MODEL_PATH = r"license_plate_weights.pt"
FRAME_SKIP = 1

# FLORENCE-2 CONFIGURATION
BASE_MODEL_ID = "microsoft/Florence-2-base-ft"
ADAPTER_PATH = r"adaptor_florance_baseFT" 
device = "cuda" if torch.cuda.is_available() else "cpu"

VEHICLE_CONFIDENCE_THRESHOLD = 0.2
PLATE_CONFIDENCE_THRESHOLD = 0.5
CLASSES_TO_TRACK = [0,1,2,3,4,5,6,7]

# SPEED CONFIGURATION
LINE_ORIENTATION = "horizontal"
REAL_WORLD_DISTANCE_METERS = 4
RIGHT_DIRECTION_LABEL = "Right Direction" 
WRONG_DIRECTION_LABEL = "Wrong Direction" 

MAX_QUEUE_SIZE = 500
LOG_COOLDOWN_SECONDS = 30
CLEANUP_INTERVAL = 50
STALE_THRESHOLD_SECONDS = 180
DETECTION_COOLDOWN = 0.4

CSV_FILENAME = "detection/detected_plates_log.csv"
VALID_DETECTIONS_FOLDER = "detection/valid_detections"
RECOVERED_DETECTIONS_FOLDER = "detection/recovered_detections"
REPROCESSED_CSV_FILENAME = "detection/recovered_plate_detection.csv"
INVALID_CSV_FILENAME = "detection/invalid_plate_detection.csv"
FINAL_INVALID_FOLDER = "detection/final_invalid_detections"
CROPPED_PLATE_FOLDER = "detection/cropped_plates"

# Global Model Placeholders
model = None
processor = None

# probe frame size
cap_probe = cv2.VideoCapture(VIDEO_PATH)
if cap_probe.isOpened():
    FRAME_WIDTH = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()
else:
    FRAME_WIDTH = 1440
    FRAME_HEIGHT = 810
    print("Warning: Could not probe frame size.")

VALID_STATE_CODES = [
    "AN","AP","AR","AS","BR","CH","DN","DD","DL","GA","GJ",
    "HR","HP","JK","KA","KL","LD","MP","MH","MN","ML","MZ",
    "NL","OR","PY","PB","RJ","SK","TN","TR","UP","WB","TS",
    "UK","LA","CG","JH"
]

# threading infra
frame_queue = queue.Queue(maxsize=1)
vehicle_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
job_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
reprocessing_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
stop_event = threading.Event()
count_lock = threading.Lock()

# FLORENCE HELPER FUNCTIONS
def run_florence_inference(image_cv, task_prompt, text_input=None, use_adapter=True):
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    prompt = task_prompt + text_input if text_input else task_prompt
    inputs = processor(text=prompt, images=image_pil, return_tensors="pt").to(device)
    
    # Toggle adapter based on flag
    context = torch.no_grad() if use_adapter else model.disable_adapter()
    
    with context:
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], 
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, 
            do_sample=False, 
            num_beams=3, 
            use_cache=False,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image_pil.width, image_pil.height)
    )
    return parsed_answer

# SETUP & UTILS
def setup_lines_manual(default_offset_px=100):
    """
    Manual two-line setup.
    Returns: lines_endpoints = [((x1,y1),(x2,y2)), ((x3,y3),(x4,y4))]
    """
    print("[Setup] Starting manual 2-line setup.")
    cap_local = cv2.VideoCapture(VIDEO_PATH)
    if not cap_local.isOpened():
        print("[Setup] ERROR: cannot open video for setup.")
        exit()
    success, frame = cap_local.read()
    cap_local.release()
    if not success or frame is None:
        print("[Setup] ERROR: failed to grab setup frame.")
        exit()

    win = "Line Setup - Draw 1st line (2 clicks) -> press 'c', then 2nd line -> 'c' -> Enter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    points = []      
    lines = []       
    selected = None  
    dragging = False
    prev_mouse = (0, 0)
    endpoint_radius = 8
    hit_dist = 12

    def point_distance(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def point_on_segment_distance(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        c1 = vx*wx + vy*wy
        if c1 <= 0: return math.hypot(px-ax, py-ay)
        c2 = vx*vx + vy*vy
        if c2 <= c1: return math.hypot(px-bx, py-by)
        b = c1 / c2
        projx, projy = ax + b*vx, ay + b*vy
        return math.hypot(px-projx, py-projy)

    def mouse_cb(event, x, y, flags, param):
        nonlocal points, lines, selected, dragging, prev_mouse
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(lines) == 2:
                for li, line in enumerate(lines):
                    (ax, ay), (bx, by) = line
                    if point_distance((x,y), (ax,ay)) <= hit_dist:
                        selected = ('endpoint', li, 0); dragging = True; prev_mouse = (x, y); return
                    if point_distance((x,y), (bx,by)) <= hit_dist:
                        selected = ('endpoint', li, 1); dragging = True; prev_mouse = (x, y); return
                for li, line in enumerate(lines):
                    (ax, ay), (bx, by) = line
                    if point_on_segment_distance(x, y, ax, ay, bx, by) <= hit_dist:
                        selected = ('body', li); dragging = True; prev_mouse = (x, y); return
                return
            else:
                if len(points) < 2: points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging and selected is not None:
                mx, my = x, y
                dx, dy = mx - prev_mouse[0], my - prev_mouse[1]
                typ, li = selected[0], selected[1]
                if typ == 'endpoint':
                    ep_idx = selected[2]
                    (ax, ay), (bx, by) = lines[li]
                    if ep_idx == 0: lines[li] = ((ax + dx, ay + dy), (bx, by))
                    else: lines[li] = ((ax, ay), (bx + dx, by + dy))
                elif typ == 'body':
                    (ax, ay), (bx, by) = lines[li]
                    lines[li] = ((ax + dx, ay + dy), (bx + dx, by + dy))
                prev_mouse = (mx, my)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False; selected = None

    cv2.setMouseCallback(win, mouse_cb)

    while True:
        disp = frame.copy()
        for p in points: cv2.circle(disp, (int(p[0]), int(p[1])), 5, (0,255,255), -1)
        for idx, line in enumerate(lines):
            (ax, ay), (bx, by) = line
            color = (0, 255, 0) if idx == 0 else (0, 0, 255)
            cv2.line(disp, (int(ax), int(ay)), (int(bx), int(by)), color, 2)
            cv2.circle(disp, (int(ax), int(ay)), endpoint_radius, color, -1)
            cv2.circle(disp, (int(bx), int(by)), endpoint_radius, color, -1)

        if len(lines) == 0:
            cv2.putText(disp, "Click 2 points -> Press 'c' to confirm first line", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        elif len(lines) == 1:
            cv2.putText(disp, "Click 2 points -> Press 'c' to confirm second line", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(disp, "Drag endpoints. Press Enter when done.", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('c'):
            if len(points) == 2 and len(lines) < 2:
                lines.append((points[0], points[1]))
                points = []
                print(f"[Setup] Confirmed line #{len(lines)}")
        elif key == ord('r'):
            lines = []; points = []
            print("[Setup] Reset lines.")
        elif key in (13, 10):
            if len(lines) == 2: break
        elif key == 27:
            cv2.destroyWindow(win); exit()

    cv2.destroyWindow(win)
    return lines

def is_valid_indian_plate(text):
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    pattern_1 = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
    pattern_2 = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
    if len(clean_text) > 10: return False
    if pattern_1.match(clean_text): return clean_text[:2] in VALID_STATE_CODES
    elif pattern_2.match(clean_text): return True
    return False

def resize_proportionally_if_needed(image, target_width=200, target_height=150):
    h, w = image.shape[:2]
    if h == 0 or w == 0: return image
    if w < target_width or h < target_height:
        scale_factor = max(target_width / w, target_height / h)
        return cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_LINEAR)
    return image

# HREAD FUNCTIONS
def grab_frames(video_path, frame_queue):
    print("[Frame Grabber Thread] Starting...")
    capg = cv2.VideoCapture(video_path)
    if not capg.isOpened(): stop_event.set(); return
    frame_count = 0
    while not stop_event.is_set():
        success, frame = capg.read()
        if not success: stop_event.set(); break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0: continue
        try: frame_queue.put(frame, block= False)
        except queue.Full: pass
    capg.release()
    print("[Frame Grabber Thread] Stopped.")

def track_vehicles_and_calculate_speed(frame_queue, vehicle_queue, vehicle_model, tracker, fps, lines_endpoints):
    print(f"[Vehicle Tracker Thread] Starting with FPS: {fps:.2f}")
    vehicle_class_names = vehicle_model.model.names
    vehicle_class_names[2] = "4Wheeler"; vehicle_class_names[3] = "2Wheeler"
    
    orderrr = ["2Wheeler", "3Wheeler", "4Wheeler", "bus", "truck"]
    vehicle_type_counts = {name: 0 for name in orderrr}

    frame_count = 0
    vehicle_data = {}
    last_cleanup_time = time.time()

    if LINE_ORIENTATION == "horizontal":
        line1_pos = int(round((lines_endpoints[0][0][1] + lines_endpoints[0][1][1]) / 2.0))
        line2_pos = int(round((lines_endpoints[1][0][1] + lines_endpoints[1][1][1]) / 2.0))
    else:
        line1_pos = int(round((lines_endpoints[0][0][0] + lines_endpoints[0][1][0]) / 2.0))
        line2_pos = int(round((lines_endpoints[1][0][0] + lines_endpoints[1][1][0]) / 2.0))

    cv2.namedWindow("Vehicle Tracking", cv2.WINDOW_NORMAL)

    while not stop_event.is_set():
        try: frame = frame_queue.get(timeout=1)
        except queue.Empty: continue

        frame_count += 1
        current_time = time.time()
        results = vehicle_model(frame, conf=VEHICLE_CONFIDENCE_THRESHOLD, imgsz=1024, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = tracker.update_with_detections(detections)

        for xyxy, confidence, class_id, tracker_id in zip(tracked_detections.xyxy, tracked_detections.confidence, tracked_detections.class_id, tracked_detections.tracker_id):
            x1, y1, x2, y2 = map(int, xyxy)
            track_id = tracker_id
            vehicle_type = vehicle_class_names[class_id]

            if track_id not in vehicle_data:
                vehicle_data[track_id] = {'first_crossing_frame': 0, 'first_crossing_line': None, 'second_crossing_frame': 0, 'speed_kmh': 0, 'direction': 'N/A', 'job_queued': False, 'vehicle_type': vehicle_type, 'last_pos': None}
            vehicle_data[track_id]['last_seen_time'] = current_time

            track_point = y2 if LINE_ORIENTATION == "horizontal" else (x1 + x2) // 2
            l1, l2 = line1_pos, line2_pos
            last_pos = vehicle_data[track_id]['last_pos']

            if last_pos is not None:
                if vehicle_data[track_id]['first_crossing_frame'] == 0:
                    if (last_pos < l1 <= track_point) or (last_pos > l1 >= track_point):
                        vehicle_data[track_id].update({'first_crossing_line': 1, 'first_crossing_frame': frame_count})
                    elif (last_pos < l2 <= track_point) or (last_pos > l2 >= track_point):
                        vehicle_data[track_id].update({'first_crossing_line': 2, 'first_crossing_frame': frame_count})
                elif vehicle_data[track_id]['second_crossing_frame'] == 0:
                    crossed_other = False
                    if vehicle_data[track_id]['first_crossing_line'] == 1 and ((last_pos < l2 <= track_point) or (last_pos > l2 >= track_point)): crossed_other = True
                    elif vehicle_data[track_id]['first_crossing_line'] == 2 and ((last_pos < l1 <= track_point) or (last_pos > l1 >= track_point)): crossed_other = True
                    
                    if crossed_other:
                        vehicle_data[track_id]['second_crossing_frame'] = frame_count
                        frame_diff = abs(vehicle_data[track_id]['second_crossing_frame'] - vehicle_data[track_id]['first_crossing_frame'])
                        if frame_diff > 0:
                            speed_kmh = (REAL_WORLD_DISTANCE_METERS / (frame_diff / fps)) * 3.6
                            vehicle_data[track_id]['speed_kmh'] = speed_kmh
                            vehicle_data[track_id]['direction'] = RIGHT_DIRECTION_LABEL if vehicle_data[track_id]['first_crossing_line'] == 1 else WRONG_DIRECTION_LABEL

            vehicle_data[track_id]['last_pos'] = track_point
            speed = vehicle_data[track_id]['speed_kmh']

            if speed > 0 and not vehicle_data[track_id]['job_queued']:
                if vehicle_type in vehicle_type_counts: vehicle_type_counts[vehicle_type] += 1
                vehicle_job = {'full_frame': frame.copy(), 'vehicle_roi_coords': (x1, y1, x2, y2), 'speed_kmh': speed, 'vehicle_type': vehicle_type, 'direction': vehicle_data[track_id]['direction'], 'tracker_id': track_id}
                try: vehicle_queue.put(vehicle_job, block=False); vehicle_data[track_id]['job_queued'] = True
                except queue.Full: pass

            label = f"ID:{track_id} {vehicle_type}"
            if speed > 0: label += f"|{speed:.1f}km/h"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if current_time - last_cleanup_time > CLEANUP_INTERVAL:
            stale = [tid for tid, d in vehicle_data.items() if current_time - d.get('last_seen_time', 0) > STALE_THRESHOLD_SECONDS]
            for tid in stale: vehicle_data.pop(tid, None)
            last_cleanup_time = current_time

        # Dashboard
        dash_x, dash_y, dash_w, line_h = 10, 10, 350, 32
        dash_h = 40 + (len(vehicle_type_counts) * line_h)
        if dash_y + dash_h < frame.shape[0] and dash_x + dash_w < frame.shape[1]:
            sub = frame[dash_y:dash_y+dash_h, dash_x:dash_x+dash_w]
            res = cv2.addWeighted(sub, 0.6, np.zeros(sub.shape, dtype=np.uint8), 0.4, 1.0)
            frame[dash_y:dash_y+dash_h, dash_x:dash_x+dash_w] = res
            cv2.putText(frame, "Vehicle Counts:", (dash_x+5, dash_y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cy = dash_y+30
            for v_type, count in vehicle_type_counts.items():
                cy += line_h
                cv2.putText(frame, f"- {v_type}: {count}", (dash_x+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        (a1, a2), (b1, b2) = lines_endpoints
        cv2.line(frame, (int(a1[0]), int(a1[1])), (int(a2[0]), int(a2[1])), (0,0,255), 2)
        cv2.line(frame, (int(b1[0]), int(b1[1])), (int(b2[0]), int(b2[1])), (0,0,255), 2)
        cv2.imshow("Vehicle Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): stop_event.set()
    cv2.destroyAllWindows(); print("[Vehicle Tracker Thread] Stopped.")

def extract_plates(plate_model, camera_id):
    print("[Plate Extractor Thread] Starting...")
    last_job_creation_time = None
    while not stop_event.is_set() or not vehicle_queue.empty():
        try:
            vehicle_job = vehicle_queue.get(timeout=1)
            if last_job_creation_time and (datetime.now() - last_job_creation_time).total_seconds() < DETECTION_COOLDOWN:
                continue
            x1, y1, x2, y2 = vehicle_job['vehicle_roi_coords']
            vehicle_roi = vehicle_job['full_frame'][y1:y2, x1:x2]
            if vehicle_roi.size <= 0: vehicle_queue.task_done(); continue
            
            vehicle_roi_for_color = resize_proportionally_if_needed(vehicle_roi.copy())
            plate_results = plate_model(vehicle_roi, conf=PLATE_CONFIDENCE_THRESHOLD, verbose=False)
            plate_crop = None

            if len(plate_results) > 0 and len(plate_results[0].boxes) > 0:
                best_plate = max(plate_results[0].boxes, key=lambda b: b.conf[0])
                px1, py1, px2, py2 = map(int, best_plate.xyxy[0])
                plate_crop_raw = vehicle_roi[py1:py2, px1:px2]
                if plate_crop_raw.size > 0:
                    plate_crop = resize_proportionally_if_needed(plate_crop_raw)
                    ycrcb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2YCrCb)
                    y, cr, cb = cv2.split(ycrcb)
                    y_eq = cv2.equalizeHist(y)
                    plate_crop = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)
                    plate_crop = cv2.GaussianBlur(plate_crop, (3,3), 0)
            
            anpr_job = {**vehicle_job, 'timestamp': datetime.now(), 'plate_crop': plate_crop, 'vehicle_roi_crop': vehicle_roi_for_color}
            try: job_queue.put(anpr_job, block=False); last_job_creation_time = datetime.now()
            except queue.Full: pass
            vehicle_queue.task_done()
        except queue.Empty: continue
    print("[Plate Extractor Thread] Stopped.")

def process_and_log(camera_id):
    print("[Processing Thread] Starting...")
    global VEHICLE_COUNT
    os.makedirs(VALID_DETECTIONS_FOLDER, exist_ok=True)
    os.makedirs(CROPPED_PLATE_FOLDER, exist_ok=True)
    recent_detections = {}

    while not stop_event.is_set() or not job_queue.empty():
        try:
            job = job_queue.get(timeout=1)
            timestamp = job['timestamp']
            plate_crop = job['plate_crop']
            vehicle_roi_crop = job['vehicle_roi_crop']
            full_frame = job['full_frame']
            speed = job.get('speed_kmh', 0.0)
            vehicle_type = job.get('vehicle_type', 'Unknown')
            direction = job.get('direction', 'N/A')

            # FLORENCE OCR
            extracted_text = "NOT_FOUND"
            if plate_crop is not None:
                try:
                    res = run_florence_inference(plate_crop, "<OCR>", use_adapter=True)
                    extracted_text = res.get('<OCR>', 'NOT_FOUND')
                except Exception: extracted_text = "OCR_FAILED"
            
            # FLORENCE VQA (COLOR)
            vehicle_color = "N/A"
            try:
                res = run_florence_inference(vehicle_roi_crop, "<VQA>", "What is the primary color of the vehicle?", use_adapter=False)
                vehicle_color = res.get('<VQA>', 'N/A')
            except Exception: vehicle_color = "N/A"

            clean_text = re.sub(r'[^A-Z0-9]', '', extracted_text.upper())
            print(f"[Processing Thread] Florence says: Plate='{clean_text}', Color='{vehicle_color}'")
            
            plate_number_to_log = clean_text if clean_text else "OCR_FAILED"
            is_valid = is_valid_indian_plate(plate_number_to_log)

            if is_valid:
                if plate_number_to_log in recent_detections and (timestamp - recent_detections[plate_number_to_log]).total_seconds() < LOG_COOLDOWN_SECONDS:
                    job_queue.task_done(); continue
                recent_detections[plate_number_to_log] = timestamp

                with count_lock:
                    VEHICLE_COUNT += 1
                    tracker_id = job.get('tracker_id', 'UNKNOWN')
                    filename = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_ID_{tracker_id}.jpg"
                    full_frame_path = os.path.join(VALID_DETECTIONS_FOLDER, filename)
                    cv2.imwrite(full_frame_path, full_frame)
                    cropped_path = "N/A"
                    if plate_crop is not None:
                        cropped_path = os.path.join(CROPPED_PLATE_FOLDER, filename)
                        cv2.imwrite(cropped_path, plate_crop)

                    with open(CSV_FILENAME, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if f.tell() == 0: writer.writerow(['vehicle_count', 'camera_id', 'timestamp', 'plate_number', 'speed_kmh', 'vehicle_type', 'vehicle_color', 'direction', 'full_frame_image_path', 'cropped_plate_path', 'analytics_name'])
                        writer.writerow([VEHICLE_COUNT, camera_id, timestamp, plate_number_to_log, f"{speed:.1f}", vehicle_type, vehicle_color, direction, full_frame_path, cropped_path, ANALYTICS_NAME])
            else:
                reprocessing_job = {**job, 'original_text': plate_number_to_log, 'original_color': vehicle_color}
                try: reprocessing_queue.put(reprocessing_job, block=False)
                except queue.Full: pass
            job_queue.task_done()
        except queue.Empty: continue
        except Exception: job_queue.task_done()
    print("[Processing Thread] Stopped.")

def reprocess_invalid_plates(camera_id):
    print("[Reprocessing Thread] Starting...")
    global VEHICLE_COUNT
    os.makedirs(FINAL_INVALID_FOLDER, exist_ok=True)
    os.makedirs(RECOVERED_DETECTIONS_FOLDER, exist_ok=True)
    recent_detections = {}

    while not stop_event.is_set() or not reprocessing_queue.empty():
        try:
            job = reprocessing_queue.get(timeout=1)
            timestamp = job['timestamp']
            plate_crop = job['plate_crop']
            vehicle_roi_crop = job['vehicle_roi_crop']
            full_frame = job['full_frame']
            original_text = job['original_text']
            speed = job.get('speed_kmh', 0.0)
            vehicle_type = job.get('vehicle_type', 'Unknown')
            direction = job.get('direction', 'N/A')

            # FLORENCE OCR (RETRY)
            new_text = "NOT_FOUND"
            if plate_crop is not None:
                try:
                    res = run_florence_inference(plate_crop, "<OCR>", use_adapter=True)
                    new_text = res.get('<OCR>', 'NOT_FOUND')
                except Exception: new_text = "PARSE_ERROR"
            
            # FLORENCE VQA (COLOR)
            vehicle_color = "N/A"
            try:
                res = run_florence_inference(vehicle_roi_crop, "<VQA>", "What is the primary color of the vehicle?", use_adapter=False)
                vehicle_color = res.get('<VQA>', 'N/A')
            except Exception: vehicle_color = "N/A"

            clean_text = re.sub(r'[^A-Z0-9]', '', new_text.upper())
            print(f"[Reprocessing Thread] Florence says: Plate='{clean_text}', Color='{vehicle_color}'")

            if is_valid_indian_plate(clean_text):
                if clean_text in recent_detections and (timestamp - recent_detections[clean_text]).total_seconds() < LOG_COOLDOWN_SECONDS:
                    reprocessing_queue.task_done(); continue
                recent_detections[clean_text] = timestamp
                tracker_id = job.get('tracker_id', 'UNKNOWN')
                filename = f"RECOVERED_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_ID_{tracker_id}.jpg"
                full_frame_path = os.path.join(RECOVERED_DETECTIONS_FOLDER, filename)
                cv2.imwrite(full_frame_path, full_frame)
                cropped_path = "N/A"
                if plate_crop is not None:
                    c_dir = os.path.join(RECOVERED_DETECTIONS_FOLDER, "cropped_plates")
                    os.makedirs(c_dir, exist_ok=True)
                    cropped_path = os.path.join(c_dir, filename)
                    cv2.imwrite(cropped_path, plate_crop)

                with count_lock:
                    VEHICLE_COUNT += 1
                    with open(REPROCESSED_CSV_FILENAME, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if f.tell() == 0: writer.writerow(['vehicle_count', 'camera_id', 'timestamp', 'recovered_plate_number', 'speed_kmh', 'vehicle_type', 'vehicle_color', 'direction', 'original_text', 'image_path', 'cropped_plate', 'analytics_name'])
                        writer.writerow([VEHICLE_COUNT, camera_id, timestamp, clean_text, f"{speed:.1f}", vehicle_type, vehicle_color, direction, original_text, full_frame_path, cropped_path, ANALYTICS_NAME])
            else:
                tracker_id = job.get('tracker_id', 'UNKNOWN')
                filename = f"FINAL_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}_ID_{tracker_id}.jpg"
                full_frame_path = os.path.join(FINAL_INVALID_FOLDER, filename)
                cv2.imwrite(full_frame_path, full_frame)
                cropped_path = "N/A"
                if plate_crop is not None:
                    c_dir = os.path.join(FINAL_INVALID_FOLDER, "cropped_plates")
                    os.makedirs(c_dir, exist_ok=True)
                    cropped_path = os.path.join(c_dir, filename)
                    cv2.imwrite(cropped_path, plate_crop)
                plate_to_log = clean_text if clean_text else original_text

                with count_lock:
                    VEHICLE_COUNT += 1
                    with open(INVALID_CSV_FILENAME, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if f.tell() == 0: writer.writerow(['vehicle_count', 'camera_id', 'timestamp', 'plate_number', 'speed_kmh', 'vehicle_type', 'vehicle_color', 'direction', 'full_frame_image_path', 'cropped_plate', 'analytics_name'])
                        writer.writerow([VEHICLE_COUNT, camera_id, timestamp, plate_to_log, f"{speed:.1f}", vehicle_type, vehicle_color, direction, full_frame_path, cropped_path, ANALYTICS_NAME])
            reprocessing_queue.task_done()
        except queue.Empty: continue
        except Exception: reprocessing_queue.task_done()
    print("[Reprocessing Thread] Stopped.")

if __name__ == "__main__":
    print("[Main] Initializing...")

    # 1. LOAD FLORENCE-2 MODEL AND ADAPTER
    print(f"Loading Base Model: {BASE_MODEL_ID} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, attn_implementation="eager").to(device)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    
    print(f"Loading LoRA Adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    # Manual setup for lines
    lines_endpoints = setup_lines_manual()

    if LINE_ORIENTATION == "horizontal":
        line1_pos = int(round((lines_endpoints[0][0][1] + lines_endpoints[0][1][1]) / 2.0))
        line2_pos = int(round((lines_endpoints[1][0][1] + lines_endpoints[1][1][1]) / 2.0))
    else:
        line1_pos = int(round((lines_endpoints[0][0][0] + lines_endpoints[0][1][0]) / 2.0))
        line2_pos = int(round((lines_endpoints[1][0][0] + lines_endpoints[1][1][0]) / 2.0))

    cap_fps = cv2.VideoCapture(VIDEO_PATH)
    video_fps = cap_fps.get(cv2.CAP_PROP_FPS) if cap_fps.isOpened() else 25
    cap_fps.release()
    print(f"[Main] FPS: {video_fps}")

    try:
        vehicle_model = YOLO(VEHICLE_MODEL_PATH)
        plate_model = YOLO(PLATE_MODEL_PATH)
    except Exception as e:
        print(f"Model Load Error: {e}"); exit()

    tracker = sv.ByteTrack(lost_track_buffer=40, track_activation_threshold=0.3, minimum_matching_threshold=0.6, minimum_consecutive_frames=3)

    t1 = threading.Thread(target=grab_frames, args=(VIDEO_PATH, frame_queue))
    t2 = threading.Thread(target=track_vehicles_and_calculate_speed, args=(frame_queue, vehicle_queue, vehicle_model, tracker, video_fps, lines_endpoints))
    t3 = threading.Thread(target=extract_plates, args=(plate_model, CAMERA_ID))
    t4 = threading.Thread(target=process_and_log, args=(CAMERA_ID,))
    t5 = threading.Thread(target=reprocess_invalid_plates, args=(CAMERA_ID,))

    print("[Main] Starting threads...")
    t1.start(); t2.start(); t3.start(); t4.start(); t5.start()
    t1.join(); t2.join(); t3.join(); t4.join(); t5.join()
    print("[Main] Shutdown complete.")