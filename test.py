from flask import Flask, render_template, Response, request, jsonify , send_file
import cv2
from ultralytics import YOLO
import easyocr
import re
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
import base64
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)

# Load YOLO model
model_path = './runs/detect/train11/weights/last.pt'
model = YOLO(model_path)

# License plate details to detect
Imatricule = "2665 تونس 147"
serial = '147'
code = '2665'

# OCR setup
reader = easyocr.Reader(['ar', 'en'])

# Detection parameters
threshold = 0.25
detected_text = None
match_found = False

# Email configuration
sender_email = "selim.belkhire@etudiant-enit.utm.tn"
receiver_email = "selim.belkhire@etudiant-enit.utm.tn"
password = "tbtlxuqrcdaevnem"
smtp_server = "smtp.gmail.com"
smtp_port = 587


camera = None
camera_active = False
video_path = None
processing_video = False
video_frame = None
frame_global = None
last_detection_time = 0
detection_cooldown = 10
processed_frames = []

esp32_cam_url = None
esp32_cam_active = False
esp32_cam_frame = None
esp32_cam_thread = None


def send_email(subject, body, img_path=None):
    """Send email with optional image attachment"""
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if img_path:
            with open(img_path, 'rb') as f:
                data = f.read()
            image = MIMEImage(data, name='detected_frame.jpg')
            msg.attach(image)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully!")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


def process_frame(frame):
    """Process video frame to detect license plates"""
    global detected_text, match_found, last_detection_time

    if frame is None:
        return None

    processed_frame = frame.copy()
    current_time = time.time()

    if current_time - last_detection_time < detection_cooldown:
        return processed_frame

    results = model(processed_frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            label = results.names[int(class_id)].upper()
            cv2.putText(processed_frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)


            license_plate_region = frame[int(y1):int(y2), int(x1):int(x2)]


            if license_plate_region.size == 0 or license_plate_region.shape[0] < 10 or license_plate_region.shape[
                1] < 10:
                continue


            gray = cv2.cvtColor(license_plate_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            resized = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            ocr_results = reader.readtext(resized)


            if ocr_results:
                for (bbox, text, prob) in ocr_results:
                    if prob > 0.2:
                        print(f"Raw OCR Output: {text}")

                        text = re.sub(r"\?+", "تونس", text)
                        print(f"Processed Text: {text}")

                        cv2.putText(processed_frame, text, (int(x1), int(y2 + 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


                        if serial in text and code in text:
                            print("Target plate found: " + text)
                            detected_text = text


                            timestamp = int(time.time())
                            img_path = f'detected_frame_{timestamp}.jpg'
                            cv2.imwrite(img_path, processed_frame)


                            cv2.putText(processed_frame, "MATCH FOUND!", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)


                            last_detection_time = current_time

                            subject = "License Plate Match Found"
                            body = (f"Your specific license plate was detected: {Imatricule}\n"
                                    f"(serial number = {serial} and code = {code})\n"
                                    f"Detected text: {detected_text}\n"
                                    f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

                            email_thread = threading.Thread(
                                target=send_email,
                                args=(subject, body, img_path)
                            )
                            email_thread.start()

                            print(f"Captured frame and sending email notification")
            else:
                print("No text detected in the license plate region.")

    return processed_frame


def gen_frames_webcam():
    """Generate frames from webcam for streaming"""
    global camera, frame_global, camera_active


    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   open('static/no_camera.jpg', 'rb').read() + b'\r\n')
            return

    camera_active = True

    while camera_active:
        success, frame = camera.read()
        if not success:
            break
        else:

            frame_global = frame.copy()

            processed_frame = process_frame(frame_global)

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.03)

    if camera is not None:
        camera.release()
        camera = None


def esp32_cam_stream():
    """Function to continuously fetch frames from ESP32-CAM"""
    global esp32_cam_url, esp32_cam_active, esp32_cam_frame

    while esp32_cam_active and esp32_cam_url:
        try:
            if esp32_cam_url.endswith('/stream'):
                response = requests.get(esp32_cam_url, stream=True, timeout=5)
                if response.status_code == 200:
                    bytes_data = bytes()
                    for chunk in response.iter_content(chunk_size=1024):
                        bytes_data += chunk
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')
                        if a != -1 and b != -1:
                            jpg = bytes_data[a:b + 2]
                            bytes_data = bytes_data[b + 2:]
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            esp32_cam_frame = frame
                            if not esp32_cam_active:
                                break
                else:
                    print("Failed to connect to ESP32-CAM stream")
                    time.sleep(5)
            else:
                response = requests.get(esp32_cam_url, timeout=5)
                if response.status_code == 200:
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    esp32_cam_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    time.sleep(0.1)  # Small delay between requests
                else:
                    print(f"Failed to get frame from ESP32-CAM: {response.status_code}")
                    time.sleep(1)
        except Exception as e:
            print(f"Error fetching ESP32-CAM frame: {e}")
            time.sleep(5)  # Wait before retrying

    print("ESP32-CAM streaming stopped")


def gen_frames_esp32():
    """Generate frames from ESP32 CAM for streaming"""
    global esp32_cam_frame, esp32_cam_active, frame_global

    if esp32_cam_url is None:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               open('static/no_camera.jpg', 'rb').read() + b'\r\n')
        return

    esp32_cam_active = True

    if esp32_cam_thread is None or not esp32_cam_thread.is_alive():
        esp32_thread = threading.Thread(target=esp32_cam_stream)
        esp32_thread.daemon = True
        esp32_thread.start()

    while esp32_cam_active:
        if esp32_cam_frame is not None:
            frame_global = esp32_cam_frame.copy()

            processed_frame = process_frame(frame_global)

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame is available yet, send placeholder
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   open('static/waiting.jpg', 'rb').read() + b'\r\n')

        time.sleep(0.03)


def gen_frames_video():
    """Generate frames from video file for streaming"""
    global video_path, processing_video, processed_frames, video_frame

    if video_path is None or not os.path.exists(video_path):
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               open('static/no_video.jpg', 'rb').read() + b'\r\n')
        return

    # Don't reprocess if we already have processed frames
    if not processed_frames and not processing_video:
        processing_video = True
        processed_frames = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            processing_video = False
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   open('static/error.jpg', 'rb').read() + b'\r\n')
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 fps if detection fails

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video with {total_frames} frames at {fps} fps")

        # Process the video
        frame_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            if frame_count % 10 == 0:  # Print progress every 10 frames
                print(f"Processing frame {frame_count}/{total_frames} ({frame_count / total_frames * 100:.1f}%)")

            processed_frame = process_frame(frame)

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if ret:
                processed_frames.append(buffer.tobytes())

        cap.release()
        processing_video = False
        print(f"Video processing complete. {len(processed_frames)} frames processed.")

        # Create output video file
        if len(processed_frames) > 0:
            try:
                output_path = f"{video_path.rsplit('.', 1)[0]}_processed.mp4"

                # Need to get the first frame to determine dimensions
                first_frame = cv2.imdecode(np.frombuffer(processed_frames[0], np.uint8), cv2.IMREAD_COLOR)
                height, width, _ = first_frame.shape

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                for frame_bytes in processed_frames:
                    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                    out.write(frame)

                out.release()
                print(f"Processed video saved to {output_path}")
            except Exception as e:
                print(f"Error saving processed video: {e}")

    # Stream the processed frames
    frame_delay = 1.0 / 30  # Target 30 fps for streaming
    for frame_bytes in processed_frames:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(frame_delay)  # Proper control of playback speed


# Add these routes to your Flask app

@app.route('/check_processed_video')
def check_processed_video():
    """Check if processed video exists"""
    global video_path

    if video_path:
        processed_path = f"{video_path.rsplit('.', 1)[0]}_processed.mp4"
        if os.path.exists(processed_path):
            return jsonify({
                "status": "success",
                "video_path": processed_path
            })

    return jsonify({"status": "error", "message": "No processed video available"})


@app.route('/download_processed_video')
def download_processed_video():
    """Download the processed video file"""
    global video_path

    if video_path:
        processed_path = f"{video_path.rsplit('.', 1)[0]}_processed.mp4"
        if os.path.exists(processed_path):
            filename = os.path.basename(processed_path)
            return send_file(processed_path, as_attachment=True, attachment_filename=filename)

    return jsonify({"status": "error", "message": "No processed video available"})

@app.route('/video_feed')
def video_feed():
    """Return the video stream response"""
    source = request.args.get('source', 'webcam')

    if source == 'video' and video_path is not None:
        return Response(gen_frames_video(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif source == 'esp32':
        return Response(gen_frames_esp32(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(gen_frames_webcam(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_all_sources', methods=['POST'])
def stop_all_sources():
    """Stop all video sources"""
    global camera_active, esp32_cam_active, camera, processing_video

    camera_active = False
    esp32_cam_active = False
    processing_video = False

    if camera is not None:
        camera.release()
        camera = None

    return jsonify({"status": "success", "message": "All video sources stopped"})

@app.route('/capture', methods=['GET'])
def capture():
    """Capture a single frame for processing"""
    global frame_global

    if frame_global is not None:
        processed_frame = process_frame(frame_global.copy())

        timestamp = int(time.time())
        img_path = f'captured_frame_{timestamp}.jpg'
        cv2.imwrite(img_path, processed_frame)

        subject = "Manual Capture - License Plate Detection"
        body = f"A frame was manually captured at {time.strftime('%Y-%m-%d %H:%M:%S')}"

        email_thread = threading.Thread(
            target=send_email,
            args=(subject, body, img_path)
        )
        email_thread.start()

        return jsonify({"status": "success", "message": "Frame captured and saved"})
    else:
        return jsonify({"status": "error", "message": "No frame available to capture"})


@app.route('/')
def index():
    """Main page route"""
    return get_template()


@app.route('/connect_esp32', methods=['POST'])
def connect_esp32():
    """Connect to ESP32-CAM using provided URL"""
    global esp32_cam_url, esp32_cam_active, esp32_cam_thread, camera_active

    try:

        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"status": "error", "message": "No URL provided"})


        esp32_cam_url = data['url']

        if camera_active:
            camera_active = False
            if camera is not None:
                camera.release()

        esp32_cam_active = True
        if esp32_cam_thread is None or not esp32_cam_thread.is_alive():
            esp32_cam_thread = threading.Thread(target=esp32_cam_stream)
            esp32_cam_thread.daemon = True
            esp32_cam_thread.start()

        return jsonify({"status": "success", "message": "Connected to ESP32-CAM"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error connecting to ESP32-CAM: {str(e)}"})


@app.route('/disconnect_esp32', methods=['POST'])
def disconnect_esp32():
    """Disconnect from ESP32-CAM"""
    global esp32_cam_active, esp32_cam_url

    esp32_cam_active = False
    esp32_cam_url = None

    return jsonify({"status": "success", "message": "Disconnected from ESP32-CAM"})


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload"""
    global video_path, processing_video

    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})

    if file:

        processing_video = False

        timestamp = int(time.time())
        filename = f"uploaded_video_{timestamp}.mp4"
        file_path = os.path.join('uploads', filename)

        os.makedirs('uploads', exist_ok=True)

        file.save(file_path)
        video_path = file_path

        return jsonify({"status": "success", "message": "Video uploaded successfully"})


@app.route('/update_settings', methods=['POST'])
def update_settings():
    """Update detection settings"""
    global Imatricule, serial, code, threshold, detection_cooldown

    try:
        data = request.get_json()

        if 'imatricule' in data:
            Imatricule = data['imatricule']

        if 'serial' in data:
            serial = data['serial']

        if 'code' in data:
            code = data['code']

        if 'threshold' in data:
            threshold = float(data['threshold'])

        if 'cooldown' in data:
            detection_cooldown = int(data['cooldown'])

        return jsonify({
            "status": "success",
            "message": "Settings updated successfully",
            "settings": {
                "imatricule": Imatricule,
                "serial": serial,
                "code": code,
                "threshold": threshold,
                "cooldown": detection_cooldown
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error updating settings: {str(e)}"})


@app.route('/update_email_settings', methods=['POST'])
def update_email_settings():
    """Update email notification settings"""
    global sender_email, receiver_email, password, smtp_server, smtp_port

    try:
        data = request.get_json()

        if 'sender' in data:
            sender_email = data['sender']

        if 'receiver' in data:
            receiver_email = data['receiver']

        if 'password' in data:
            password = data['password']

        if 'smtp_server' in data:
            smtp_server = data['smtp_server']

        if 'smtp_port' in data:
            smtp_port = int(data['smtp_port'])

        test_result = send_email("Test Connection", "This is a test email to verify connection settings.")

        if test_result:
            return jsonify({"status": "success", "message": "Email settings updated and tested successfully"})
        else:
            return jsonify({"status": "warning", "message": "Settings updated but test email failed"})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error updating email settings: {str(e)}"})


@app.route('/get_current_settings', methods=['GET'])
def get_current_settings():
    """Return current detection settings"""
    return jsonify({
        "imatricule": Imatricule,
        "serial": serial,
        "code": code,
        "threshold": threshold,
        "cooldown": detection_cooldown,
        "current_source": "esp32" if esp32_cam_active else "webcam" if camera_active else "video"
    })


@app.route('/switch_source', methods=['POST'])
def switch_source():
    """Switch between webcam, ESP32-CAM, and video sources"""
    global camera_active, esp32_cam_active, camera, esp32_cam_thread

    try:
        data = request.get_json()
        source = data.get('source', 'webcam')

        camera_active = False
        esp32_cam_active = False

        if camera is not None:
            camera.release()
            camera = None


        if source == 'webcam':
            camera_active = True
            return jsonify({"status": "success", "message": "Switched to webcam"})
        elif source == 'esp32':
            if esp32_cam_url:
                esp32_cam_active = True
                if esp32_cam_thread is None or not esp32_cam_thread.is_alive():
                    esp32_cam_thread = threading.Thread(target=esp32_cam_stream)
                    esp32_cam_thread.daemon = True
                    esp32_cam_thread.start()
                return jsonify({"status": "success", "message": "Switched to ESP32-CAM"})
            else:
                return jsonify({"status": "error", "message": "ESP32-CAM URL not configured"})
        elif source == 'video':
            if video_path:
                return jsonify({"status": "success", "message": "Switched to video file"})
            else:
                return jsonify({"status": "error", "message": "No video file uploaded"})
        else:
            return jsonify({"status": "error", "message": "Invalid source specified"})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error switching source: {str(e)}"})


@app.route('/templates/index.html')
def get_template():
    """Return the HTML template for the frontend"""
    html = """
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>License Plate Detection System</title>
       <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
       <style>
           body { padding-top: 20px; }
           .video-container { position: relative; }
           .video-overlay { position: absolute; top: 10px; right: 10px; }
           .settings-container { margin-top: 20px; }
           .tab-content { padding: 20px; border: 1px solid #dee2e6; border-top: 0; }
       </style>
   </head>
   <body>
       <div class="container">
           <h1 class="text-center mb-4">License Plate Detection System</h1>

           <div class="row">
               <div class="col-md-8">
            <div class="video-container">
                <img id="videoFeed" src="/video_feed" class="img-fluid border" alt="Video Feed">
                <div class="video-controls mt-3 d-flex justify-content-between">
                    <div>
                        <button id="captureBtn" class="btn btn-primary">Capture Frame</button>
                    </div>
                    <div>
                        <button id="stopAllSources" class="btn btn-danger">Stop All Video</button>
                    </div>
                </div>
            </div>
               </div>

               <div class="col-md-4">
                   <div class="card">
                       <div class="card-header">
                           <h5>Video Source</h5>
                       </div>
                       <div class="card-body">
                           <div id="videoDownloadArea" style="display: none;" class="mt-3">
                               <a id="downloadProcessedVideo" class="btn btn-success" download>Download Processed Video</a>
                               <p class="small text-muted mt-2">The processed video is available for download.</p>
                           </div>
                           <div class="form-check">
                               <input class="form-check-input" type="radio" name="videoSource" id="webcamSource" value="webcam" checked>
                               <label class="form-check-label" for="webcamSource">Webcam</label>
                           </div>
                           <div class="form-check">
                               <input class="form-check-input" type="radio" name="videoSource" id="esp32Source" value="esp32">
                               <label class="form-check-label" for="esp32Source">ESP32-CAM</label>
                           </div>
                           <div class="form-check">
                               <input class="form-check-input" type="radio" name="videoSource" id="videoSource" value="video">
                               <label class="form-check-label" for="videoSource">Video File</label>
                           </div>

                           <div id="esp32Options" class="mt-3" style="display: none;">
                               <input type="text" id="esp32Url" class="form-control" placeholder="ESP32-CAM URL (e.g., http://192.168.1.100/stream)">
                               <button id="connectEsp32" class="btn btn-sm btn-primary mt-2">Connect</button>
                               <button id="disconnectEsp32" class="btn btn-sm btn-danger mt-2">Disconnect</button>
                           </div>

                           <div id="videoOptions" class="mt-3" style="display: none;">
                               <form id="videoUploadForm" enctype="multipart/form-data">
                                   <input type="file" id="videoFile" class="form-control" accept="video/*">
                                   <button type="submit" class="btn btn-primary mt-2">Upload Video</button>
                               </form>
                           </div>
                       </div>
                   </div>
               </div>
           </div>

           <div class="settings-container">
               <ul class="nav nav-tabs" id="settingsTabs" role="tablist">
                   <li class="nav-item" role="presentation">
                       <button class="nav-link active" id="detection-tab" data-bs-toggle="tab" data-bs-target="#detection" type="button" role="tab">Detection Settings</button>
                   </li>
                   <li class="nav-item" role="presentation">
                       <button class="nav-link" id="email-tab" data-bs-toggle="tab" data-bs-target="#email" type="button" role="tab">Email Settings</button>
                   </li>
               </ul>
               <div class="tab-content" id="settingsTabsContent">
                   <div class="tab-pane fade show active" id="detection" role="tabpanel">
                       <form id="detectionSettingsForm">
                           <div class="row">
                               <div class="col-md-4">
                                   <div class="mb-3">
                                       <label for="imatricule" class="form-label">License Plate (Imatricule)</label>
                                       <input type="text" class="form-control" id="imatricule" name="imatricule" value="2665 تونس 147">
                                   </div>
                               </div>
                               <div class="col-md-4">
                                   <div class="mb-3">
                                       <label for="serial" class="form-label">Serial Number</label>
                                       <input type="text" class="form-control" id="serial" name="serial" value="147">
                                   </div>
                               </div>
                               <div class="col-md-4">
                                   <div class="mb-3">
                                       <label for="code" class="form-label">Code</label>
                                       <input type="text" class="form-control" id="code" name="code" value="2665">
                                   </div>
                               </div>
                           </div>
                           <div class="row">
                               <div class="col-md-6">
                                   <div class="mb-3">
                                       <label for="threshold" class="form-label">Detection Threshold</label>
                                       <input type="range" class="form-range" id="threshold" name="threshold" min="0.1" max="0.9" step="0.05" value="0.25">
                                       <span id="thresholdValue">0.25</span>
                                   </div>
                               </div>
                               <div class="col-md-6">
                                   <div class="mb-3">
                                       <label for="cooldown" class="form-label">Detection Cooldown (seconds)</label>
                                       <input type="number" class="form-control" id="cooldown" name="cooldown" value="10" min="1" max="60">
                                   </div>
                               </div>
                           </div>
                           <button type="submit" class="btn btn-primary">Save Detection Settings</button>
                       </form>
                   </div>
                   <div class="tab-pane fade" id="email" role="tabpanel">
                       <form id="emailSettingsForm">
                           <div class="mb-3">
                               <label for="sender" class="form-label">Sender Email</label>
                               <input type="email" class="form-control" id="sender" name="sender" value="selim.belkhire@etudiant-enit.utm.tn">
                           </div>
                           <div class="mb-3">
                               <label for="receiver" class="form-label">Receiver Email</label>
                               <input type="email" class="form-control" id="receiver" name="receiver" value="selim.belkhire@etudiant-enit.utm.tn">
                           </div>
                           <div class="mb-3">
                               <label for="password" class="form-label">Email Password</label>
                               <input type="password" class="form-control" id="password" name="password">
                               <div class="form-text">Password is required for sending email notifications.</div>
                           </div>
                           <div class="mb-3">
                               <label for="smtp_server" class="form-label">SMTP Server</label>
                               <input type="text" class="form-control" id="smtp_server" name="smtp_server" value="smtp.gmail.com">
                           </div>
                           <div class="mb-3">
                               <label for="smtp_port" class="form-label">SMTP Port</label>
                               <input type="number" class="form-control" id="smtp_port" name="smtp_port" value="587">
                           </div>
                           <button type="submit" class="btn btn-primary">Save Email Settings</button>
                       </form>
                   </div>
               </div>
           </div>

           <div class="alert alert-info mt-4" role="alert">
               <strong>Status:</strong> <span id="statusMessage">Ready</span>
           </div>
       </div>

       <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
       <script>
           document.addEventListener('DOMContentLoaded', function() {
               // Elements
               const videoFeed = document.getElementById('videoFeed');
               const captureBtn = document.getElementById('captureBtn');
               const statusMessage = document.getElementById('statusMessage');
               const webcamSource = document.getElementById('webcamSource');
               const esp32Source = document.getElementById('esp32Source');
               const videoSource = document.getElementById('videoSource');
               const esp32Options = document.getElementById('esp32Options');
               const videoOptions = document.getElementById('videoOptions');
               const esp32Url = document.getElementById('esp32Url');
               const connectEsp32 = document.getElementById('connectEsp32');
               const disconnectEsp32 = document.getElementById('disconnectEsp32');
               const videoUploadForm = document.getElementById('videoUploadForm');
               const thresholdInput = document.getElementById('threshold');
               const thresholdValue = document.getElementById('thresholdValue');
               const detectionSettingsForm = document.getElementById('detectionSettingsForm');
               const emailSettingsForm = document.getElementById('emailSettingsForm');
               const videoDownloadArea = document.getElementById('videoDownloadArea');
               const downloadProcessedVideo = document.getElementById('downloadProcessedVideo')

               // Update threshold value display
               thresholdInput.addEventListener('input', function() {
                   thresholdValue.textContent = this.value;
               });

               // Stop all video sources button
                const stopAllSourcesBtn = document.getElementById('stopAllSources');
                stopAllSourcesBtn.addEventListener('click', function() {
                fetch('/stop_all_sources', {
                   method: 'POST',
                })
                .then(response => response.json())
                .then(data => {
                if (data.status === 'success') {
                // Display a static "video stopped" image
            videoFeed.src = 'static/no_camera.jpg';
            updateStatus('Video sources stopped', 'info');
            } else {
            updateStatus('Error: ' + data.message, 'danger');
            }
            })
             .catch(error => {
                updateStatus('Error stopping video sources: ' + error, 'danger');
            });
        });

               // Video source selection
               function updateVideoSource() {
                   if (webcamSource.checked) {
                       esp32Options.style.display = 'none';
                       videoOptions.style.display = 'none';
                       switchSource('webcam');
                   } else if (esp32Source.checked) {
                       esp32Options.style.display = 'block';
                       videoOptions.style.display = 'none';
                   } else if (videoSource.checked) {
                       esp32Options.style.display = 'none';
                       videoOptions.style.display = 'block';
                       switchSource('video');
                   }
               }

               webcamSource.addEventListener('change', updateVideoSource);
               esp32Source.addEventListener('change', updateVideoSource);
               videoSource.addEventListener('change', updateVideoSource);

               // ESP32-CAM connection
               connectEsp32.addEventListener('click', function() {
                   const url = esp32Url.value.trim();
                   if (!url) {
                       updateStatus('Error: Please enter ESP32-CAM URL', 'danger');
                       return;
                   }

                   fetch('/connect_esp32', {
                       method: 'POST',
                       headers: {
                           'Content-Type': 'application/json',
                       },
                       body: JSON.stringify({ url: url }),
                   })
                   .then(response => response.json())
                   .then(data => {
                       if (data.status === 'success') {
                           updateStatus('Connected to ESP32-CAM', 'success');
                           switchSource('esp32');
                       } else {
                           updateStatus('Error: ' + data.message, 'danger');
                       }
                   })
                   .catch(error => {
                       updateStatus('Error connecting to ESP32-CAM: ' + error, 'danger');
                   });
               });

               disconnectEsp32.addEventListener('click', function() {
                   fetch('/disconnect_esp32', {
                       method: 'POST',
                   })
                   .then(response => response.json())
                   .then(data => {
                       if (data.status === 'success') {
                           updateStatus('Disconnected from ESP32-CAM', 'success');
                           webcamSource.checked = true;
                           updateVideoSource();
                       } else {
                           updateStatus('Error: ' + data.message, 'danger');
                       }
                   })
                   .catch(error => {
                       updateStatus('Error disconnecting from ESP32-CAM: ' + error, 'danger');
                   });
               });

               // Video upload
               videoUploadForm.addEventListener('submit', function(e) {
                   e.preventDefault();
                   const fileInput = document.getElementById('videoFile');
                   const file = fileInput.files[0];

                   if (!file) {
                       updateStatus('Please select a video file', 'warning');
                       return;
                   }

                   const formData = new FormData();
                   formData.append('file', file);

                   updateStatus('Uploading video...', 'info');

                   fetch('/upload_video', {
                       method: 'POST',
                       body: formData,
                   })
                   .then(response => response.json())
                   .then(data => {
                       if (data.status === 'success') {
                           updateStatus('Video uploaded successfully', 'success');
                           switchSource('video');
                       } else {
                           updateStatus('Error: ' + data.message, 'danger');
                       }
                   })
                   .catch(error => {
                       updateStatus('Error uploading video: ' + error, 'danger');
                   });
               });

               // Switch video source
function switchSource(source) {
    fetch('/switch_source', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ source: source }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Update video feed URL with a cache-busting parameter
            videoFeed.src = '/video_feed?source=' + source + '&t=' + new Date().getTime();
            updateStatus('Switched to ' + source, 'success');
            
            // Show download button only for video source
            if (source === 'video') {
                // Check if processed video exists after a delay
                setTimeout(checkProcessedVideo, 1000);
            } else {
                videoDownloadArea.style.display = 'none';
            }
        } else {
            updateStatus('Error: ' + data.message, 'danger');
        }
    })
    .catch(error => {
        updateStatus('Error switching source: ' + error, 'danger');
    });
}
function checkProcessedVideo() {
    fetch('/check_processed_video')
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success' && data.video_path) {
            downloadProcessedVideo.href = '/download_processed_video';
            videoDownloadArea.style.display = 'block';
        } else {
            videoDownloadArea.style.display = 'none';
        }
    })
    .catch(error => {
        console.error('Error checking processed video:', error);
        videoDownloadArea.style.display = 'none';
    });
}
               // Capture frame
               captureBtn.addEventListener('click', function() {
                   fetch('/capture')
                   .then(response => response.json())
                   .then(data => {
                       if (data.status === 'success') {
                           updateStatus('Frame captured successfully', 'success');
                       } else {
                           updateStatus('Error: ' + data.message, 'danger');
                       }
                   })
                   .catch(error => {
                       updateStatus('Error capturing frame: ' + error, 'danger');
                   });
               });

               // Detection settings form
               detectionSettingsForm.addEventListener('submit', function(e) {
                   e.preventDefault();

                   const formData = {
                       imatricule: document.getElementById('imatricule').value,
                       serial: document.getElementById('serial').value,
                       code: document.getElementById('code').value,
                       threshold: document.getElementById('threshold').value,
                       cooldown: document.getElementById('cooldown').value
                   };
                   
                   fetch('/update_settings', {
                       method: 'POST',
                       headers: {
                           'Content-Type': 'application/json',
                       },
                       body: JSON.stringify(formData),
                   })
                   .then(response => response.json())
                   .then(data => {
                       if (data.status === 'success') {
                           updateStatus('Detection settings updated successfully', 'success');
                       } else {
                           updateStatus('Error: ' + data.message, 'danger');
                       }
                   })
                   .catch(error => {
                       updateStatus('Error updating settings: ' + error, 'danger');
                   });
               });
               
               // Email settings form
               emailSettingsForm.addEventListener('submit', function(e) {
                   e.preventDefault();
                   
                   const formData = {
                       sender: document.getElementById('sender').value,
                       receiver: document.getElementById('receiver').value,
                       password: document.getElementById('password').value,
                       smtp_server: document.getElementById('smtp_server').value,
                       smtp_port: document.getElementById('smtp_port').value
                   };
                   
                   updateStatus('Updating email settings...', 'info');
                   
                   fetch('/update_email_settings', {
                       method: 'POST',
                       headers: {
                           'Content-Type': 'application/json',
                       },
                       body: JSON.stringify(formData),
                   })
                   .then(response => response.json())
                   .then(data => {
                       if (data.status === 'success') {
                           updateStatus('Email settings updated successfully', 'success');
                       } else if (data.status === 'warning') {
                           updateStatus(data.message, 'warning');
                       } else {
                           updateStatus('Error: ' + data.message, 'danger');
                       }
                   })
                   .catch(error => {
                       updateStatus('Error updating email settings: ' + error, 'danger');
                   });
               });
               
               // Helper function to update status message
               function updateStatus(message, type) {
                   statusMessage.textContent = message;
                   const alertElement = document.querySelector('.alert');
                   alertElement.className = 'alert mt-4 alert-' + (type || 'info');
                   
                   // Auto-clear success messages after 5 seconds
                   if (type === 'success') {
                       setTimeout(() => {
                           statusMessage.textContent = 'Ready';
                           alertElement.className = 'alert mt-4 alert-info';
                       }, 5000);
                   }
               }
               
               // Load current settings on page load
               fetch('/get_current_settings')
               .then(response => response.json())
               .then(data => {
                   document.getElementById('imatricule').value = data.imatricule;
                   document.getElementById('serial').value = data.serial;
                   document.getElementById('code').value = data.code;
                   document.getElementById('threshold').value = data.threshold;
                   document.getElementById('cooldown').value = data.cooldown;
                   thresholdValue.textContent = data.threshold;
                   
                   // Set current source
                   if (data.current_source === 'esp32') {
                       esp32Source.checked = true;
                       updateVideoSource();
                   } else if (data.current_source === 'video') {
                       videoSource.checked = true;
                       updateVideoSource();
                   } else {
                       webcamSource.checked = true;
                       updateVideoSource();
                   }
               })
               .catch(error => {
                   console.error('Error loading settings:', error);
               });
           });
       </script>
   </body>
   </html>
   """
    return html


if __name__ == '__main__':
   os.makedirs('uploads', exist_ok=True)
   os.makedirs('static', exist_ok=True)

   if not os.path.exists('static/no_camera.jpg'):
       placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 200
       cv2.putText(placeholder, "No Camera Available", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
       cv2.imwrite('static/no_camera.jpg', placeholder)

   if not os.path.exists('static/no_video.jpg'):
       placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 200
       cv2.putText(placeholder, "No Video Selected", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
       cv2.imwrite('static/no_video.jpg', placeholder)

   if not os.path.exists('static/error.jpg'):
       placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 200
       cv2.putText(placeholder, "Error Loading Video", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
       cv2.imwrite('static/error.jpg', placeholder)

   if not os.path.exists('static/waiting.jpg'):
       placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 200
       cv2.putText(placeholder, "Waiting for ESP32-CAM...", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
       cv2.imwrite('static/waiting.jpg', placeholder)

   app.run(host='0.0.0.0', port=5000, debug=True)
