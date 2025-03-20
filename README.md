# License Plate Recognition System

A Flask-based web application for real-time license plate detection and recognition using YOLOv8 and EasyOCR, with email notifications when specific plates are detected.

## Features

- Multiple video sources:
  - Built-in webcam
  - ESP32-CAM integration
  - Video file upload and processing
- License plate recognition using YOLO object detection and OCR
- Automated email notifications when target license plates are detected
- Configurable detection parameters (threshold, cooldown)
- Customizable email settings
- Processed video download option
- User-friendly web interface

## Requirements

- Python 3.7+
- Flask
- OpenCV (cv2)
- Ultralytics YOLO
- EasyOCR
- NumPy
- Requests

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/license-plate-recognition.git
   cd license-plate-recognition
   ```

2. Install dependencies:
   ```
   pip install flask opencv-python ultralytics easyocr numpy requests
   ```

3. Make sure you have a trained YOLO model for license plate detection. The code is configured to use a model at `./runs/detect/train11/weights/last.pt`

## Usage

1. Start the server:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. The web interface allows you to:
   - Switch between webcam, ESP32-CAM, and video file sources
   - Configure license plate detection parameters
   - Set up email notifications
   - Capture frames manually
   - Upload and process video files

## Configuration

### Detection Settings

- **Imatricule**: The license plate number to detect (e.g., "2665 تونس 147")
- **Serial**: The serial number part of the license plate (e.g., "147")
- **Code**: The code part of the license plate (e.g., "2665")
- **Threshold**: Detection confidence threshold (0.0-1.0)
- **Cooldown**: Time between detections in seconds

### Email Settings

- **Sender Email**: Your email address
- **Receiver Email**: Email address to receive notifications
- **Password**: App password for the sender email
- **SMTP Server**: Mail server (e.g., smtp.gmail.com)
- **SMTP Port**: Mail server port (e.g., 587)

## ESP32-CAM Integration

To use an ESP32-CAM:
1. Select "ESP32-CAM" as the video source
2. Enter the camera's URL (e.g., `http://192.168.1.100/stream` or `http://192.168.1.100/capture`)
3. Click "Connect"

## Video Processing

To process a video file:
1. Select "Video File" as the video source
2. Upload a video file
3. Once processing is complete, you can download the processed video

## Custom YOLO Model

This application uses a custom-trained YOLOv8 model for license plate detection. If you want to use your own model:
1. Train a YOLOv8 model on license plate data
2. Update the `model_path` variable in the code to point to your model

## Arabic Language Support

The system supports Arabic text recognition for license plates, using EasyOCR with Arabic and English language packages.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
