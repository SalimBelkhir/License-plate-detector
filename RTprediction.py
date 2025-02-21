import cv2
from ultralytics import YOLO
import easyocr
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
# Load YOLO model
model_path = './runs/detect/train11/weights/last.pt'
model = YOLO(model_path)

Imatricule = "2665 تونس 147"
serial='147'
code ='2665'

reader = easyocr.Reader(['ar', 'en'])  # Add 'ar' for Arabic text recognition & 'en' is for english


threshold = 0.25
detected_text = None
match_found = False

sender_email = "selim.belkhire@etudiant-enit.utm.tn"
receiver_email = "selim.belkhire@etudiant-enit.utm.tn"
password = "tbtlxuqrcdaevnem"
smtp_server = "smtp.gmail.com"
smtp_port = 587

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def send_email(subject, body , img_path = None):
    try:
        # Create the email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if img_path :
            with open(img_path , 'rb') as f :
                data = f.read()
            image = MIMEImage(data ,name = 'detected_frame.jpg')
            msg.attach(image)

        # Connect to the SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, password)  # Log in to the email account
            server.sendmail(sender_email, receiver_email, msg.as_string())  # Send the email
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform YOLO detection
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            label = results.names[int(class_id)].upper()
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            license_plate_region = frame[int(y1):int(y2), int(x1):int(x2)]

            gray = cv2.cvtColor(license_plate_region, cv2.COLOR_BGR2GRAY)

            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # Resize for better OCR accuracy
            resized = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)


            ocr_results = reader.readtext(resized)

            # Display the OCR results
            if ocr_results:
                for (bbox, text, prob) in ocr_results:
                    if prob > 0.2:
                        print(f"Raw OCR Output: {text}")
                        # Replace any sequence of "?" with "تونس" using regex
                        text = re.sub(r"\?+", "تونس", text)
                        print(f"Processed Text: {text}")
                        if serial in text and code in text:
                            print("match found !")
                            match_found = True
                            detected_text = text
                            cv2.imwrite('detected_frame.jpg', frame)
                            break
                        cv2.putText(frame, text, (int(x1), int(y2 + 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                print("No text detected in the license plate region.")


    cv2.imshow('Real-Time License Plate Detection', frame)
    if match_found:
        break

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q') or match_found:
        break


cap.release()
cv2.destroyAllWindows()

if match_found :
    print(f"detected text: {detected_text}")
    subject = "License Plate Match Found"
    body = (f"Your license plate was detected: {Imatricule} \n (serial number = {serial} and code = {code} ) \n"
            f"the following detected liscence plate : {detected_text}")
    send_email(subject, body,img_path='detected_frame.jpg')