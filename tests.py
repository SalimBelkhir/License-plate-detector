import torch
import ultralytics
print(torch.__version__)

'''from roboflow import Roboflow

rf = Roboflow(api_key="a9LNoItRCV1FwPChD8OE")
project = rf.workspace().project("liscence-plate-detector")
model = project.version("1").model

job_id, signed_url, expire_time = model.predict_video(
    "KrahbTwensa.mp4",
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

print(results)'''

from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt

# Initialize Roboflow and access the model
# rf = Roboflow(api_key="a9LNoItRCV1FwPChD8OE")
# project = rf.workspace().project("liscence-plate-detector")
# model = project.version("1").model

cap = cv2.VideoCapture(0)
while 1 :
    ret , frame = cap.read( )
    cv2.imshow('video stream',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow()






