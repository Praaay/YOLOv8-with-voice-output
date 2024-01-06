import cv2
import pyttsx3
from ultralytics import YOLO

# text to speech
text_speech = pyttsx3.init()

# model
model = YOLO("yolov8s.pt")

# resolution (640 x 480) webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    
    # Inference
    results = model.predict(img, conf=0.7) # only conf > 0.7 will output
    
    # Results
    print(results[0].boxes.data)

    if len(results[0].boxes.data) != 0:
        for box in results[0].boxes.data:
            x1, y1, x2, y2, score, label = box
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=3)
            cv2.putText(img, f"{model.names[int(label)].upper()} {score:.2f}", (int(x1)+10, int(y1)+20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Detection Window", img)
            cv2.waitKey(2)
            answer = model.names[int(label)] # the output
            
            newVoiceRate = 160 # 160 wpm
            text_speech.setProperty('rate', newVoiceRate)
            text_speech.say('Detects ' + answer)
            
            text_speech.runAndWait()
