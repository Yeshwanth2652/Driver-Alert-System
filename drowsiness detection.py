import vonage
import cv2
from keras.models import load_model
import os
import numpy as np
from pygame import mixer
import time


client = vonage.Client(key="a92541d9", secret="V33Ao1ncWNc2k6Kg")
sms = vonage.Sms(client)

#Load CNN model for eye state classification
model = load_model('driver_state.h5')
path = os.getcwd()
#Initialize the video capture devicew
cap = cv2.VideoCapture(0)
#used to define font
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
lbl=['Close','Open']
c=0
s=0
t=2
#Load the pre-trained Haar cascade classifiers for detecting faces and eyes from their XML files.
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')
r=[99]
l=[99]
#Initialize sound from pygame
mixer.init()
alert = mixer.Sound('alert.wav')
#Start an infinite loop to continuously capture frames from the webcam.
while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]                                                                                  
#Convert the frame from BGR to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    start = time.time()

    # Detect faces and eyes using Haar Cascade
    faces = face.detectMultiScale(gray, minNeighbors=10, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    end = time.time()
    time_taken = end - start
    print(f"Time taken for Haar Cascade detection: {time_taken:.6f} seconds")

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        c= c + 1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        r = model.predict(r_eye)
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        c= c + 1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        l = model.predict(l_eye)

        break
#If the eyes are closed, increase the score and display "Closed" text on the frame.
    if np.any(r[0] < 0.01) and np.any(l[0] < 0.01):
        s = s + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        #If the eyes are open, decrease the score and display "Open" text on the frame.
    else:
        s = s - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if(s<0):
        s=0
    cv2.putText(frame,'Score:' + str(s), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if(s>15):
        #person is feeling sleepy so we beep the alarm
        # cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        # responseData = sms.send_message(
        #     {
        #         "from": "Vonage APIs",
        #         "to": "916380720153",
        #         "text": "driver is sleepy",
        #     }
        # )
        #
        # if responseData["messages"][0]["status"] == "0":
        #     print("Message sent successfully.")
        # else:
        #     print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
        try:
            alert.play()

        except:  # isplaying = False
            pass
        if(t<16):
            t= t + 2
        else:
            t= t - 2
            if(t<2):
                t=2
        cv2.rectangle(frame, (0,0), (width,height), (0,0,255), t)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
