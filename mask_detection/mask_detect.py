import cv2
import numpy as np
from tensorflow import keras
from tensorflow_hub import KerasLayer

cam = cv2.VideoCapture(0)

cam.set(3, 960) # width 
cam.set(4, 540)

model = keras.models.load_model(r"F:\Tutorials_1\opencv_new_series\Mask_detection\classifier.h5", custom_objects={'KerasLayer':KerasLayer})
maskDetect = cv2.CascadeClassifier(r"F:\Tutorials_1\opencv_new_series\Mask_detection\haarcascade_frontalface_default.xml")

COLOR = [(0, 255, 0), (0, 0, 255), (255, 255, 255), (255, 255, 0)]
MSG = [" YES ", " NO "]
SIZE = (224, 224)

thickness = 2
p, n = None, None


def preprocess(img):
    img = cv2.resize(img, SIZE)
    img = img[np.newaxis, ...]
    return img


while True:

    hascap, img = cam.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     

    faces = maskDetect.detectMultiScale(imgGray, 1.089, 6)

    if len(faces):

        for (x,y,w,h) in faces:
            area = w*h
            if area > 9000 :
                
                face = img[y:y+h, x:x+w]
                img_mat = preprocess(face)

                res = model.predict(img_mat)
                c = np.argmax(res)
                p = round(np.max(res)*100, 2)

                cv2.rectangle(img, (x,y), (x+w, y+h), COLOR[c], thickness)
                cv2.rectangle(img,(x-1,y-40),(x+w+1,y),COLOR[c], -1)
                cv2.putText(img, MSG[c], (x,y-5), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR[2], 2)
            
            
                
                

    display1 = f"FACES : {len(faces)}"
    display2 = f"PROBABILITY : {p} %"
    
    cv2.putText(img, display1, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR[3], 2)
    cv2.putText(img, display2, (250, 40), cv2.FONT_HERSHEY_DUPLEX, 1, COLOR[3], 2)

    cv2.imshow("Face Mask Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


