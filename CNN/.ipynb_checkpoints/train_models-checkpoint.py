import cv2
import os
import numpy as np
from PIL import Image,ImageOps
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "training_data")

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

current_label = 0
labels = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root)

            if label not in labels:
                labels[label] = current_label
                current_label += 1
            
            #Open image as gray and resize 
            image = cv2.imread(path)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray, (240, 320), interpolation=cv2.INTER_AREA)
            final_image = np.array(resized_img,'uint8')

            faces = face_cascade.detectMultiScale(final_image,scaleFactor=1.05,minNeighbors=5)
            print(label)

            for (x,y,w,h) in faces:
                face_crop = final_image[y:y+h, x:x+w]
                print(face_crop.shape)
                face_crop = cv2.resize(face_crop, (80, 80), interpolation=cv2.INTER_AREA)
                x_train.append(face_crop)
                y_labels.append(labels[label])

with open("face_labels.p",'wb') as f:
    pickle.dump(labels,f)

LBPHS = cv2.face.LBPHFaceRecognizer_create()
LBPHS.train(x_train, np.array(y_labels))
LBPHS.save("models/LBPHF.yml")

EigenFace=cv2.face.EigenFaceRecognizer_create()
EigenFace.train(x_train, np.array(y_labels))
EigenFace.write('models/Eigen.yml')

Fisher = cv2.face.FisherFaceRecognizer_create()
Fisher.train(x_train, np.array(y_labels))
Fisher.save("models/Fisher.yml")

