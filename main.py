from scipy.spatial import distance as dist
import cv2
import numpy as np
import pickle
from keras.models import load_model
from twilio.rest import Client 

def WriteVideo(write_path, frames, fps):
    if len(frames) == 0:
        raise Exception("frames array cannot be 0")

    h, w, l = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_vid = cv2.VideoWriter(write_path, fourcc, fps, (w, h))

    for frame in frames:
        out_vid.write(frame)

    out_vid.release()

account_sid = 'secret'
auth_token = 'token' 
client = Client(account_sid, auth_token)

model=load_model("VGG19/model3-006.model")


labels_dict={0:'mask', 1:'without mask'}
color_dict={0:(0,255,0), 1:(0,0,255), 2: (128, 0, 128)}

size = 2
webcam = cv2.VideoCapture(0) #Use camera 0
classifier = cv2.CascadeClassifier('haar_data/haarcascade_frontalface_alt2.xml')


#Recognizer Models
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("VGG19/models/LBPHF.yml")

unmasked_recognizer = cv2.face.LBPHFaceRecognizer_create()
unmasked_recognizer.read('models/LBPH.yml')

# dict initializations
labels = {}
scores = {}
time_in_violation = {}
total_time = {}

with open("VGG19/mask_face_labels.p",'rb') as f:
    name_labels = pickle.load(f)
    for key, value in name_labels.items():
        labels[value] = key
        scores[value] = 100
        time_in_violation[value] = 0
        total_time[value] = 0

print(labels)

(rval, im) = webcam.read()

frame_width = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
half_frame_width = frame_width // 2

resolution = frame_width * frame_height

f_length = 696

Cf = 30

print(f'cam res: {frame_width}, {frame_height}')

frames = []

while True:
    (rval, im) = webcam.read()
    # im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini, scaleFactor=1.04,minNeighbors=3)

    # Draw rectangles around each face
    for f_index, f in enumerate(faces):
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(128,128))
        gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,128,128,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)

        z = 6 * f_length / w

        mask_label=np.argmax(result,axis=1)[0]

        
        # draw rectangular filled box as text background
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[mask_label],-1)

        # determine if person is violating social distancing restrictions with
        # respect to the camera
        in_violation = mask_label == 1 and z < 48

        if in_violation:
            cv2.rectangle(im, (x,y), (x+w,y+h), color_dict[2], 2)
        else:
            cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[mask_label],2)

        
        label,conf = recognizer.predict(gray)
        if (label > 2):
            label = 2

        # display social score at the bottom of bounding box for known people
        
        if labels[label] != 'other':
            total_time[label] += 1
            if in_violation:
                time_in_violation[label] += 1
                cv2.rectangle(im,(x,y+h),(x+w,y+h+40),color_dict[2],-1)

            scores[label] = 100 - time_in_violation[label] * 100 / total_time[label]
            cv2.rectangle(im,(x,y+h),(x+w,y+h+40),color_dict[mask_label],-1)
            cv2.putText(im, f"score:{scores[label]:.2f}", (x,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if conf <= 130:
            if in_violation:
                cv2.putText(im, f"STAND BACK {labels[label]}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                message = client.messages.create(  
                              messaging_service_sid='MGdffe67f123ee87804ccaf812cad3448b', 
                              body='You have violated Social Distancing Guidelines. Please Wear your Bask',      
                              to='+13124592527' 
                          ) 
            else:
                cv2.putText(im, f"{labels_dict[mask_label]}: {labels[label]}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            cv2.putText(im, labels_dict[mask_label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    # Show the image
    # frames.append(im)
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27: #The Esc key
        break
    elif key == ord('c'):
        print(w)
# Stop video
webcam.release()

# WriteVideo('recording.avi', frames, 15)

# Close all started windows
cv2.destroyAllWindows()

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

