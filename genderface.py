import argparse
import face_recognition
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import time
import math
import argparse


input_video = cv2.VideoCapture("den.mp4")
length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
output_video = cv2.VideoWriter('output_478.avi', fourcc, 30.57, (640, 352))

def getFaceBox(net, frame, conf_threshold=0.7):

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frame, bboxes

parser = argparse.ArgumentParser(description='Use this script to run gender recognition using OpenCV.')
parser.add_argument('--input',
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
padding = 20
args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

genderList = ['Male', 'Female']

genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

birinci_image = face_recognition.load_image_file("cagatay.jpeg")
birinci_face_encoding = face_recognition.face_encodings(birinci_image)[0]

ikinci_image = face_recognition.load_image_file("metinn.jpeg")
ikinci_face_encoding = face_recognition.face_encodings(ikinci_image)[0]

known_faces = [
    birinci_face_encoding,
    ikinci_face_encoding,
]

face_locations = []
face_encodings = []
face_names = []
frame_number = 0
cgt_frame=0
metin_frame=0

kac_kisi_var=[]
kac_kisi_var2=[]
while True:
    ret, frame = input_video.read()
    frame_number += 1

    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]


    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
       print("No face Detected, Checking next frame")
       continue

    for bbox in bboxes:
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        label = "{}".format(gender)


    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if(len(face_locations)>=1):
        kac_kisi_var.append((len(face_locations)))

    else:
        kac_kisi_var.append(0)
    face_names = []
    for face_encoding in face_encodings:

        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)


        name = None
        if match[0]:
            name = "cagatay"
            cgt_frame+=1
        elif match[1]:
            name = "metin"
            metin_frame+=1

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name+" , "+label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        saniye_bilgi=str(int(frame_number/30.47))
        bilgi="Videonun "+saniye_bilgi+". saniyesinde"+str(len(face_locations))+" kisi var"

        cv2.putText(frame,bilgi,(1, 342), font,0.5,(255, 255, 255), 1)

    print("Writing frame {} / {}".format(frame_number, length))
    output_video.write(frame)


k=0
for i in kac_kisi_var:
    k+=1
    if(k&30==0):
        kac_kisi_var2.append(i)

cgt_info=str(int(cgt_frame/30.24))
metin_info=str(int(metin_frame/30.24))

print("cagatay bu videoda "+cgt_info+" saniye kalmıştır.")
print("metin bu videoda "+metin_info+" saniye kalmıştır.")

fig_size = [16,9]
plt.figure(figsize=fig_size)
x=np.linspace(0,12,len(kac_kisi_var2))
y=kac_kisi_var2
print(y)
plt.plot(x,y)
plt.ylabel("kisi sayisi")
# x eksenine isim ekliyoruz
plt.xlabel("zaman(sn)")
# grafiğe başlık ekliyoruz
plt.title("kisi sayisi analizi");
# ızgara ekliyoruz
plt.grid()
plt.show();
input_video.release()
cv2.destroyAllWindows()
