import argparse
import os
from flask import Flask
import face_recognition
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import time
import math
import argparse
from keras.preprocessing import image
from imutils import face_utils
from threading import Thread
import dlib
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
# input video dosyasını al
input_video = cv2.VideoCapture("video.mp4")
length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
app = Flask(__name__)
@app.route("/")
def get():
 return "Index!"
if __name__ == '__main__':
 (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 if int(major_ver) < 3:
    fps = input_video.get(cv2.cv.CV_CAP_PROP_FPS)

    print( "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

 else:
    fps = input_video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
width = input_video.get(3) 
height = input_video.get(4)
# output videosu oluştur.Resim kare hızı, genişlik, yükseklik input video ile aynı olmalı.
zaman=time.asctime().replace(" ", "_")
fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
output_video = cv2.VideoWriter("/home/demet/PycharmProjects/pazar2/outputs/"+zaman+'.avi', fourcc, fps, (int(width), int(height)))
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[9]) 
    B = dist.euclidean(mouth[4], mouth[7]) 
    C = dist.euclidean(mouth[0], mouth[6]) 
    mar = (A + B) / (2.0 * C)
    return mar
MOUTH_AR_THRESH = 0.5

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

from keras.models import model_from_json

model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') 
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
parser = argparse.ArgumentParser(description='Use this script to run gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
padding = 20
args = parser.parse_args()
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat', help="path to facial landmark predictor")
ap.add_argument("-v", "--input", default="den.mp4", help="video path input")
args = vars(ap.parse_args())
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
(mStart, mEnd) = (49, 68) # ağız bulunan noktaların başlangıç ve bitişi
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)
# Bazı örnek resimler yükleyin.
known_faces_encodings = []
known_faces_names = []
for dosya in os.listdir("dosyalar"):
    imagge = face_recognition.load_image_file("/home/demet/PycharmProjects/pazar2/dosyalar/"+dosya)
    face_encoding = face_recognition.face_encodings(imagge)[0]
    known_faces_encodings.append(face_encoding)
    known_faces_names.append(dosya.split(".")[0])
# boş sözlük ve listeleri oluştur
face_locations = []
face_encodings = []
face_names = []
frame_number = 0
kac_kisi_var=[]
kac_kisi_var2=[]
kisiler=[]
frmsys={}
duygu={}
konusma={}
info={}
belgeler = open("/home/demet/PycharmProjects/pazar2/outputs/txtdosyalari/"+zaman + "duygular" + ".txt", "w")
belgeler.write("isim" + "," + "information" + "," + "framesayisi" + "," +"duygu"+ "," + "konusmasuresi\n")
while True:
    ret, frame = input_video.read()
    frame_number += 1
    #input video bitince çık.
    if not ret:
        break
    rgb_frame = frame[:, :, ::-1]
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("yüz tespit edilmedi")
        continue
    for bbox in bboxes:
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        label = "{}".format(gender)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        detected_face = frame[int(y):int(y + h), int(x):int(x + w)] # bulunan yüzü kes
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) # gray scale yap
        detected_face = cv2.resize(detected_face, (48, 48)) # yeniden boyutlandır48x48
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255 #normalize et
        predictions = model.predict(img_pixels) 
        # maksimumu bul 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # ağızı bul 
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        # ağızı görselleştir
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if(len(face_locations)>=1):
        kac_kisi_var.append((len(face_locations)))
    else:
        kac_kisi_var.append(0)
    face_names = []
    for face_encoding in face_encodings:
 # yüzler eşleşiyor mu kontrol et
        match = face_recognition.compare_faces(known_faces_encodings, face_encoding, tolerance=0.50)
        name = None
        i=0
        for m in match:
            if m:
                name=known_faces_names[i]
                if name in frmsys:
                    frmsys[name] = frmsys[name] + 1
                    info[name]=str(round(frmsys[name]/fps,2))
                    belgeler.write(str(name)+","+str(info[name])+","+str(frmsys[name]))
                else:
                    frmsys[name]=0
                    frmsys[name]=frmsys[name]+1
                    info[name] = str(round(frmsys[name] / fps, 2))
                    belgeler.write(str(name) + "," + str(info[name]) + "," + str(frmsys[name]))
                if name in duygu:
                    duygu[name] = duygu[name]+[emotion]
                    belgeler.write(","+str(emotion))
                else:
                    duygu[name]=[]
                    kisiler.append(name)
                    duygu[name] = duygu[name]+[emotion]
                    belgeler.write("," + str(emotion))
                if mar > MOUTH_AR_THRESH:
                    if name in konusma:
                        konusma[name] = konusma[name] + 1
                        belgeler.write("," +str(round(konusma[name]/ fps,2))+"\n" )
                    else:
                        konusma[name]=0
                        konusma[name]=konusma[name]+1
                        belgeler.write("," + str(round(konusma[name]/ fps,2)) + "\n")
                else:
                    if name in konusma:
                        belgeler.write("," + str(round(konusma[name]/ fps,2)) + "\n")
                    else:
                        belgeler.write("," + "0" + "\n")
            i+=1
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name+" , "+label+" , "+ emotion, (left + 6, bottom - 6), font, 0.5,(255, 255, 255), 1)
        saniye_bilgi=str(int(frame_number/30.47))
        bilgi="Videonun "+saniye_bilgi+". saniyesinde"+str(len(face_locations))+" kisi var"
        cv2.putText(frame,bilgi,(1, 342), font,0.5,(255, 255, 255), 1)
    print("Writing frame {} / {}".format(frame_number, length))
    output_video.write(frame)
#videoda o saniyede kaç kişi olduğunu hesapla
k=0
for i in kac_kisi_var:
    k+=1
    if(k%int(fps)==0):
        kac_kisi_var2.append(i)
print("kim kaç saniye kaldı analizi:")
print(info)
print("duygu analizi:")
print(duygu)
print("toplantıya katılan toplam kişi sayısı "+str(len(info)))
for d in kisiler:
    print(d)
    print(info[d])
print(duygu[kisiler[0]])
print(len(duygu[kisiler[0]]))
c=0
for k in kisiler:
    c+=1
    plt.figure(int(c))

    x=np.linspace(0,int(frame_number/fps),len(duygu[k]))
    y=duygu[k]
    plt.plot(x,y)
    plt.ylabel("duygu durumu")
    # x eksenine isim ekliyoruz
    plt.xlabel("zaman(sn)")
    # grafiğe başlık ekliyoruz
    plt.title(k+" duygu analizi");
    plt.grid()
    plt.show();
belgeler.close()
#grafik boyutu
plt.figure(int(c+1))
x=np.linspace(0,int(frame_number/fps),len(kac_kisi_var2))
y=kac_kisi_var2
print(y)
plt.plot(x,y)
#y eksenine isim ekliyoruz
plt.ylabel("kisi sayisi")
# x eksenine isim ekliyoruz
plt.xlabel("zaman(sn)")
# grafiğe başlık ekliyoruz
plt.title("kisi sayisi analizi");
plt.grid()
plt.show();
input_video.release()
cv2.destroyAllWindows()
