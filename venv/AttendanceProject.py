import cv2
import numpy as np
import face_recognition
import os
# from datetime import datetime
# from PIL import ImageGrab
 
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path) # resimleri os modülündeki listdir fonksiyonu kullanılarak myList değişkenine atanır.
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
# 'ImagesAttendance' klasöründeki her bir resim dosyasını okuyarak bu resimlerin sınıf adlarını ve resimlerin kendisini içeren bir liste oluşturur.

def findEncodings(images):                                    # yüz kodlamalarını hesaplama.
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
#'images' adlı bir resim listesi alarak, her bir resimdeki yüzlerin özelliklerini (yüz kodlamalarını) kodlamak için kullanılır.


# yoklama sistemi(Belki kullanılacak)
# def markAttendance(name):
#     with open('Attendance.csv','r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'n{name},{dtString}')
 
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    # kare kare görüntüleri okuyarak yüz tanıma işlemlerini gerçekleştirir.

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):            # her bir yüz için eşleştirme işlemi yapılır.
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace) # Bool değişkeni
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) # yüzlerin birbirine olan benzerlik dercesi
        #print(faceDis)
        matchIndex = np.argmin(faceDis) 
 
        if faceDis[matchIndex]< 0.50:
            name = classNames[matchIndex].upper()
        else: name = 'Unknown'
            #print(name)
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        # Bir görüntü üzerinde yüz tespiti yapılmış bir bölgeyi çerçevelemek, 
        # bir isim etiketi koymak ve çerçevenin altına bir arka plan kutusu çizmek için kullanılır.
        # Kod satırları, yüz tespitinden elde edilen yüz konumunu belirlemek için bir değişken olan "faceLoc" kullanır. Bu değişken, 
        # dört değer içeren bir demet olarak tanımlanır: y1, x2, y2, x1. Bu değerler, 
        # yüzün sol üst ve sağ alt köşelerinin koordinatlarını içerir.
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)