import face_recognition
import cv2
import numpy as np
import os

path = "./train/"

knownImages = []
knownNames = []

imagesList = os.listdir(path)

for singleImg in imagesList:
    try:
        curImg = cv2.imread(f'{path}/{singleImg}')
        knownImages.append(curImg)
        knownNames.append(os.path.splitext(singleImg)[0])
    except:
        print("Something went wrong in appending names.")
print(knownNames)

def findEncodings(images):
    encodeList = []
    for i in range(len(knownImages)):
        try:
            img = knownImages[i]
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except:
            print(knownNames[i])
    return encodeList

encodeListKnown = findEncodings(knownImages)
print("Encoding Complete")

def imageRecognition():
    try:
        test_img_path = "./test/"
        imgList = os.listdir(test_img_path)
        i = 1
        testImages = []
        for singleImg in imgList:
            testImages.append(singleImg)
            print(f'{i}. {singleImg}')
            i+=1

        selected = int(input("Enter an image number from the above list:"))

        testImage = cv2.imread(f'{test_img_path}/{testImages[selected-1]}')

        face_locations = face_recognition.face_locations(testImage)
        face_encodings = face_recognition.face_encodings(testImage, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
            name = ""

            face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
            best_match = np.argmin(face_distances)

            if matches[best_match]:
                name = knownNames[best_match]
                name = name.split("_")[0].capitalize()
            else:
                name = "Unknown"

            cv2.rectangle(testImage, (left, top), (right, bottom), (0,255,0), 2)
            cv2.rectangle(testImage, (left, bottom - 35), (right, bottom), (0,255,0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(testImage, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2)

        cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
        testImage = cv2.resize(testImage,(1000,580))
        cv2.imshow("Result", testImage)
        
        cv2.waitKey(0)
        cond = input("Do you want to recognize anymore? (y/anykey)")
        cv2.destroyAllWindows()
        if cond == "y":
            imageRecognition()
        else:
            Menu()
    except Exception as e:
        print(e)
    
    

def liveRecognition(encodeListKnown,knownNames):
    capture = cv2.VideoCapture(0)
    print("Press 'x' to go to the Menu.")

    while True:
        try:
            success, img = capture.read()
            imgS = cv2.resize(img,(0,0),None,0.25,0.25)
            imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

            for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                try:
                    matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name = knownNames[matchIndex]
                        name = name.split("_")[0].capitalize()
                        y1,x2,y2,x1 = faceLoc
                        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                    else:
                        name = "Unknown"
                        y1,x2,y2,x1 = faceLoc
                        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                except:
                    print("Something went wrong")

            cv2.namedWindow("Web Camera",cv2.WINDOW_NORMAL)
            cv2.imshow('Web Camera',img)
            key = cv2.waitKey(1)

            if key == ord('x'):
                capture.release()
                print("Camera off.")
                cv2.destroyAllWindows()
                Menu()
                break
        except:
            print("Something went wrong")

def Menu():
    print("Face Recognition Menu.....................\nEnter 1 to recognize from specific Image.\nEnter 2 to recognize from live Camera.\nEnter 0 to exit the program.")
    optionKey = int(input("Enter your choice:"))
    if optionKey == 1:
        imageRecognition()
    if optionKey == 2:
        liveRecognition(encodeListKnown,knownNames)
    if optionKey == 0:
        print("Program ended.")

Menu()
