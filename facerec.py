# this program will capture persons face using Haar Cascade and convert it to a Local Binary Pattern Histogram and search for a close match on the dataset.
#calling cv2 library
import cv2
#calling cv2.face library for createLBPHFaceRecognizer()/.load
import cv2.face
#calling numpy for array
import numpy as np
# detect face using Haar Cascade. calling xml classifier
faceDetect = cv2.CascadeClassifier('C://Users//leo//Downloads//haarcascade_frontalface_default.xml')
# Selecting a video capture 0 = main camera, 1,2... are for other camera install
cam=cv2.VideoCapture(0);
# createLBPHFaceRecognizer = recognizer
recognizer = cv2.face.createLBPHFaceRecognizer();
# load dataset from file location.
recognizer.load('C://Users//leo//Documents//save//database//test.data')
# id number define/ use to get id from dataset and search it in txt database using python search function
id = 0
# while camera is running and user didnt break the loop
while True:
    # read image data from webcam
    ret, img = cam.read()
    # get gray image conversion from opencv/convert BGR to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # face detectMultiScale (model fix size gray = covert gray image for Cascade,
    #1.3 distance for detection the smaller the number the closer you have to be for detection
    #5 = sensitivate, higher the number the lower the sensativity for face detection.
    face = faceDetect.detectMultiScale(gray, 1.3, 5)
    # for the x axis, y axis, width and hight.
    for (x,y,w,h) in face:
        # create a rectangle (img = wecam, get x y of face detect and, get the width and hight
        # (255,0,0) = color of rectangle
        # 2 = thickness of the rectangle.higher = thicker
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        # get prediction of the detection face (convert to gray with the detection x,y) in the dataset
        # there is no confidents level (conf) with id so it will search for the closes related image in the dataset
        #it will convert the face to gray then to a local binary pattern histogram then compare with the ones in the dataset
        #it will only find the closes one so if you show an image of someone not in the dataset it will search for the closes relationship
        
        id=recognizer.predict(gray[y:y+h,x:x+w])
        #get name using the id number save in the dataset. I was not able to have opencv to train and save the string of the file.
        #I was not able to find a solution but, I beleive that there isnt any function for numpy to array a list of strings by abc order.
        
        search = open("C://Users//leo//Documents//save//database//test.txt")
        for line in search:
            if str(id) in line:
                # if id is found print to image using font below
               font = cv2.FONT_HERSHEY_SIMPLEX
               # input text to image (name of img window,string(line), location in image(100,400), font from above, font size, color, thickness, lineType)
               cv2.putText(img, line , (100, 400), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    # display image
    cv2.imshow("Face", img);
    # if user press the 'q'd button, break loop
    if(cv2.waitKey(1) == ord('q')):
        break;
    # safely shutdown camera
cam.release()
# distry all windows associated in the program.
cv2.destroyAllWindows()

