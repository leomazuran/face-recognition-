# this program will use the a webcam, detect a face using Haar Cascade and let the user save crop image of the person's face and save it on a local documnet folder
# then the program will get the images from the document folder and convert it to a data set using opencv's Local Binary Pattern Histogram Face Recognition functions.
#calling cv2
import cv2
# calling cv2. face for create LBPHFaceRegonition()
import cv2.face
import numpy as np
from PIL import Image
import os
#creating local binary pattern images from saved images
recognizer = cv2.face.createLBPHFaceRecognizer()
# get path location of image for training
path = 'C://Users//leo//Documents//save//images'

# get face cascade data to find possible faces in video feed.
face_cascade = cv2.CascadeClassifier('C://Users//leo//Downloads//haarcascade_frontalface_default.xml')
# get and select webcam, 0 = main webcam.
cap = cv2.VideoCapture(0)
# counter for multiple crop images saved in file.
count = 0;
#user input information/ save to txt database.
e = input ("Enter Name")
name =input ("Enter Id for Name")
n = name
# while webcam is running check for possibl faces in webcam feed using the CascadeClassifer.

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    # if face exist, for everyface display a rectange box on face.
    # not recommended for user to show multiple faces and saving crop images for single ID.
    for (x,y,w,h) in face:
         # create a rectangle (img = wecam, get x y of face detect and, get the width and hight
        # (255,0,0) = color of rectangle
        # 2 = thickness of the rectangle.higher = thicker
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        # crop_img get face using the face detection parameters
        crop_img = gray[y:y+h, x:x+w]
        
        #roi_gray = gray[y:y+h, x:x+w]
        roi_color = img [y:y+h, x:x+w]
 # run image show       
    cv2.imshow('img', img)
# if user press the 't' key save and print gray crop image.
    if cv2.waitKey(30) & 0xFF == ord('t'):
        count = count+1

        cv2.imshow('crop'+str(count), crop_img)
        cv2.imwrite("C://Users//leo//Documents//save//images//id_" +n+"_"+str(count)+  ".jpg", crop_img)


# if user press 's' save full image. 
    if cv2.waitKey(30) & 0xFF == ord('s'):
        cv2.imwrite("C://Users//leo//Documents//save//full//" + n + "f.jpg", img)
    k = cv2.waitKey(30) & 0xff
    # if user press 'esc' break while loop
    if k == 27:

        break
cap.release()
cv2.destroyAllWindows()
# save user input to database.
# it is important not use 'w' when opening the document as it will overwrite the database with other information, use 'a' to keep existing data plus new.
f = open("C://Users//leo//Documents//save//database//test.txt", 'a')
f.write( e + " " + str(name) + "\n"  )
f.close()

#**********************************************************

# start traing of all images in saved document folder

#**********************************************************
print ("train start")

def getpath(path) :
    # gewt all the lists of files in the document folder.
    location = [os.path.join(path,f) for f in os.listdir(path)]
    f = []
    I= []
    for image in location :
        #convert gray image to luminance (L)
        fImg= Image.open(image). convert ('L')
        # numpy array image to a matrix unsigned 8-bit integer
        fNp= np.array(fImg, 'uint8')
        # get the specific path name of the image file
        # It first gets the image file in the document specified.
        # then it will start at the end of the file name id_1_x.jpg, [-1] will skip the file type .jpg
        # then it will .split and skip first '_' on file and then get single id number
        ID = int (os.path.split (image) [-1].split('_')[1]) 
        f.append(fNp)
        I.append(ID)
        cv2.imshow("train", fNp)  
    return I,f
I,f =getpath(path)
# this will do all the local binary pattern histagram 
recognizer.train(f,np.array(I))
# then input it to a dataset file and save to a .data file
recognizer.save('C://Users//leo//Documents//save//database//test.data')
print ("Done")
# close all existing windows associated in the python program
cv2.destroyAllWindows()

