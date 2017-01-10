# face-recognition-
############## getandtrain.py#####################
This project is intended for recognize face using haar cascade clasifier provided from:
https://github.com/opencv/opencv/tree/master/data/haarcascades.
When the user starts the program it will be greeted with a user and id.
There is no validator for the inputs so make sure that there is no duplicate ids in the txt database.
(I will patch this so that users dont put duplicate names and ids in the input and get inaccurate results in the future)
All images are save to a specified folder after user esc the program. It will taking images of the recipient
the software will convert each images to a Local Binary Pattern Histogram and save it into a Dataset.
#############facerec.py######################
when starting up, the user will load the Dataset file and start up the computers local webcam.
Using the haar cascade clasifier it will search for a face and find covert it to a LBP histogram and compare it to the dataset. It will 
look for the closest related histogram associated with it and print out the id from the dataset and search for the user in the database text file.
Visit this site to view a demo of this python program:
https://www.youtube.com/watch?v=Bjz2lmWh0ws

both python files are need and are both commented to better explain what each function does.
Libraries used:
opencv3.0, PIL from image, numpy, os.
language: python 3.5.
Please feel free to comment or give any feedback or corrections.
