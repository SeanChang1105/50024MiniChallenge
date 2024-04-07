import cv2 as cv
import os
import csv
import sys
from natsort import natsorted 
def getLabels(filepath):
    labels=[]
    with open(filepath,mode='r')as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[2]=='Category':
                continue
            labels.append(row[2])
    return labels


def cropFaces():
    # Load the cascade
    failed=0
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    filepath=r"C:\Users\user\Desktop\ECE524\test"
    savepath=r'C:\Users\user\Desktop\ECE524\test_onlyone'
    #savepath2=r'\Users\user\Desktop\ECE524\cropped_onlyone'
    for dir in natsorted(os.listdir(filepath)):
        try:
            img = cv.imread(filepath+"\\"+dir)
            # Convert into grayscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.2, 4,minSize=(32,32))
            if len(faces)==1:
                cropped=img[faces[0][1]:faces[0][1]+faces[0][3],faces[0][0]:faces[0][0]+faces[0][2]]
                #cv.imwrite(savepath2+'\\'+dir,cropped)
            else:
                print("Can't find face ",dir)
                cropped=img
            cv.imwrite(savepath+'\\'+dir,cropped)
        except:
            failed+=1
            print("  !!PROBLEM!!  ", dir)
            cv.imwrite(savepath+'\\'+dir,img)
    print("Total failed: ",failed)


#labels=getLabels('train_small.csv')
cropFaces()
print("done")