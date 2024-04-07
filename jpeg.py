import cv2 as cv
import os
import csv
import sys
from natsort import natsorted 
from PIL import Image

oslist=os.listdir("train")
sortedList=natsorted(oslist)

for i in sortedList:
    path="train\\"+i
    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        # Save the image as JPEG format
        print("problem at ",i)
        image.save(path, 'JPEG')