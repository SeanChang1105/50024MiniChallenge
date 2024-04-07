import boto3
import io
from PIL import Image, ImageDraw
import os
import sys
from natsort import natsorted 

def show_faces(bucket):

    session = boto3.Session(profile_name='default')
    client = session.client('rekognition')

    # Load image from S3 bucket
    s3_connection = boto3.resource('s3')

    oslist=os.listdir("train_small")
    sortedList=natsorted(oslist)
    idx=0
    for i in sortedList:
        photo = i
        s3_object = s3_connection.Object(bucket, photo)
        s3_response = s3_object.get()
        stream = io.BytesIO(s3_response['Body'].read())
        image = Image.open(stream)
        image = image.convert('RGB')
        # Call DetectFaces
        try:
            response = client.detect_faces(Image={'S3Object': {'Bucket': bucket, 'Name': photo}},Attributes=['ALL'])

            imgWidth, imgHeight = image.size

            # calculate and display bounding boxes for each detected face
            if response['FaceDetails']==[]:
                # no face
                print("No face at "+i)
                image.save("train_small_aws/"+photo)
            else:
                for faceDetail in response['FaceDetails']:
                    box = faceDetail['BoundingBox']
                    left = imgWidth * box['Left']
                    top = imgHeight * box['Top']
                    width = imgWidth * box['Width']
                    height = imgHeight * box['Height']

                    right = left + width
                    bottom = top + height
                    cropped_image = image.crop((left, top, right, bottom))
                    cropped_image.save("train_small_aws/"+photo)
                    break
        except:
            print("Problem at "+i)
            image.save("train_small_aws/"+photo)
        idx+=1
        if idx%100==0:
            print(idx)





def main():
    bucket = "ece50024train-small"
    faces_count = show_faces(bucket)
    print("DONE")


if __name__ == "__main__":
    main()