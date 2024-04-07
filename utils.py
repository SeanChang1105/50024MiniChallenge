import csv
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from natsort import natsorted 

# Create label from the csv
def createLabel():
    labels=[]
    path='train.csv'
    with open(path,mode='r')as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[2]=='Category':
                continue
            labels.append(row[2])
    return labels

# Create dictionary to map class to id (name to id)
def nameToId():
    out={}
    with open('category.csv',mode='r')as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[0]=='':
                continue
            name=row[1]
            out[name]=int(row[0])
    return out

# Create dictionary to map id to class (id to name)
def idToName():
    out={}
    with open('category.csv',mode='r')as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if row[1]=='Category':
                continue
            idx=int(row[0])
            out[idx]=row[1]
    return out

# Function to load and preprocess images
def load_images_from_folder(folder):
    labelCSV=createLabel()
    categoryDict=nameToId()
    images = []
    labels = []
    oslistdir=os.listdir(folder)
    sorted=natsorted(oslistdir)
    for pic_name in sorted:
        img_path=folder+'\\'+pic_name
        img = Image.open(img_path).convert('RGB')  # Convert to RGB if your images are in color
        images.append(img)

        # Get the picture index by name
        idx=int(pic_name[:-4])
        categoryIdx=categoryDict[labelCSV[idx]]
        labels.append(categoryIdx)
    return images, labels

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)

        return image, target
    

def outputCsv(labels,fileName):
    id2Name=idToName()
    with open(fileName,'w',newline='') as file:
        writer = csv.writer(file)
        field=['Id','Category']
        writer.writerow(field)
        for idx,label in enumerate(labels):
            writer.writerow([idx,id2Name[label]])
