import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from PIL import Image
from utils import *
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1


# Model path and csvfile name
modelPath='models\model_onlyone_5.pth'
csvfilename='otuput5.csv'

print("Getting data...")
data_transforms=transforms.Compose([
        transforms.Resize((192,192)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Get test data
images,labels=load_images_from_folder('test_awsCropped')
dataset = CustomDataset(images,labels,data_transforms)
test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

# Load in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2',classify=True,num_classes=100,device=device)
model.load_state_dict(torch.load(modelPath))
model.eval()
outputLabel=[]

print("Start predicting...")
with torch.no_grad():
    for idx,(images,_) in enumerate(test_loader):
        images=images.to(device)
        outputs=model(images)
        _,predictions=torch.max(outputs,1)
        outputLabel.append(predictions.item())

print(len(outputLabel))
print(outputLabel)


# Write to Csv
print("Writing to csv...")
outputCsv(outputLabel,csvfilename)
print("Done!")

