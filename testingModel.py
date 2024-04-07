import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from PIL import Image
from utils import load_images_from_folder,CustomDataset
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import sys

# Model path and data path
modelPath='models\model_onlyone_5.pth'
dataPath='train_small_aws'


# Load and preprocess data
data_transforms=transforms.Compose([
        transforms.Resize((192,192)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Get test data
images,labels=load_images_from_folder(dataPath)
dataset = CustomDataset(images,labels,data_transforms)

# train_size = int(0.5 * len(dataset))
# test_size = len(dataset) - train_size
# _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

test_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)



# Load in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2',classify=True,num_classes=100,device=device)
model.load_state_dict(torch.load(modelPath))
model.eval()


print("Start testing...")
total=0
correct=0
predictLabel=[]
with torch.no_grad():
    for idx,(images,labels) in enumerate(test_loader):
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        _,predictions=torch.max(outputs,1)
        # print('P:',predictions.item())
        # print('T:',labels.item())
        #predictLabel.append(predictions.item())
        
        total+=labels.size(0)
        correct+=(predictions==labels).sum().item()
    accuracy=correct/total

print('Accuracy: {:.2f}%'.format(100 * accuracy))
#print(predictLabel)

