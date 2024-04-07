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
import time

# Specify model name and dataset path!
modelName="model_onlyone_5"
dataPath='cropped_onlyone'

data_transforms=transforms.Compose([
        transforms.Resize((192,192)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Get data
print("Getting data...")
images,labels=load_images_from_folder(dataPath)
dataset = CustomDataset(images,labels,data_transforms)

# Uses GPU
print("Setting up GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 20



train_size = int(0.95 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)
print("train size: ",len(train_loader.dataset))
print("test size: ",len(test_loader.dataset))



# Load FaceNet model
model = InceptionResnetV1(pretrained='vggface2',classify=True,num_classes=100,device=device).to(device)
# Load model to train in part 2
#model.load_state_dict(torch.load("models\model_full_2.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)


print("\nStart training!\n")
st=time.strftime("%H:%M:%S", time.localtime())
minLost=100

# Training loop
for epoch in range(num_epochs):
    epoch_loss=0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        try:
            data, targets = data.to(device), targets.to(device)
            
            # Forward
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()
        except:
            print("Problem at batch",batch_idx)

    # Calculate average epoch loss
    epoch_loss /= len(train_loader)
    if epoch>10 and epoch_loss<minLost:
        minLost=epoch_loss
        print("Saving model as: ",modelName,"in epoch ",epoch)
        torch.save(model.state_dict(),"models/"+modelName+".pth")
    # Print intermediate results
    print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}') #######???????????

print("\nTraining Done!\n")
print("Start time: ",st)
print("End time: ",time.strftime("%H:%M:%S", time.localtime()))

# print("Saving model as: ",modelName)
# torch.save(model.state_dict(),"models/"+modelName+".pth")




print("Testing!")
model.eval()
total=0
correct=0
with torch.no_grad():
    for idx,(images,labels) in enumerate(test_loader):
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        _,predictions=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predictions==labels).sum().item()
    accuracy=correct/total

print('Accuracy: {:.2f}%'.format(100 * accuracy))