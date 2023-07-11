#이미지 사이즈와 픽셀값 / 손실함수 관련 중요변수 논의 필요
#주소 다 데려와야함

# outputs = (모델이 준 답안), labels = (실제 해답).aka.ground

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import InriaDataset, YourDataset
from torch.utils.data.dataset import Subset
from fusionnet import FusionNet
import loss

#--중요 변수--#
#이미지 사이즈와 픽셀값
sizex = #224
sizey = #224
pix = #0.5

#훈련 루프 변수
learning_rate = 0.001
betas = (0.9, 0.999)
epsilon = 1e-8
batch_size = 12
num_epochs = 200
group_norm_size = 32

#import 및 저장 주소
fusionload = #fusionnet 모델 로드 주소 ''

Inria = #기본 데이터셋 주소 ''
#Your = #해상도 다른 데이터셋 추가할 시 주소 ''

best_S1 = #제일 잘나온 s1 모델 저장 주소 ''

#--#

# Define the S1 model following the pipeline
class S1Net(nn.Module):
    def __init__(self):
        super(S1Net, self).__init__()
        # Add layers as per your pipeline
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Add more layers as per your requirements

        self.fc = nn.Linear(64 * sizex//2 * sizey//2, 2)  # Adjust output dimensions based on your task

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        # Add more layers and pooling as per your pipeline

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create the S1 model instance
model = S1Net()

# Load the pretrained FusionNet model weights into the S1 model
fusion_model = torch.load(fusionload)
model.load_state_dict(fusion_model.state_dict())

#--#

# Create the Adam optimizer 아담 옵티마이저
optimizer = optim.Adam(
    params=model.parameters(),
    lr=learning_rate,
    betas=betas,
    eps=epsilon
)

#--#

#데이터 준비 - transform 정의 / 데이터 가져오기 / 데이터 로더 생성

# transform 정의 -- 데이터 크기 확인 및 픽셀 어떻게 할지
transform = transforms.Compose([
    transforms.Resize((sizex, sizey)),  # Resize the images to a consistent size 크기 일정하게
    transforms.ToTensor(),  # Convert the images to tensors 텐서로 바꿔서 쓸수 있게 만들어 줌
    transforms.Normalize((pix, pix, pix), (pix, pix, pix))  # Normalize the pixel values 픽셀 값 지정해주기
])

# 데이터 가져오기, 분류 및 데이터 로더 만들기
inria_dataset = InriaDataset(root= Inria, transform=transform)

inria_train_indices = range(86400)
inria_val_indices = range(86400, 115200)
inria_test_indices = range(115200, 144000)

inria_train_set = Subset(inria_dataset, inria_train_indices)
inria_val_set = Subset(inria_dataset, inria_val_indices)
inria_test_set = Subset(inria_dataset, inria_test_indices)

inria_train_loader = DataLoader(inria_train_set, batch_size=batch_size, shuffle=True)
inria_val_loader = DataLoader(inria_val_set, batch_size=batch_size)
inria_test_loader = DataLoader(inria_test_set, batch_size=batch_size)

'''
your_dataset = YourDataset(root= Your, transform=transform)

your_train_indices = range(4500)
your_val_indices = range(4500, 6750)
your_test_indices = range(6750, 9000)

your_train_set = Subset(your_dataset, your_train_indices)
your_val_set = Subset(your_dataset, your_val_indices)
your_test_set = Subset(your_dataset, your_test_indices)

your_train_loader = DataLoader(your_train_set, batch_size=batch_size, shuffle=True)
your_val_loader = DataLoader(your_val_set, batch_size=batch_size)
your_test_loader = DataLoader(your_test_set, batch_size=batch_size)'''

#--#

# 훈련 루프
best_accuracy = 0.0
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for images, labels in inria_train_loader:
        # Perform the forward pass, compute loss, and backward pass
        outputs = model(images)
        loss = loss.s1_loss(outputs, labels)
        # Update the model's parameters using the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in inria_val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    # Validation phase 결과 따라 가장 정확도 높은 모델 저장
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_S1)

    # Adjust the learning rate every 50 epochs
    if (epoch + 1) % 50 == 0:
        learning_rate /= 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    #print(f"Learning rate adjusted to: {learning_rate}")
