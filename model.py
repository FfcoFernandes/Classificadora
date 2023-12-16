import torch
from PIL import Image, ImageTk
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.features_size = 8 * 8 * 256

        self.fc1 = nn.Linear(self.features_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

        self.load_state_dict(torch.load('./modelo_treinado.pth'))
        self.eval()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x) # 128

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x) # 64

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x) # 32

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x) # 16

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x) # 8

        x = x.view(-1, self.features_size)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

classes = {
    'Cat' : 'Gato',
    'Dog' : 'Cachorro'
}

model = SimpleCNN(len(classes)).to(device)

def classificar_imagem(path: str):
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    imagem = Image.open(path).convert('RGB')
    imagem = transform(imagem).unsqueeze(0)

    with torch.no_grad():
        output = model(imagem)

    _, predicao = torch.max(output, 1)
    predicao = predicao.item()

    probabilidade = round(F.softmax(output, dim=1)[0, predicao].item(), 2) * 100

    return classes[list(classes.keys())[predicao]], probabilidade