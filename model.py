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
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 32 * 64, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.load_state_dict(torch.load('./modelo_treinado.pth'))
        self.eval()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(-1, 32 * 32 * 64)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

classes = {
    'Airplane' : 'Avião',
    'Car' : 'Carro',
    'Motorcycle' : 'Moto',
    'Truck' : 'Caminhão'
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