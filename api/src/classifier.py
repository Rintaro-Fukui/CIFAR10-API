import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(in_features=4 * 4 * 128, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Classifier:
    def __init__(self, params_path='../utils/cifar10_cpu.pth'):
        self.model_params = torch.load(params_path)
        self.model = CNN()

    def preprocessing(self, img):
        img = Image.open(img).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        img_transformed = transform(img).unsqueeze(0)
        return img_transformed

    def predict(self, input):
        model_ = self.model.load_state_dict(self.model_params)
        output = model_(input)
        pred = nn.Softmax(dim=1)(output)
        label = int(pred.argmax(1))
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return {'class:', classes[label], 'score:', float(pred[0][label])}