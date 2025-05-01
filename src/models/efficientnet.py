from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=None):
        super(EfficientNetModel, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last few layers if you want fine-tuning (optional)
        for param in self.model.features[-2:].parameters():  # Unfreezing last 2 blocks
            param.requires_grad = True

        # Replace the classifier
        in_features = self.model.classifier[1].in_features  # EfficientNet's last fc layer
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)  # 16 classes
        )

    def forward(self, x):
        return self.model(x)

    @property
    def classifier(self):
        return self.model.classifier

# class EfficientNetModel(nn.Module):
#     def __init__(self):
#         super(EfficientNetModel, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
#         self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
#         self.fc1 = nn.Linear(128 * 28 * 28, 512) 
#         self.fc2 = nn.Linear(512, 15)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  
#         x = self.pool(F.relu(self.conv2(x)))  
#         x = self.pool(F.relu(self.conv3(x))) 
        
#         x = x.view(-1, 128 * 28 * 28)  # Flattening here(very important, error was occuring because of it)
        
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x