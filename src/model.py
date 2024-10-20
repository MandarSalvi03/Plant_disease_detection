import torch.nn as nn

from torchvision import models
from torchvision.models import ResNet34_Weights

def build_model(pretrained=True, fine_tune=True, num_classes=10):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    else:
        print('[INFO]: Not loading pre-trained weights')
        model = models.resnet34(weights=None)
        
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    model.fc = nn.Linear(in_features=512, out_features=num_classes)
    return model