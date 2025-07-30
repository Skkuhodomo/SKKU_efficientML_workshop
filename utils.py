import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import random
import math

def set_seed(seed: int = 0):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_layers_resnet(model):
    """
    Return an *ordered* list of (name, module) pairs containing
    **every Conv2d or Linear** that has a weight we want to prune.
    Ordering == forward order so that sequential calibration still works.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))
    return layers

def find_layers_resnet(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))
    return layers

# Seed & Device
def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR 전용 ResNet18 정의
def resnet18_cifar(num_classes=100):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def print_all_module_names(model):
    for name, module in model.named_modules():
        print(name)
@torch.no_grad()
def get_acc(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    return acc
