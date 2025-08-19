import os
import torch.backends.cudnn as cudnn
import gdown
import torch
import torch.nn as nn
from utils import resnet18_cifar, get_device, set_seed, get_acc
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import *  

    # -------------------------
    # 1) Checkpoint 다운로드
    # -------------------------
file_id = "1FH8zBN81t1sIfWvZTuypR_25T_Xwdekq"
ckpt_path = "resnet18_cifar100_pretrained.ckpt"
if not os.path.exists(ckpt_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading checkpoint from {url}")
        gdown.download(url, ckpt_path, quiet=False)
else:
    print(f"Checkpoint already exists: {ckpt_path}")

    # -------------------------
    # 2) 시드 & 디바이스
    # -------------------------
seed, batch_size, nsamples = 0, 128, 512
set_seed(seed)
device = get_device()
cudnn.benchmark = True


    # 3) CIFAR-100 데이터셋
    # -------------------------
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # -------------------------
    # 3) 캘리브레이션 데이터셋
    # -------------------------
calibset = torch.utils.data.Subset(testset, list(range(nsamples)))
calibloader = torch.utils.data.DataLoader(calibset, batch_size=1, shuffle=False, num_workers=4)




    # -------------------------
    # 4) 모델 로드
    # -------------------------    
model = resnet18_cifar(num_classes=100)
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval().to(device)
print("✅ Pretrained Model Loaded")
