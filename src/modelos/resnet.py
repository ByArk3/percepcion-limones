"""ResNet18 del proyecto (celdas 10.1 y 10.2 del notebook): layer4 + fc entrenables, resto congelado."""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from src.comun import CLASES, DESV, MEDIA

aug = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                          transforms.RandomRotation(180), transforms.ColorJitter(brightness=0.15),   # sin cambiar el tono
                          transforms.ToTensor(), transforms.Normalize(MEDIA, DESV)])
sin_aug = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(MEDIA, DESV)])


class DS(torch.utils.data.Dataset):
    """imgs: lista de imágenes BGR; idx: posiciones a usar; y: etiquetas de todas las imágenes."""

    def __init__(s, imgs, idx, tf, y):
        s.imgs, s.idx, s.tf, s.y = imgs, idx, tf, y

    def __len__(s):
        return len(s.idx)

    def __getitem__(s, k):
        i = s.idx[k]
        return s.tf(np.ascontiguousarray(s.imgs[i][:, :, ::-1])), int(s.y[i])


def crear_resnet(device):
    try:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception as e:
        print("No se pudieron descargar los pesos de ImageNet, se usa inicialización aleatoria:", e)
        net = models.resnet18(weights=None)
    for nombre, p in net.named_parameters():                       # congelar hasta layer3
        p.requires_grad = nombre.startswith("layer4")
    net.fc = nn.Linear(512, len(CLASES))                           # cabeza nueva: FC(512 -> 3)
    return net.to(device)


def predecir(net, imgs, idx, y, device):
    net.eval()
    out = []
    with torch.no_grad():
        for xb, _ in torch.utils.data.DataLoader(DS(imgs, idx, sin_aug, y), batch_size=32):
            out.append(torch.softmax(net(xb.to(device)), 1).cpu())
    return torch.cat(out).numpy()
