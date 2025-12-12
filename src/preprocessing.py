import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ImageNet statistik
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_preprocessing_transform():
    """
    Returnerar transform-pipeline för VGG16.
    Inkluderar resize, tensor-konvertering och normalisering.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def load_image(image_path, device):
    """
    Laddar en bild, preprocessar den och lägger den på GPU/CPU.
    Returnerar en tensor med shape (1, 3, 224, 224).
    """
    image = Image.open(image_path).convert('RGB')
    transform = get_preprocessing_transform()
    image_tensor = transform(image).unsqueeze(0) # Lägg till batch-dimension
    return image_tensor.to(device)

def deprocess_image(tensor):
    """
    Konverterar en normaliserad tensor tillbaka till en visningsbar numpy-array (bild).
    Ger en array med värden mellan 0 och 1 (eller 0-255).
    """
    # Ta bort batch-dimension om den finns
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Flytta till CPU och numpy
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Denormalisera: img = img * std + mean
    img = img * np.array(STD) + np.array(MEAN)
    
    # Clampa värden till 0-1
    img = np.clip(img, 0, 1)
    
    return img

def show_image(tensor, title=None):
    """
    Hjälpfunktion för att snabbt visa en tensorbild.
    """
    img = deprocess_image(tensor)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
