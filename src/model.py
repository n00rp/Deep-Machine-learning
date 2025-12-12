import torch
import torchvision.models as models

def load_model(device):
    """
    Laddar VGG16 modellen med förtränade vikter.
    Sätter modellen i evalueringsläge och fryser parametrarna för att spara minne,
    eftersom vi inte ska träna modellen, bara visualisera.
    """
    print("Laddar VGG16...")
    # Använd weights=models.VGG16_Weights.IMAGENET1K_V1 eller VGG16_Weights.DEFAULT
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    # Skicka till enhet (GPU/CPU)
    model = model.to(device)
    
    # Sätt i eval mode (viktigt för dropout/batchnorm)
    model.eval()
    
    # Frys parametrar (vi ska inte backpromovera för att ändra vikter, bara input)
    for param in model.parameters():
        param.requires_grad = False
        
    print(f"Modell laddad på {device}")
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    print(model)
