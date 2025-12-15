import torch
import torchvision.models as models

def load_model(device):
    """
    Laddar VGG16 modellen med förtränade vikter.
    Sätter modellen i evalueringsläge men fryser INTE parametrarna med requires_grad=False.
    Detta är viktigt för att verktyg som torchcam ska kunna traversera grafen bakåt.
    Vi förlitar oss på torch.no_grad() i main-loopar om vi inte vill ha gradients,
    men för CAM och DeepDream behöver vi dem.
    """
    print("Laddar VGG16...")
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    # Skicka till enhet (GPU/CPU)
    model = model.to(device)
    
    # Sätt i eval mode (viktigt för dropout/batchnorm)
    model.eval()
    
    # OBS: Vi fryser INTE parametrar här, för att undvika problem med 
    # autograd-grafer i vissa bibliotek (som torchcam). 
    # Vi sätter explicit requires_grad=True på input-bilden istället när det behövs.
    
    # Men för säkerhets skull, se till att modellen har gradients påslagna om de var avslagna
    for param in model.parameters():
        param.requires_grad = True
        
    print(f"Modell laddad på {device}")
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    print(model)
