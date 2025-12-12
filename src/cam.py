import torch
from torchcam.methods import SmoothGradCAMpp, GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

def generate_cam(model, input_tensor, target_layer=None, class_idx=None):
    """
    Genererar Class Activation Map (CAM) för en given input.
    Måste hantera hook-registrering och borttagning snyggt, 
    men torchcam context manager löser det ofta.
    
    Här använder vi 'SmoothGradCAMpp' som är en kraftfull variant, 
    eller vanlig 'GradCAM'.
    """
    
    # Välj target_layer om angivet (strängnamn), annars tar torchcam sista conv-lagret automatiskt
    # För VGG16 är modellens features-del en Sequential.
    # Exempel: 'features.28' är sista conv-lagret innan maxpool.
    
    # Vi använder SmoothGradCAMpp för snyggare resultat
    cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)
    
    # Kör forward pass
    out = model(input_tensor)
    
    # Om ingen klass angiven, ta den med högst score
    if class_idx is None:
        class_idx = out.argmax(dim=1).item()
        
    print(f"Genererar CAM för klassindex: {class_idx}")
    
    # Hämta aktiveringskartan
    # Notera: input_tensor har shape (1, 3, H, W)
    activation_map = cam_extractor(class_idx, out)
    
    # cam_extractor returnerar en lista av maps (en per target layer)
    # Vi tar den första (och enda om vi angav ett lager)
    heatmap = activation_map[0]
    
    # Städa upp hooks
    cam_extractor.remove_hooks()
    
    return heatmap

def visualize_cam(heatmap, original_tensor, title="CAM"):
    """
    Lägger heatmap ovanpå originalbilden och visar resultatet.
    """
    # Convertera original-tensor till PIL Image för overlay funktionen
    # Vi måste deprocessa först till 0-1, men overlay_mask förväntar sig PIL image
    from src.preprocessing import deprocess_image
    
    img_arr = deprocess_image(original_tensor) # numpy 0-1
    # overlay_mask vill ha PIL Image
    pil_img = to_pil_image(torch.from_numpy(img_arr.transpose(2, 0, 1)).float())
    
    # Konvertera heatmap till PIL image 
    # torchcam overlay_mask tar hand om resizing av heatmap automatiskt
    
    result = overlay_mask(pil_img, to_pil_image(heatmap, mode='F'), alpha=0.5)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(pil_img)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title(title)
    plt.axis('off')
    
    plt.show()
    return result
