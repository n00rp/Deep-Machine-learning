import torch
from torchcam.methods import SmoothGradCAMpp, GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

def generate_cam(model, input_tensor, target_layer=None, class_idx=None):
    """
    Genererar Class Activation Map (CAM) för en given input.
    Säkerställer att gradienter kan beräknas.
    """
    
    # VIKTIGT: För att kunna backpromovera måste vi ha en graf.
    # Se till att input_tensor har requires_grad=True
    if not input_tensor.requires_grad:
        input_tensor.requires_grad = True
        
    # Använd SmoothGradCAMpp (mindre brusig) eller GradCAM
    # Vi wrappar i enable_grad för säkerhets skull
    with torch.enable_grad():
        cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)
        
        # Forward pass
        out = model(input_tensor)
        
        # Om ingen klass angiven, ta den med högst score
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
            
        print(f"Genererar CAM för klassindex: {class_idx}")
        
        # Hämta aktiveringskartan (backward triggas här inne)
        activation_map = cam_extractor(class_idx, out)
        
        # Resultat
        heatmap = activation_map[0]
        
        cam_extractor.remove_hooks()
    
    return heatmap

def visualize_cam(heatmap, original_tensor, title="CAM", show=True):
    """
    Lägger heatmap ovanpå originalbilden och visar resultatet.
    """
    from src.preprocessing import deprocess_image
    
    # Konvertera original till PIL
    img_arr = deprocess_image(original_tensor.detach()) # Detach för säkerhet
    pil_img = to_pil_image(torch.from_numpy(img_arr.transpose(2, 0, 1)).float())
    
    # Overlay
    result = overlay_mask(pil_img, to_pil_image(heatmap, mode='F'), alpha=0.5)
    
    if show:
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
