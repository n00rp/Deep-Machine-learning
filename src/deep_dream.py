import torch
import torch.nn as nn
import torch.optim as optim
from src.preprocessing import deprocess_image
import numpy as np
import copy
import torchvision.transforms.functional as TF

def deep_dream(model, input_image_tensor, layer_names, iterations=20, lr=0.02, octave_scale=1.4, num_octaves=4, device='cpu'):
    """
    Förbättrad DeepDream implementation.
    """
    model.eval()
    
    # Klonar bilden och flyttar till device
    img = input_image_tensor.clone().detach().to(device)
    
    # Förbered oktaver
    octaves = [img]
    for i in range(num_octaves - 1):
        new_size = [int(dim / octave_scale) for dim in octaves[-1].shape[-2:]]
        scaled_img = torch.nn.functional.interpolate(octaves[-1], size=new_size, mode='bilinear', align_corners=False)
        octaves.append(scaled_img)
    
    octaves = octaves[::-1] # Börja med minsta
    
    detail = torch.zeros_like(octaves[0]).to(device)
    
    # Loop over octaves
    for octave_idx, octave_base in enumerate(octaves):
        print(f"Drömmer i oktav {octave_idx+1}/{num_octaves} ({octave_base.shape[-2]}x{octave_base.shape[-1]})...")
        
        # Upscale detail from previous octave
        if octave_idx > 0:
            detail = torch.nn.functional.interpolate(detail, size=octave_base.shape[-2:], mode='bilinear', align_corners=False)
            
        # Combine base image (downscaled original) + details found so far
        input_img = (octave_base + detail).clone().detach().requires_grad_(True)
        
        optimizer = optim.Adam([input_img], lr=lr)
        
        # Hooks
        activations = {}
        hooks = []
        def get_hook(name):
            def hook(model, input, output):
                activations[name] = output
            return hook
            
        for name in layer_names:
            if 'features' in name:
                idx = int(name.split('.')[1])
                handle = model.features[idx].register_forward_hook(get_hook(name))
                hooks.append(handle)
                
        # Gradient Ascent with Jitter
        for i in range(iterations):
            # Apply jitter (random shift)
            ox, oy = np.random.randint(-2, 3), np.random.randint(-2, 3)
            img_jittered = torch.roll(input_img, shifts=(ox, oy), dims=(2, 3))
            
            optimizer.zero_grad()
            model(img_jittered)
            
            loss = 0
            for name in layer_names:
                act = activations[name]
                # Vi vill maximera L2-normen -> minimera negativ
                loss -= torch.norm(act)
            
            loss.backward()
            
            # Normalisera gradienten (viktigt för snygg deep dream!)
            if input_img.grad is not None:
                grad_std = torch.std(input_img.grad.data)
                input_img.grad.data /= (grad_std + 1e-8)
            
            optimizer.step()
            
            # Un-jitter bild (vi gör inte det manuellt här då vi uppdaterade 'input_img' direkt 
            # och rullningen var temporär input till modellen, men gradienten hamnar rätt)
            # DOCK: Om man rullar inputten till model(), hamnar gradienten i den rullade tensor positionen.
            # PyTorch `torch.roll` är differentierbar. Så `input_img` uppdateras korrekt 'in-place'?
            # Nej, `input_img` flyttas inte. Vi skickade in en rullad kopia.
            # Så input_img.grad får gradienter motsvarande rullad bild.
            # Vi måste rulla tillbaka gradienten! 
            
            # Fix av jitter-gradient:
            # Eftersom vi rullade INNAN forward, så motsvarar input_img.grad[x,y] pixeln i img_jittered[x,y].
            # Vi måste rulla TILLBAKA gradienten.
            with torch.no_grad():
                input_img.grad.data = torch.roll(input_img.grad.data, shifts=(-ox, -oy), dims=(2, 3))
                
            # Clamping för att hålla färger vettiga? 
            # Vi gör det efter loopen
            
        # Extract details added in this octave
        detail = input_img - octave_base
        
        # Clean hooks
        for h in hooks:
            h.remove()
            
    return deprocess_image(input_img)
