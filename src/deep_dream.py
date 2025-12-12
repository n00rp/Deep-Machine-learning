import torch
import torch.nn as nn
import torch.optim as optim
from src.preprocessing import deprocess_image
import numpy as np
import copy

def deep_dream(model, input_image_tensor, layer_names, iterations=20, lr=0.01, octave_scale=1.4, num_octaves=4, device='cpu'):
    """
    Implementerar DeepDream algoritm.
    
    Args:
        model: VGG16 model (eval mode)
        input_image_tensor: Startbild (tensor)
        layer_names: Lista av lagernamn att optimera mot (t.ex. ['features.24'])
        iterations: Antal gradient steps per oktav
        lr: Learning rate
        octave_scale: Skalningsfaktor mellan oktaver
        num_octaves: Antal skalor
        device: 'cuda' eller 'cpu'
    """
    model.eval()
    img = input_image_tensor.clone().detach().to(device)
    img.requires_grad = True
    
    # Förbered oktaver (skalor)
    octaves = [img]
    for i in range(num_octaves - 1):
        # Skapa mindre versioner av bilden
        new_size = [int(dim / octave_scale) for dim in octaves[-1].shape[-2:]]
        # Använd interpolate för downsampling
        scaled_img = torch.nn.functional.interpolate(octaves[-1], size=new_size, mode='bilinear', align_corners=False)
        octaves.append(scaled_img)
    
    # Börja från minsta oktaven och arbeta uppåt
    # Vi vänder på listan så vi börjar med minsta
    octaves = octaves[::-1]
    
    # Grundbild (detaljer som vi lägger till)
    detail = torch.zeros_like(octaves[0]).to(device)
    
    for octave_idx, octave_base in enumerate(octaves):
        print(f"Bearbetar oktav {octave_idx+1}/{num_octaves} med storlek {octave_base.shape[-2:]}")
        
        # Om det inte är första oktaven, skala upp detaljerna från föregående
        if octave_idx > 0:
            # Skala upp detail till nuvarande oktavs storlek
            detail = torch.nn.functional.interpolate(detail, size=octave_base.shape[-2:], mode='bilinear', align_corners=False)
            
        # Lägg till detaljerna till basbilden för denna oktav
        # Observera: octave_base är bara en nedskalad version av originalet.
        # Vi optimerar egentligen 'img' men vi gör det via denna loopstruktur för multi-scale.
        # För enkelhets skull i denna implementering: Vi låter `img` vara det vi optimerar.
        # Men DeepDream brukar "drömma" på en startbild och sen skala upp.
        
        # Enklare variant för labb:
        # Vi sätter current_img till octave_base + detail
        current_img = (octave_base + detail).clone().detach().requires_grad_(True)
        
        optimizer = optim.Adam([current_img], lr=lr)
        
        # Hook för att få aktiveringar från FLERA lager
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
        
        for i in range(iterations):
            # Jitter: flytta bilden slumpmässigt för att undvika rutmönster
            shift_x, shift_y = np.random.randint(-2, 3), np.random.randint(-2, 3)
            # Applicera shift (rulla)
            img_shifted = torch.roll(current_img, shifts=(shift_x, shift_y), dims=(2, 3))
            
            optimizer.zero_grad()
            _ = model(img_shifted)
            
            loss = 0
            for name in layer_names:
                act = activations[name]
                # Maximera L2 normen av aktiveringen (standard DeepDream objective)
                loss -= torch.norm(act)
            
            loss.backward()
            
            # Un-jitter gradient (om vi skulle uppdatera manuellt), men Adam hanterar params.
            # Vi måste dock "rulla tillbaka" bilden om vi vill behålla positionen, 
            # men här uppdaterar vi 'current_img'. 
            # Egentligen borde vi rulla tillbaka gradienten innan step, men pytorch autograd hanterar
            # operationerna. Men vänta, om vi skickar in rullad bild, beräknas grad på rullad bild.
            # Vi låter det vara för enkelhetens skull, eller rullar tillbaka strömbilden? NO.
            # Korrekt jitter-implementation kräver lite mer meck med gradients manuellt.
            # Vi kör utan explicit jitter-roll-back för denna enkla version, 
            # eller lägger in en enkel manuell update om vi vill vara exakta.
            
            # Låt oss köra utan jitter för enkelhet och stabilitet i koden, 
            # eller väldigt litet jitter.
            
            optimizer.step()
            
            # Normalisera/Clampa?
            # DeepDream brukar normalisera gradienterna.
            
        # Spara detaljerna som vi "drömt" fram
        # detail = current_img - octave_base
        detail = current_img - octave_base
        
        # Ta bort hooks
        for h in hooks:
            h.remove()
            
    return deprocess_image(current_img)
