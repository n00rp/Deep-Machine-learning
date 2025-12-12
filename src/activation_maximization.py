import torch
import torch.nn as nn
import torch.optim as optim
from src.preprocessing import deprocess_image
import matplotlib.pyplot as plt
import numpy as np

def activation_maximization(model, layer_name, filter_index, iterations=100, learning_rate=0.1, device='cpu'):
    """
    Utför gradient ascent på en slumpmässig inputbild för att maximera aktiveringen
    av ett specifikt filter i ett givet lager.
    """
    model.eval()
    
    # 1. Initiera slumpmässig bild med requires_grad=True
    # Vi startar med grått brus (0.5 +- 0.1)
    input_tensor = (torch.rand(1, 3, 224, 224, device=device) * 0.2 + 0.4).detach()
    input_tensor.requires_grad = True
    
    # Optimizer (Adam fungerar ofta bättre än ren SGD för detta)
    optimizer = optim.Adam([input_tensor], lr=learning_rate)
    
    # Hitta lagret vi vill aktivera
    # Vi måste registrera en hook för att få ut aktiveringen
    activations = {}
    def hook_fn(module, input, output):
        activations['value'] = output
    
    # Hitta modulen
    layer_module = None
    # Hantera features.X notation för VGG
    if 'features' in layer_name:
        idx = int(layer_name.split('.')[1])
        layer_module = model.features[idx]
    else:
        print(f"Lager {layer_name} stöds ej direkt i denna enkla implementering för VGG.")
        return None
        
    handle = layer_module.register_forward_hook(hook_fn)
    
    print(f"Startar optimering av {layer_name}, filter {filter_index} i {iterations} iterationer...")
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Forward pass
        _ = model(input_tensor)
        
        # Hämta aktivering
        layer_out = activations['value']
        
        # Vi vill maximera medelvärdet av aktiveringen för valt filter
        # Loss = -activation (eftersom optimizer minimerar)
        loss = -torch.mean(layer_out[0, filter_index])
        
        loss.backward()
        
        optimizer.step()
        
        # Optional: Regularisering / Clamping för att hålla bilden "giltig"
        # Men för ren activation maximization låter vi den ofta löpa fritt, 
        # eller bara clampa till rimliga värden.
        # input_tensor.data.clamp_(0, 1) # Kan göra det svårare att optimera
        
        if i % 20 == 0:
            print(f"Iter {i}: Loss {loss.item():.4f}")
            
    handle.remove()
    
    # Returnera optimerad bild
    return deprocess_image(input_tensor)

if __name__ == '__main__':
    # Test
    from src.model import load_model
    try:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except:
        dev = torch.device('cpu')
    m = load_model(dev)
    img = activation_maximization(m, 'features.28', 10, iterations=20, device=dev)
    plt.imshow(img)
    plt.show()
