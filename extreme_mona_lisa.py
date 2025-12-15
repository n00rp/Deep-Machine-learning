import torch
import matplotlib.pyplot as plt
import numpy as np
from src.model import load_model
from src.preprocessing import load_image, deprocess_image
from src.deep_dream import deep_dream
import os
from torchvision import transforms
from PIL import Image

def get_safe_device():
    """
    Checks if GPU is actually usable for VGG16. If not, falls back to CPU.
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    device = torch.device('cuda')
    try:
        # Minimal test
        m = torch.nn.Conv2d(3, 64, kernel_size=3).to(device)
        t = torch.randn(1, 3, 32, 32).to(device)
        _ = m(t)
        print("GPU check passed.")
        return device
    except Exception as e:
        print(f"GPU check failed ({e}). Falling back to CPU for stability.")
        return torch.device('cpu')

def recursive_deep_dream(model, img_tensor, layers, iterations=20, num_octaves=4, 
                        lr=0.15, num_passes=5, device='cuda'):
    """
    Kör DeepDream rekursivt för att skapa extremt intensiva hallucinationer
    """
    current_img = img_tensor.clone()
    results = []
    
    print(f"Startar rekursiv DeepDream med {num_passes} pass...")
    
    # Transform för att konvertera numpy array tillbaka till tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for pass_num in range(num_passes):
        print(f"\n--- Pass {pass_num + 1}/{num_passes} ---")
        
        # Kör DeepDream med nuvarande bild
        dream_img = deep_dream(
            model, current_img, layers,
            iterations=iterations,
            num_octaves=num_octaves,
            lr=lr,
            device=device
        )
        
        # Debug: Kontrollera shape
        print(f"Debug: dream_img.shape = {dream_img.shape}")
        print(f"Debug: dream_img.dtype = {dream_img.dtype}")
        print(f"Debug: min/max värden = {dream_img.min():.3f} / {dream_img.max():.3f}")
        
        # Spara numpy-array för visning
        results.append(dream_img)
        
        # Spara mellanresultat
        plt.figure(figsize=(10, 8))
        plt.imshow(dream_img)
        plt.title(f"Rekursiv Dream - Pass {pass_num + 1}", fontsize=14)
        plt.axis('off')
        os.makedirs(f'outputs/extreme_dream/pass_{pass_num+1}', exist_ok=True)
        plt.savefig(f'outputs/extreme_dream/pass_{pass_num+1}/mona_lisa_pass_{pass_num+1}.png', 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        # Konvertera tillbaka till tensor för nästa pass
        # dream_img är redan numpy array i (H, W, 3) format med värden 0-1
        dream_pil = Image.fromarray((dream_img * 255).astype(np.uint8))
        current_img = transform(dream_pil).unsqueeze(0).to(device)
        
        # Lägg till lite variation för att undvika repetitiva mönster
        if pass_num < num_passes - 1:
            # Lätt jitter
            noise = torch.randn_like(current_img) * 0.02
            current_img = torch.clamp(current_img + noise, 0, 1)
    
    return results

def main():
    print("=== EXTREME MONA LISA DEEPDREAM ===")
    print("Förbereder för att skapa galna hallucinationer...\n")
    
    # Setup
    device = get_safe_device()
    print(f"Använder enhet: {device}")
    
    model = load_model(device)
    
    # Ladda bild
    img_path = 'data/images/unnamed.jpg'
    print(f"Laddar bild: {img_path}")
    
    try:
        img_tensor = load_image(img_path, device)
    except FileNotFoundError:
        print(f"Bild saknas: {img_path}")
        return
    
    # Skapa outputs-mapp
    os.makedirs('outputs/extreme_dream', exist_ok=True)
    
    # Experiment 1: Klassiska deep lager
    print("\n=== EXPERIMENT 1: Klassiska djupa lager ===")
    layers_classic = ['features.24', 'features.28']
    results1 = recursive_deep_dream(
        model, img_tensor, layers_classic,
        iterations=30, num_octaves=5, lr=0.15, 
        num_passes=5, device=device
    )
    
    # Experiment 2: Endast djupaste lagret
    print("\n=== EXPERIMENT 2: Endast djupaste lagret ===")
    layers_deep = ['features.28']
    results2 = recursive_deep_dream(
        model, img_tensor, layers_deep,
        iterations=25, num_octaves=4, lr=0.12,
        num_passes=4, device=device
    )
    
    # Experiment 3: Mellan-lager för annorlunda texturer
    print("\n=== EXPERIMENT 3: Mellan-lager ===")
    layers_mid = ['features.20', 'features.24']
    results3 = recursive_deep_dream(
        model, img_tensor, layers_mid,
        iterations=35, num_octaves=4, lr=0.18,
        num_passes=4, device=device
    )
    
    # Visa alla slutresultat
    print("\n=== VISAR ALLA SLUTRESULTAT ===")
    
    # Samla slutresultaten
    final_results = [
        (results1[-1], "Klassiska lager (5 pass)"),
        (results2[-1], "Endast djupaste (4 pass)"),
        (results3[-1], "Mellan-lager (4 pass)")
    ]
    
    # Skapa grid med alla resultat
    plt.figure(figsize=(18, 6))
    for i, (img, title) in enumerate(final_results):
        plt.subplot(1, 3, i+1)
        plt.imshow(img)
        plt.title(title, fontsize=14)
        plt.axis('off')
        
        # Spara separat
        safe_title = title.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f'outputs/extreme_dream/final_{safe_title}.png', 
                   bbox_inches='tight', dpi=200)
    
    plt.tight_layout()
    plt.savefig('outputs/extreme_dream/all_extreme_results.png', 
               bbox_inches='tight', dpi=200)
    plt.show()
    
    # Skapa progression för ett experiment
    print("\nSkapar progression för klassiska lager...")
    plt.figure(figsize=(20, 4))
    for i in range(len(results1)):
        plt.subplot(1, len(results1), i+1)
        plt.imshow(results1[i])
        plt.title(f"Pass {i+1}", fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/extreme_dream/progression_classic.png', 
               bbox_inches='tight', dpi=150)
    plt.show()
    
    print("\n=== KLART! ===")
    print("Alla resultat sparade i outputs/extreme_dream/")
    print("- Enkla resultat: dream_[typ].png")
    print("- Progression: progression_classic.png")
    print("- Mellanresultat i pass_1/, pass_2/, etc.")

if __name__ == "__main__":
    main()
