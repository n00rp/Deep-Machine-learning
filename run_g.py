import torch
import matplotlib.pyplot as plt
from src.model import load_model
from src.preprocessing import load_image, show_image
from src.cam import generate_cam, visualize_cam

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

def main():
    print("=== Startar G-delen: Feature Attribution (CAM) ===")
    
    # 1. Setup
    device = get_safe_device()
    print(f"Använder enhet: {device}")
    
    model = load_model(device)
    
    # 2. Ladda Data
    img_path1 = 'data/images/dog.png'
    img_path2 = 'data/images/castle.png'
    
    try:    
        img_tensor1 = load_image(img_path1, device)
        img_tensor2 = load_image(img_path2, device)
    except FileNotFoundError:
        print("Kunde inte hitta bilder. Kör du scriptet från projektroten?")
        return

    # 3. Kör CAM Experiment & Samla resultat
    results = []
    titles = []
    
    try:
        print("\n--- Experiment 1: Hund ---")
        # Sista conv-lagret
        heatmap1 = generate_cam(model, img_tensor1, 'features.28')
        res1 = visualize_cam(heatmap1, img_tensor1, show=False)
        results.append(res1)
        titles.append("Hund - features.28")
        
        # Tidigt lager
        heatmap2 = generate_cam(model, img_tensor1, 'features.10')
        res2 = visualize_cam(heatmap2, img_tensor1, show=False)
        results.append(res2)
        titles.append("Hund - features.10")
        
        print("\n--- Experiment 2: Slott ---")
        heatmap3 = generate_cam(model, img_tensor2, 'features.28')
        res3 = visualize_cam(heatmap3, img_tensor2, show=False)
        results.append(res3)
        titles.append("Slott - features.28")
        
        heatmap4 = generate_cam(model, img_tensor2, 'features.10')
        res4 = visualize_cam(heatmap4, img_tensor2, show=False)
        results.append(res4)
        titles.append("Slott - features.10")
        
        # Visa alla i en grid
        print("\nVisar samlad vy...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, ax in enumerate(axes.flat):
            if i < len(results):
                ax.imshow(results[i])
                ax.set_title(titles[i])
                ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        print("\nG-delen klar! Stäng fönstret för att avsluta.")
    except Exception as e:
        print(f"\nEtt fel inträffade under körning: {e}")
        if device.type == 'cuda':
            print("Tips: Prova att tvinga CPU om problemet kvarstår.")

if __name__ == "__main__":
    main()
