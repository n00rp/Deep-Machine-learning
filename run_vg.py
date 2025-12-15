import torch
import matplotlib.pyplot as plt
from src.model import load_model
from src.preprocessing import load_image
from src.activation_maximization import activation_maximization
from src.deep_dream import deep_dream

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
    print("=== Startar VG-delen: Activation Max & DeepDream ===")
    
    # 1. Setup
    device = get_safe_device()
    print(f"Använder enhet: {device}")
    
    model = load_model(device)
    
    # 2. Activation Maximization Experiment
    print("\n--- Del 1: Activation Maximization ---")
    
    # Testa flera lager för att visa hierarki
    layers_to_test = ['features.5', 'features.10', 'features.20', 'features.28']
    filters_per_layer = {
        'features.5': [0, 10, 20],      # Tidiga lager: enkla mönster (64 filter totalt)
        'features.10': [15, 45, 80],    # Mellanlager: texturer (128 filter totalt)
        'features.20': [100, 200, 250], # Djupare lager: komplexa former (256 filter totalt)
        'features.28': [15, 60, 250]    # Sista lagret: semantiska delar (512 filter totalt)
    }
    
    am_results = []
    am_titles = []
    
    for layer in layers_to_test:
        filters = filters_per_layer[layer]
        for f_idx in filters:
            print(f"Maximerar filter {f_idx} i {layer}...")
            # Spara progression för ett filter per lager
            save_prog = (f_idx == filters[0])  # Spara bara för första filtret
            try:
                am_img = activation_maximization(
                    model, layer, f_idx, 
                    iterations=500, 
                    device=device,
                    save_progress=save_prog,
                    output_dir=f'outputs/activation_max/{layer}'
                )
                am_results.append((layer, f_idx, am_img))
                am_titles.append(f"{layer}: Filter {f_idx}")
            except (RuntimeError, torch.AcceleratorError) as e:
                print(f"GPU fel: {e}. Testar CPU...")
                am_model = load_model('cpu')
                am_img = activation_maximization(
                    am_model, layer, f_idx, 
                    iterations=500, 
                    device='cpu',
                    save_progress=save_prog,
                    output_dir=f'outputs/activation_max/{layer}'
                )
                am_results.append((layer, f_idx, am_img))
                am_titles.append(f"{layer}: Filter {f_idx}")
            except IndexError as e:
                print(f"Filter {f_idx} existerar inte i {layer}: {e}")
                continue
    
    # Visa alla AM resultat i en grid
    print("\nVisar Activation Maximization resultat...")
    num_results = len(am_results)
    cols = 4
    rows = (num_results + cols - 1) // cols
    
    plt.figure(figsize=(16, 4 * rows))
    for i, (layer, f_idx, img) in enumerate(am_results):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.title(f"{layer.split('.')[1]}:{f_idx}", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/activation_max/all_results.png')
    plt.show()
    
    # 3. DeepDream Experiment
    print("\n--- Del 2: DeepDream ---")
    img_path = 'data/images/castle.png'
    dream_img = None
    
    try:
        img_tensor = load_image(img_path, device)
        layers_to_dream = ['features.24', 'features.28']
        print(f"Drömmer med lager: {layers_to_dream}")
        
        try:
            dream_img = deep_dream(model, img_tensor, layers_to_dream, iterations=10, num_octaves=3, device=device)
        except (RuntimeError, torch.AcceleratorError) as e:
            print(f"GPU fel vid DeepDream: {e}. Faller tillbaka till CPU...")
            cpu_device = torch.device('cpu')
            # Ladda modellspecifikt för CPU
            model_cpu = load_model(cpu_device)
            img_cpu = load_image(img_path, cpu_device)
            dream_img = deep_dream(model_cpu, img_cpu, layers_to_dream, iterations=10, num_octaves=3, device=cpu_device)
            
    except FileNotFoundError:
        print("Bild saknas.")

    # 4. Visa DeepDream resultat separat
    if dream_img is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(dream_img)
        plt.title("DeepDream Resultat", fontsize=14)
        plt.axis('off')
        plt.savefig('outputs/deep_dream/dream_result.png')
        plt.show()
    
    print("\nVG-delen klar! Resultat sparade i outputs/ mappen.")

if __name__ == "__main__":
    main()
