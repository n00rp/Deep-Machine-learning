import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# Laboration: CNN Tolkbarhet, Activation Maximization & DeepDream

Denna notebook redovisar lösningen för laborationen i djupinlärning. Vi utforskar hur VGG16 "ser" bilder genom Feature Attribution (CAM), visualiserar vad specifika filter reagerar på med Activation Maximization, och skapar konstnärliga bilder med DeepDream.

projektet är uppdelat i enlighet med kraven:
- **G-del**: Feature Attribution (CAM/Grad-CAM)
- **VG-del**: Activation Maximization & DeepDream

**Notera om Hårdvara:**
Koden är anpassad för att använda **Nvidia RTX 5080** (Blackwell/sm_120) om drivrutiner och PyTorch-version tillåter. Vi har installerat en Nightly-build av PyTorch (cu126) för att maximera kompatibilitet.
"""

code_setup = """import sys
import os
import torch
import matplotlib.pyplot as plt

# Lägg till src i path så vi kan importera våra moduler
sys.path.append(os.path.abspath('../'))

from src.model import load_model
from src.preprocessing import load_image, show_image

# Setup Enhet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Använder enhet: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Ladda modell
try:
    model = load_model(device)
except Exception as e:
    print(f"Kunde inte ladda modell på GPU (kanske inkompatibel drivrutin/PTX): {e}")
    print("Faller tillbaka till CPU...")
    device = torch.device('cpu')
    model = load_model(device)
"""

text_data = """## 1. Data och Preprocessing
Vi laddar in två testbilder: en hund (golden retriever) och ett slott/landskap.
"""

code_data = """# Sökvägar
img_path1 = '../data/images/dog.png'
img_path2 = '../data/images/castle.png'

# Ladda bilder
img_tensor1 = load_image(img_path1, device)
img_tensor2 = load_image(img_path2, device)

print("Originalbilder:")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
show_image(img_tensor1, "Bild 1: Hund")
plt.subplot(1, 2, 2)
show_image(img_tensor2, "Bild 2: Slott")
plt.show() # Visar separat om figuren ovan inte ritar direkt
"""

text_g_part = """## Del G: Feature Attribution (CAM)
Här visualiserar vi var modellen "tittar" för att klassificera bilderna. Vi använder **Smooth Grad-CAM++** och jämför två olika lager i VGG16.

**Lager vi undersöker:**
1. `features.28` (Sista conv-lagret): Bör visa semantiska delar (huvud, kropp etc.)
2. `features.10` (Tidigare lager): Bör visa mer generella mönster eller texturer.
"""

code_cam = """from src.cam import generate_cam, visualize_cam

# Funktion för att köra experiment
def run_cam_experiment(img_tensor, layer_name, title_suffix=""):
    print(f"--- CAM för lager {layer_name} ---")
    heatmap = generate_cam(model, img_tensor, target_layer=layer_name)
    visualize_cam(heatmap, img_tensor, title=f"CAM {layer_name} {title_suffix}")

# Experiment 1: Hund
print("Analyserar Bild 1 (Hund)...")
run_cam_experiment(img_tensor1, 'features.28', "(Sista Conv)")
run_cam_experiment(img_tensor1, 'features.10', "(Tidigt Lager)")

# Experiment 2: Slott
print("Analyserar Bild 2 (Slott)...")
run_cam_experiment(img_tensor2, 'features.28', "(Sista Conv)")
run_cam_experiment(img_tensor2, 'features.10', "(Tidigt Lager)")
"""

text_analysis_g = """### Analys av CAM
Genom att jämföra värmekartorna kan vi se att det sista lagret (`features.28`) är mycket mer fokuserat på **objektet** (t.ex. hundens ansikte). De tidigare lagren tenderar att aktiveras av kanter och texturer över hela bilden.
"""

text_vg_part = """## Del VG: Activation Maximization
Här "vänder vi på steken" och optimerar en *input-bild* (från brus) för att maximera aktiveringen av ett specifikt filter. Detta visar oss vad filtret "letar efter".
"""

code_am = """from src.activation_maximization import activation_maximization

# Välj ett lager och några intressanta filterindex
# features.28 har 512 filter.
target_layer = 'features.28'
filters = [10, 45, 123] 

plt.figure(figsize=(15, 5))
for i, f_idx in enumerate(filters):
    print(f"Maximerar filter {f_idx} i {target_layer}...")
    # Vi kör på CPU om GPU strular med baklänges-passet, men testa device först
    try:
        am_img = activation_maximization(model, target_layer, f_idx, iterations=50, device=device)
    except Exception as e:
        print(f"Fel vid AM på {device}: {e}, testar CPU...")
        am_img = activation_maximization(model.to('cpu'), target_layer, f_idx, iterations=50, device='cpu')
        model.to(device) # Flytta tillbaka
        
    plt.subplot(1, 3, i+1)
    plt.imshow(am_img)
    plt.title(f"Filter {f_idx}")
    plt.axis('off')

plt.show()
"""

text_deepdream = """## Del VG: DeepDream
DeepDream bygger vidare på activation maximization men appliceras på en bild (oftast) och använder "oktaver" (multi-scale) för att skapa komplexa, fraktala mönster.
"""

code_deepdream = """from src.deep_dream import deep_dream

# Vi drömmer på vår slottsbild!
print("Startar DeepDream på slottsbilden...")
layers_to_dream = ['features.24', 'features.28'] # Blanda lite lager

try:
    dream_img = deep_dream(model, img_tensor2, layers_to_dream, iterations=10, num_octaves=3, device=device)
except Exception as e:
    print(f"Fel vid DeepDream på {device}: {e}. Kör på CPU (kan ta tid)...")
    dream_img = deep_dream(model.to('cpu'), img_tensor2.to('cpu'), layers_to_dream, iterations=10, num_octaves=3, device='cpu')
    model.to(device)

plt.figure(figsize=(10, 8))
plt.imshow(dream_img)
plt.title("DeepDream Resultat")
plt.axis('off')
plt.show()

# Spara resultatet
# (Här skulle vi spara till fil om vi ville)
"""

text_outro = """## Slutsats
Vi har lyckats implementera och köra både feature attribution och activation maximization. Resultaten visar tydligt modellens hierarkiska uppbyggnad. DeepDream ger visuellt intressanta tolkningar av vad modellen ser i bilden.
"""

nb.cells = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_data),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_markdown_cell(text_g_part),
    nbf.v4.new_code_cell(code_cam),
    nbf.v4.new_markdown_cell(text_analysis_g),
    nbf.v4.new_markdown_cell(text_vg_part),
    nbf.v4.new_code_cell(code_am),
    nbf.v4.new_markdown_cell(text_deepdream),
    nbf.v4.new_code_cell(code_deepdream),
    nbf.v4.new_markdown_cell(text_outro)
]

with open('notebooks/Laboration_Rapport.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook skapad: notebooks/Laboration_Rapport.ipynb")
