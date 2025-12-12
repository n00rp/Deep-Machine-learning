# Utvecklingsplan – CNN‑tolkbarhet, Activation Maximization & DeepDream (VGG16)

## Översikt
Detta dokument beskriver en **komplett utvecklingsplan** för att lösa laborationen i djupinlärning med fokus på:
- CNN‑tolkbarhet (interpretability)
- Feature attribution (CAM / Grad‑CAM) – betyg **G**
- Feature visualization via **activation maximization** – betyg **VG**
- Skapande av **DeepDream‑liknande bilder**

Vi använder **VGG16** som basmodell.

---

## Mål
- Förstå hur olika lager i ett CNN arbetar
- Visualisera aktiveringar och attribution
- Implementera **gradient ascent på inputbilden**
- Skapa färgglada, visuellt intressanta DeepDream‑bilder
- Leverera en tydlig och körbar rapport (Jupyter Notebook eller liknande)

---

## Teknisk stack
- Python 3.10+
- PyTorch
- torchvision
- matplotlib
- numpy
- torch‑cam (för G‑delen)

---

## Projektstruktur (rekommenderad)

```
project/
│
├── notebooks/
│   └── interpretability_vgg16.ipynb
│
├── src/
│   ├── model.py
│   ├── preprocessing.py
│   ├── cam.py
│   ├── activation_maximization.py
│   └── deep_dream.py
│
├── data/
│   └── images/
│       ├── image1.jpg
│       └── image2.jpg
│
├── outputs/
│   ├── cam/
│   └── deep_dream/
│
└── README.md
```

---

## Steg 0 – Setup & Miljö
**Syfte:** säker utvecklingsmiljö

- Skapa virtuell miljö
- Installera beroenden
- Verifiera CUDA (om GPU finns)

**Klart när:**
- Du kan ladda VGG16 och göra inference på en bild

---

## Steg 1 – Modell: VGG16
**Val:** Pretrained VGG16 från torchvision

Varför VGG16?
- Tydlig lagerstruktur
- Mycket bra för visualisering
- Vanlig i DeepDream‑sammanhang

**Att göra:**
- Ladda modellen med `pretrained=True`
- Sätt `model.eval()`
- Inspektera `model.features`

**Klart när:**
- Du vet exakt vilka conv‑lager du vill visualisera

---

## Steg 2 – Data (bilder)
**Krav:**
- Minst 2 bilder
- Gärna olika motiv (djur, ansikte, natur, byggnad)

**Att göra:**
- Lägg bilder i `data/images/`
- Implementera:
  - `load_image`
  - `preprocess_image`
  - `deprocess_image`

**Klart när:**
- Bild → tensor → modell → prediction fungerar

---

## Steg 3 – Gemensam visualiserings‑pipeline
Bygg återanvändbara funktioner:

- Bildinläsning & normalisering
- Modellprediktion (top‑k)
- Hook‑mekanism för att få ut aktiveringar
- Visualisering av tensors som bilder

**Klart när:**
- Samma pipeline används för CAM och DeepDream

---

## Steg 4A – Betyg G: Feature Attribution (CAM / Grad‑CAM)

**Mål:**
- Visualisera minst **2 lager**
- Testa minst **2 bilder**

**Metod:**
- Använd `torch‑cam`
- Välj:
  - Ett tidigt conv‑lager
  - Ett sent conv‑lager

**Analys:**
- Vad tittar modellen på?
- Skillnader mellan lager
- Skillnader mellan bilder

**Output:**
- Heatmaps ovanpå originalbilder
- Tydlig textanalys

---

## Steg 4B – Betyg VG: Activation Maximization (kärnan)

### Grundidé
Optimera **inputbilden** så att ett visst lager/filter aktiveras maximalt.

### Algoritm (gradient ascent)
1. Initiera bild (slump eller befintlig)
2. Sätt `requires_grad=True`
3. Forward pass
4. Loss = +mean(aktivering)
5. Backprop till input
6. Uppdatera inputbilden
7. Clamp / regularisera
8. Upprepa

### Experiment
- Olika lager (tidigt vs sent)
- Olika filter
- Visa progression (iteration 0 / 50 / 100)

**Klart när:**
- Du kan visa tydliga mönster som maximerar aktivering

---

## Steg 5 – DeepDream (färgglada drömbilder)

DeepDream = Activation maximization + visuella tricks

### Förbättringar
- **Jitter:** slumpmässig förskjutning
- **Oktaver:** multi‑scale dreaming
- **Regularisering:**
  - L2‑loss
  - Total variation loss
- **Clamp:** håll pixelvärden stabila

### Varianter
- Start från riktig bild
- Start från brus
- Optimera mot:
  - lager
  - filter
  - klasslogit

**Output:**
- Färgglada, surrealistiska bilder
- Spara i `outputs/deep_dream/`

---

## Steg 6 – Rapport / Notebook‑struktur

1. Introduktion (interpretability)
2. Modell & data
3. Feature Attribution (G‑del)
4. Activation Maximization (VG‑del)
5. DeepDream‑experiment
6. Analys & slutsats

**Viktigt:**
- Visa bilder
- Resonera i text
- Motivera val av lager

---

## Bedömningschecklista

### G
- [ ] Minst 2 lager
- [ ] Minst 2 bilder
- [ ] CAM / Grad‑CAM
- [ ] Tydlig analys

### VG
- [ ] Egen gradient ascent
- [ ] Activation maximization
- [ ] Flera lager/filter
- [ ] DeepDream‑liknande bilder
- [ ] Resonemang i distill.pub‑stil

---

## Slutsats
Denna plan leder till:
- Uppfyllt labbkrav (G och VG)
- Djup förståelse för CNN‑hierarkier
- Möjlighet att skapa visuellt imponerande DeepDream‑bilder

---

Lycka till 🚀
