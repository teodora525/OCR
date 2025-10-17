# Handwritten Text Recognition (OCR)

Projekt: Handwritten text recognition system  
Repo: paniicj0/OCR

Opis
----
Ovaj repozitorijum sadrži implementaciju sistema za prepoznavanje rukom pisanih cifara (Optical Character Recognition —  OCR). Cilj je obraditi slike rukom pisanih zapisa i iz njih izvući cifre pomoću modela mašinskog  učenja (npr. CNN+RNN+CTC, Transformer ili sličnih arhitektura).

Sadržaj README-a
-----------------
- Kratak pregled i karakteristike
- Preduslovi
- Instalacija
- Korišćenje 
- Pretprocesiranje i augmentacija
- Licenca
- Kontakt

Ključne karakterstike
----------------
- Preprocesiranje slika (normalizacija, binarizacija, augmentacija)
- Mogućnost treninga i izvođenja 
- Podrška za modele sa gubitkom
- Evaluacija metrikama kao što su CER (Character Error Rate) i WER (Word Error Rate)
- Primjeri ulaza/izlaza i jednostavan API za korištenje

Preduslovi
---------
Preporučeno okruženje:
- Python 3.8+
- GPU (za ubrzani trening) s instaliranim NVIDIA driverima + CUDA (ako koristite TensorFlow)
- Virtualno okruženje (venv / conda)

Tipične biblioteke (zavisno od implementacije):
- numpy
- scipy
- Pillow (PIL)
- OpenCV (cv2)
- TensorFlow / Keras
- torchvision (ako koristite PyTorch)
- pandas (za rad s anotacijama)
- tqdm

Instalacija
----------
1. Klonirajte repozitorijum:
   git clone https://github.com/paniicj0/OCR.git
   cd OCR

2. Kreirajte i aktivirajte virtualno okruženje:
   python -m venv .venv
   source .venv/bin/activate  # Linux / macOS
   .venv\Scripts\activate     # Windows (PowerShell)

3. Instalirajte zahteve:
   pip install -r requirements.txt

Korišćenje
---------

1) Trening
Jednostavni poziv trening skripte:
   python src/ocr/train/train_mnist.py  --output-dir artifacts/

Mogući argumenti u train_mnist.py:
- --config : putanja do YAML/JSON konfiguracije (model, optimizer, lr, batch size, augmentacije)
- --resume : putanja do checkpointa za nastavak
- --gpu : id GPU-a ili lista

2) Evaluacija
   python src/eval/evaluate.py 

Pretprocesiranje i augmentacija
-------------------------
Preporučene tehnike:
- Promena veličine / normalizacija visine (zadržavanje mera)
- Adaptive thresholding ili contrast adjustment
- Random cropping, rotation (male rotacije), elastic distortions 
- Random brightness/contrast/noise


Licenca
-------
MIT License — pogledajte LICENSE datoteku.

Kontakt
-------
Autori: paniicj0 (https://github.com/paniicj0) i teodora525 (https://github.com/teodora525)

Za dodatna pitanja pišite issue na GitHub repo ili kontaktirajte autore.


