# ✅ OCR Project – Task Board

Koristimo ovaj fajl da pratimo napredak po sprintovima.  
Svaki član tima čekira ✅ kad završi svoj deo.

---

## 🟢 Sprint 1 – Osnova projekta i podaci
- [X] **A:** Kreirati strukturu foldera `src/ocr/...` i dodati `__init__.py`
- [X] **A:** Implementirati `load_mnist()` (normalizacija + split train/val/test)
- [X] **B:** Napraviti `test_mnist.py` i ispisati oblike i opseg piksela
- [X] **B:** Vizuelizovati 10 uzoraka i sačuvati u `artifacts/sample_images.png`

---

## 🟢 Sprint 2 – Model i trening
- [X] **A:** Implementirati `build_cnn()` u `cnn.py`
- [X] **B:** Napraviti `train_mnist.py` (fit, callback-ovi, čuvanje modela)
- [X] **B:** Dodati `CSVLogger` i potvrditi da se log čuva u `artifacts/training_log.csv`
- [X] **B:** Potvrditi da se model `mnist_cnn.keras` čuva u `models/`

---

## 🟢 Sprint 3 – Evaluacija i analiza
- [X] **A:** Napraviti `evaluate.py` (confusion matrix + classification report)
- [X] **B:** Napraviti `plot_training.py` (loss/accuracy krive iz CSV-a)
- [X] **A:** Iscrtati 10 pogrešno klasifikovanih primera i sačuvati u `artifacts/misclassified_samples.png`
- [x] **B:** Napisati kratak opis grešaka i dodati u README/izveštaj

---

## 🟢 Sprint 4 – Demo aplikacija
- [X] **A:** Implementirati backend deo – učitavanje modela i pretprocesiranje slike (single digit)
- [X] **A:** Dodati funkciju za segmentaciju i predikciju niza cifara (`pipeline.py`)
- [X] **B:** Implementirati frontend deo (Streamlit upload + top-3 predikcija + checkbox za višecifarski mod)
- [ ] **A+B:** Testirati na različitim računarima

---

## 🟢 Sprint 5 – Dokumentacija i prezentacija
- [X] **A:** Napraviti deo izveštaja o modelu (arhitektura, broj parametara, augmentacija)
- [x] **B:** Napraviti deo izveštaja o evaluaciji (grafici, matrica konfuzije, analiza grešaka)
- [x] **A+B:** Pripremiti prezentaciju (slajdovi sa grafikama i demo prikazom)
- [x] **A+B:** Finalno testiranje celog projekta

---

## 📌 Trenutno stanje
✅ Završeno:  
- Struktura projekta i `load_mnist()`  
- Definicija CNN modela (`cnn.py`)  
- Trening skripta + callback-ovi + snimanje modela  
- Augmentacija podataka i trening sa >99% tačnosti  
- Evaluacija, matrica konfuzije, i vizualizacija pogrešnih primera  
- Backend pretprocesiranje i `predict_single` za jednu cifru  

