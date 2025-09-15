# ✅ OCR Project – Task Board

Koristimo ovaj fajl da pratimo napredak po sprintovima.  
Svaki član tima čekira ✅ kad završi svoj deo.

---

## 🟢 Sprint 1 – Osnova projekta i podaci
- [X] **A:** Kreirati strukturu foldera `src/ocr/...` i dodati `__init__.py`
- [ ] **A:** Implementirati `load_mnist()` (normalizacija + split train/val/test)
- [ ] **B:** Napraviti `test_mnist.py` i ispisati oblike i opseg piksela
- [ ] **B:** Vizuelizovati 10 uzoraka i sačuvati u `artifacts/sample_images.png`

---

## 🟢 Sprint 2 – Model i trening
- [ ] **A:** Implementirati `build_cnn()` u `cnn.py`
- [ ] **B:** Napraviti `train_mnist.py` (fit, callback-ovi, čuvanje modela)
- [ ] **B:** Dodati `CSVLogger` i potvrditi da se log čuva u `artifacts/training_log.csv`
- [ ] **B:** Potvrditi da se model `mnist_cnn.keras` čuva u `models/`

---

## 🟢 Sprint 3 – Evaluacija i analiza
- [ ] **A:** Napraviti `evaluate.py` (confusion matrix + classification report)
- [ ] **B:** Napraviti `plot_training.py` (loss/accuracy krive iz CSV-a)
- [ ] **A:** Iscrtati 10 pogrešno klasifikovanih primera i sačuvati u `artifacts/misclassified_samples.png`
- [ ] **B:** Napisati kratak opis grešaka i dodati u README/izveštaj

---

## 🟢 Sprint 4 – Demo aplikacija
- [ ] **A:** Implementirati backend deo – učitavanje modela i pretprocesiranje slike
- [ ] **B:** Implementirati frontend deo (Streamlit upload + top-3 predikcija)
- [ ] **A+B:** Testirati na različitim računarima

---

## 🟢 Sprint 5 – Dokumentacija i prezentacija
- [ ] **A+B:** Napraviti finalni izveštaj (opis modela, podataka, metrika)
- [ ] **A+B:** Pripremiti prezentaciju (grafici, matrica konfuzije, demo)
- [ ] **A+B:** Finalno testiranje celog projekta
