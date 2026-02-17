# ❤️ CardioSentinel: Heart Disease Prediction with XGBoost

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

**CardioSentinel** è un sistema di Machine Learning progettato per supportare la diagnosi precoce di malattie cardiache. Il progetto utilizza dati clinici oggettivi per identificare pazienti a rischio con un focus specifico sulla massimizzazione della **Recall** (Sensibilità), minimizzando i falsi negativi potenzialmente letali.

---

## 🎯 Obiettivi del Progetto
* **Analisi Critica:** Confronto tra dati soggettivi (Sondaggi CDC) e dati clinici oggettivi (Cleveland Dataset).
* **Modellazione:** Sviluppo e confronto di pipeline basate su *Decision Tree*, *Random Forest* e *XGBoost*.
* **Sicurezza:** Ottimizzazione del modello per garantire un'alta sensibilità clinica.
* **Explainability:** Utilizzo di SHAP per interpretare le decisioni del modello ("Black Box" vs "White Box").

---

## 📂 Struttura della Repository
* `data_engineering.py`: Script per la pulizia, normalizzazione e split dei dati.
* `modello_finale.py`: Pipeline completa che addestra i modelli, genera i grafici di confronto e l'analisi SHAP.
* `grafico_confronto.py`: Script ausiliario per la visualizzazione delle metriche comparate.
* `heart_cleveland.csv`: Dataset originale.
* `tesi_*.png`: Grafici generati per la documentazione (Matrice di confusione, ROC, Feature Importance).

---

## 🚀 Come Eseguire il Codice

1. **Clona la repository:**
   ```bash
   git clone [https://github.com/simone-d-a/CardioSentinel.git](https://github.com/simone-d-a/CardioSentinel.git)
   cd CardioSentinel
2. Installa le dipendenze:  
      ```bash
       pip install pandas matplotlib seaborn scikit-learn xgboost shap
4. Prepara i dati:
    ```bash
      python data_engineering.py
5. Addestra il modello e genera i risultati:
   ```bash
    python modello_finale.py

## 👨‍💻 Autore
Simone Domenico Avitabile - Università degli Studi di Salerno Matr.0512120134
