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

## 📊 Dataset e Preprocessing
Il progetto utilizza il **Cleveland Heart Disease Dataset (UCI)**.
A differenza dei dataset basati su sondaggi, questo set di dati contiene 14 parametri biomedici tra cui:
* `cp`: Tipo di dolore toracico.
* `thal`: Talassemia.
* `ca`: Numero di vasi colorati alla fluoroscopia.
* `oldpeak`: Depressione ST indotta dall'esercizio.

**Operazioni effettuate:**
1.  Pulizia dei valori mancanti.
2.  Data Engineering conservativo (nessun One-Hot Encoding eccessivo per mantenere l'interpretabilità).
3.  Normalizzazione MinMax.

---

## 🏆 Risultati Sperimentali

Dopo aver confrontato diverse architetture, **XGBoost** è stato selezionato come modello definitivo.

| Modello | Accuracy | Recall (Sensibilità) | Precision |
| :--- | :--- | :--- | :--- |
| Decision Tree | 75% | 57% | 84% |
| Random Forest | 81% | 75% | 84% |
| **XGBoost (CardioSentinel)** | **86%** | **82%** | **88%** |

> **Nota:** Sebbene la Random Forest mostri una robustezza generale simile, XGBoost è stato preferito per la sua superiore capacità di individuare i casi positivi (Recall 82% vs 75%).

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
   git clone [https://github.com/TUO_USERNAME/CardioSentinel.git](https://github.com/TUO_USERNAME/CardioSentinel.git)
   cd CardioSentinel