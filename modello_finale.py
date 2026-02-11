import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, recall_score, precision_score

# === CONFIGURAZIONE ===
IMG_DIR = "" # Lascia vuoto per salvare nella stessa cartella

def run_final_pipeline():
    print("--- INIZIO PIPELINE DI CONFRONTO MODELLI ---")

    # 1. CARICAMENTO DATI
    try:
        X_train = pd.read_csv('X_train_final.csv')
        X_test = pd.read_csv('X_test_final.csv')
        y_train = pd.read_csv('y_train_final.csv').values.ravel()
        y_test = pd.read_csv('y_test_final.csv').values.ravel()
    except FileNotFoundError:
        print("ERRORE: Non trovo i file dati. Esegui prima 'data_engineering.py'!")
        return

    # 2. DEFINIZIONE DEI 3 MODELLI
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # 3. ADDESTRAMENTO E CONFRONTO
    print("\nAddestramento modelli in corso...")
    results = []
    plt.figure(figsize=(10, 8))
    
    best_model = None 

    for name, model in models.items():
        # Addestramento
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Probabilità per ROC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred 
            
        # Metriche
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        
        results.append({"Modello": name, "Accuracy": acc, "Recall": rec, "Precision": prec})
        
        # Plot ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
        if name == "XGBoost":
            best_model = model

    # 4. SALVATAGGIO GRAFICO ROC COMPARATIVO
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falsi Allarmi (False Positive Rate)')
    plt.ylabel('Sensibilità (Recall)')
    plt.title('Confronto Performance: Curve ROC', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(f'{IMG_DIR}tesi_confronto_roc.png', dpi=300)
    plt.close()

    # 5. STAMPA TABELLA
    print("\n" + "="*65)
    print(f"{'MODELLO':<15} | {'ACCURACY':<10} | {'RECALL':<10} | {'PRECISION':<10}")
    print("-" * 65)
    for res in results:
        print(f"{res['Modello']:<15} | {res['Accuracy']:.2%}     | {res['Recall']:.2%}     | {res['Precision']:.2%}")
    print("="*65 + "\n")

    # 6. ANALISI APPROFONDITA VINCITORE (XGBoost)
    print("--- GENERAZIONE GRAFICI VINCITORE (XGBoost) ---")
    
    y_pred_best = best_model.predict(X_test)
    
    # A) Matrice di Confusione
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})
    plt.title('Matrice di Confusione (XGBoost)', fontsize=14)
    plt.ylabel('Realtà')
    plt.xlabel('Predizione')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}tesi_matrice_best.png', dpi=300)
    plt.close()
    
    # B) FEATURE IMPORTANCE (Il grafico nuovo che volevi!)
    # Creiamo un DataFrame con le feature e la loro importanza
    importances = best_model.feature_importances_
    feature_names = X_train.columns
    
    # Dizionario per tradurre i nomi brutti in Italiano per la tesi
    traduzione = {
        'cp': 'Tipo Dolore Petto',
        'thal': 'Talassemia',
        'ca': 'Num. Vasi Colorati',
        'oldpeak': 'Depressione ST (ECG)',
        'thalach': 'Battito Cardiaco Max',
        'age': 'Età',
        'chol': 'Colesterolo',
        'trestbps': 'Pressione a Riposo',
        'sex': 'Sesso',
        'slope': 'Pendenza ST',
        'exang': 'Angina da Sforzo',
        'restecg': 'ECG a Riposo',
        'fbs': 'Glicemia > 120'
    }
    
    # Creiamo la tabella per il grafico
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importanza': importances})
    # Traduciamo i nomi
    df_imp['Feature_IT'] = df_imp['Feature'].map(traduzione).fillna(df_imp['Feature'])
    # Ordiniamo dal più importante al meno importante
    df_imp = df_imp.sort_values('Importanza', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Disegniamo il grafico a barre
    sns.barplot(x="Importanza", y="Feature_IT", data=df_imp, palette="viridis")
    
    plt.title("Quali esami medici contano di più? (Feature Importance)", fontsize=14)
    plt.xlabel("Peso decisionale del modello (0-1)", fontsize=12)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}tesi_feature_importance.png', dpi=300)
    plt.close()

    print("\n--- TUTTO COMPLETATO ---")
    
if __name__ == "__main__":
    run_final_pipeline()