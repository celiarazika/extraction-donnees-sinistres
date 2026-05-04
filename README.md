# Extraction de données de sinistres à partir de contrats d'assurance
Projet de Deep Learning/IA générative de Master 2

**Données:** https://www.kaggle.com/datasets/litvinenko630/insurance-claims

## Structure du Projet

```
extraction-donnees-sinistres/
├── src/                          # Code production (réutilisable)
│   ├── __init__.py
│   ├── data_processor.py         # Preprocessing (nettoyage, encoding, normalisation)
│   └── model.py                  # Architecture du modèle Deep Learning
│
├── notebooks/                    # Exploration & expérimentation
│   ├── main.ipynb               # (À déplacer pour exploration uniquement)
│   └── analysis.ipynb           # (Optionnel: analyses spécifiques)
│
├── data/                         # Données
│   ├── X_train.csv
│   ├── X_test.csv
│   └── data_processed.csv
│
├── models/                       # Modèles et preprocesseurs
│   ├── insurance_model.h5       # Modèle entraîné
│   ├── scaler.pkl               # Normalisation
│   └── label_encoders.pkl       # Encodeurs catégoriels
│
├── train.py                      # Script d'entraînement standalone
├── app.py                        # Interface Streamlit (inférence)
└── README.md                     # Documentation

```

## Démarrage Rapide

### 1. Installation des dépendances

```bash
pip install pandas numpy scikit-learn tensorflow streamlit
```

### 2. Preprocessing et Entraînement

```bash
python train.py
```

Cela va:
- Charger les données
- Nettoyer et normaliser
- Entraîner le modèle Deep Learning
- Sauvegarder les résultats

### 3. Interface Web (Inférence)

```bash
streamlit run app.py
```

Puis ouvrez: http://localhost:8501

## Utilisation

### Depuis un script Python

```python
from src import DataProcessor, create_model, train_model

# Preprocessing
processor = DataProcessor()
df = processor.load_data('Insurance claims data.csv')
df_clean = processor.clean_data(df)
df_transformed = processor.transform_data(df_clean)

# Modélisation
model = create_model(input_dim=df_transformed.shape[1])
history = train_model(model, X_train, y_train)
```

### Depuis un Jupyter Notebook

Voir `notebooks/` pour des exemples d'exploration (optionnel)

## Workflow

```
[Données brutes]
      ↓
   train.py  (ou notebook pour exploration)
      ↓
[Preprocessing: nettoyage, encoding, normalisation]
      ↓
[Entraînement: tensorflow/keras]
      ↓
[Modèle + Preprocesseurs sauvegardés]
      ↓
   app.py  (Streamlit)
      ↓
[Interface d'inférence]
```

## Notes

- **`src/`** contient le code réutilisable et modulaire
- **`train.py`** script autonome pour entraînement (peut être lancé en production)
- **`app.py`** interface utilisateur pour inférence
- **Notebooks** sont utilisés uniquement pour exploration/expérimentation
- Les préprocesseurs (scaler, encodeurs) sont sauvegardés pour cohérence train/test

## 🛠 Développement Futur

- [ ] Ajouter validation cross-validation
- [ ] Implémenter des métriques métier
- [ ] Ajouter des tests unitaires
- [ ] Configuration avec `.env`
- [ ] Logging et monitoring
- [ ] API REST (FastAPI/Flask)
