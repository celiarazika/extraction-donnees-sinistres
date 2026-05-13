# Génération de données de synthèse de sinistres et descriptions en LLM

Projet de Deep Learning/IA générative de Master 2 ES

**Données:** https://www.kaggle.com/datasets/litvinenko630/insurance-claims

## Objectif

Générer des données synthétiques de sinistres à partir d'une base de données réelle pour améliorer la tarification d'assurance automobile, en combinant:
- **ML/DL Génératif** (CTGAN, TVAE) pour la génération de données tabulaires synthétiques
- **XGBoost** pour la validation et la classification
- **LLM Local** (Ollama + phi3.5) pour enrichir les données avec des descriptions textuelles professionnelles

## Architecture du Projet

### 4 Étapes Complètes

1. **Étape 1 - Prétraitement** (`notebooks/etape1_preprocessing.ipynb`)
   - Nettoyage et contrôle qualité des données
   - One-hot encoding et normalisation
   - Analyse du déséquilibre des classes

2. **Étape 2 - EDA** (`notebooks/etape2_eda.ipynb`)
   - Analyse exploratoire complète
   - Visualisations et corrélations
   - Statistiques descriptives

3. **Étape 3 - Modélisation** (`notebooks/etape3_modelisation.ipynb`)
   - Génération CTGAN/TVAE de sinistres synthétiques
   - Entraînement XGBoost avec augmentation de données
   - Génération LLM de descriptions textuelles

4. **Étape 4 - Évaluation** (`notebooks/etape4_evaluation.ipynb`)
   - Comparaison des modèles (Baseline, CTGAN, TVAE, SMOTE)
   - Tests statistiques (KS test, courbes ROC/PR)
   - Analyse de sensibilité

## Démarrage Rapide

### 1. Prérequis: Installer Ollama (optionnel mais recommandé)

Ollama fournit un accès local gratuit aux LLMs sans coûts API.

**Installation:**
- Télécharger: https://ollama.com
- Lancer le service: `ollama serve` (garder ce terminal ouvert)
- Dans un autre terminal, télécharger le modèle: `ollama pull phi3.5`

### 2. Installation des dépendances Python

```bash
pip install -r requirements.txt
```

### 3. Lancer l'application interactive

```bash
streamlit run app.py
```
Ouvrez: http://localhost:8501

## Ressources et Références

- [Ollama](https://ollama.com) - LLM local et gratuit
- [Streamlit](https://docs.streamlit.io/) - Interface interactive
- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV) - Générateurs tabulaires CTGAN/TVAE
- [XGBoost](https://xgboost.readthedocs.io/) - Classification et évaluation
- [Scikit-learn](https://scikit-learn.org/) - ML et preprocessing
- [OpenAI Python Client](https://github.com/openai/openai-python) - Intégration LLM

## Crédits

- **Dataset**: https://www.kaggle.com/datasets/litvinenko630/insurance-claims (58,592 polices)
- **Architecture Générative**: CTGAN et TVAE (MIT SDV)
- **LLM Local**: Ollama + phi3.5 (Microsoft)
- **Interface**: Streamlit
- **Stack Python**: pandas, numpy, scikit-learn, xgboost, sdv, streamlit
