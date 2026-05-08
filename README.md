# Extraction et Generation de Descriptions de Sinistres avec LLM

Projet de Deep Learning/IA générative de Master 2 ES
**Données:** https://www.kaggle.com/datasets/litvinenko630/insurance-claims

## Objectif

Transformer les données structurées de sinistres d'assurance en **descriptions textuelles coherentes et naturelles** en utilisant un Language Model. Au lieu de prédire des valeurs numeriques, le systeme génère du texte explicatif.

## Démarrage 

### 1. Prérequis: Installer Ollama

Ollama fournit un accès local gratuit à des LLMs sans coûts API.

**Installation:**
- Télecharger: https://ollama.ai
- Lancer Ollama: `ollama serve` (garder ce terminal ouvert)
- Dans un autre terminal, télécharger le modele: `ollama pull neural-chat`

### 2. Installation des dependencies Python

```bash
pip install -r requirements.txt
```

### 3. Générer des Descriptions

```bash
streamlit run app_llm_v2.py
```
Ouvrez: http://localhost:8501

## Pipeline Technique

```
Données brutes CSV
    |
    v
Preprocessing (DataProcessor)
    |
    v
Structuration en Dict
    |
    v
Création de Prompt structuré
    |
    v
Ollama API (localhost:11434)
    |
    v
Descriptions générées
    |
    v
Streamlit / CSV / Retour programmmatique
```
## Ressources

- [Ollama Documentation](https://ollama.ai)
- [Ollama Models](https://ollama.ai/library)
- [OpenAI Python Client](https://github.com/openai/openai-python)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## Credits

- Dataset: https://www.kaggle.com/datasets/litvinenko630/insurance-claims
- LLM: Ollama + Neural-chat/Mistral
- Interface: Streamlit
- Python Stack: pandas, numpy, scikit-learn, transformers