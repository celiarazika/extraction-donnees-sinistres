# Extraction et Generation de Descriptions de Sinistres avec LLM

Projet de Deep Learning/IA générative de Master 2 ES

**Données:** https://www.kaggle.com/datasets/litvinenko630/insurance-claims

## Objectif

Générer des données de synthèse de sinistres à partir de la base de données réelle jointe pour aider à la tarification et tansformer les données structurées de sinistres d'assurance en descriptions textuelles en utilisant un Language Model. Au lieu de prédire des valeurs numeriques, le systeme génère des données et/ou du texte explicatif.

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

### 3. Générer

```bash
streamlit run app_llm_v2.py
```
Ouvrez: http://localhost:8501

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
