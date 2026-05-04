# Extraction et Génération de Descriptions de Sinistres avec LLM
Projet de Deep Learning/IA générative de Master 2 - **Utilise un Language Model (LLM) pour générer des descriptions textuelles**

**Données:** https://www.kaggle.com/datasets/litvinenko630/insurance-claims

## 🎯 Objectif

Transformer les données structurées de sinistres d'assurance en **descriptions textuelles cohérentes et naturelles** en utilisant un Language Model. Au lieu de prédire des valeurs numériques, le système génère du texte explicatif.

### Exemple :

```
Données brutes:
  Type: Accident Auto
  Montant: €5000
  Statut: Ouvert
  Cause: Collision

Description générée:
"Un accident automobile a été déclaré avec un montant estimé de 5000 euros. 
Le sinistre résulte d'une collision et est actuellement en cours de traitement."
```

## 📁 Structure du Projet

```
extraction-donnees-sinistres/
├── src/                          # Code production (réutilisable)
│   ├── __init__.py
│   ├── data_processor.py         # Preprocessing (nettoyage, normalisation)
│   └── model.py                  # 🆕 Classe ClaimsLLMGenerator (génération LLM)
│
├── notebooks/                    # Exploration & expérimentation
│   ├── analysis.ipynb           # EDA et exploration
│
├── data/                         # Données
│   └── claims_with_descriptions.csv  # Résultats générés
│
├── train.py                      # 🆕 Script pour générer descriptions (sans entraînement)
├── app_llm.py                    # 🆕 Interface Streamlit LLM
├── requirements.txt              # Dépendances (transformers, torch)
└── README.md                     # Documentation
```

## 🚀 Démarrage Rapide

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

**Dépendances principales:**
- `transformers` - HuggingFace pour les modèles LLM
- `torch` - PyTorch (backend pour les LLM)
- `streamlit` - Interface web
- `pandas`, `scikit-learn` - Data processing

### 2. Générer des Descriptions

```bash
python train.py
```

Cela va:
- Charger les données brutes
- Initialiser le LLM (GPT-2 par défaut)
- Générer des descriptions pour 10 premiers sinistres (configurable)
- Sauvegarder les résultats dans `data/claims_with_descriptions.csv`

**Output:**
```
📋 Exemples:
Sinistre #1
  Description: "Un sinistre de type sinistre habitation avec un montant..."

Sinistre #2
  Description: "Accident automobile déclaré pour 5000 euros..."
```

### 3. Interface Web (Streamlit)

```bash
streamlit run app_llm.py
```

Ouvrez: http://localhost:8501

**Sections disponibles:**
- 🏠 **Accueil** - Vue d'ensemble du projet
- ✍️ **Générer Description** - Génération pour 1 sinistre
- 📁 **Batch Processing** - Traiter fichiers CSV complets
- 📊 **Informations** - Stats du système

## 💻 Utilisation Programmée

### Depuis un script Python

```python
from src.model import create_generator

# Créer le générateur
generator = create_generator(model_name='gpt2')

# Générer une description
claim_data = {
    "Type": "Accident Auto",
    "Montant": 5000,
    "Statut": "Ouvert",
    "Cause": "Collision"
}

description = generator.generate(claim_data, max_length=100)
print(description)
```

### Traitement batch

```python
import pandas as pd
from src.model import create_generator

# Charger les données
df = pd.read_csv('Insurance claims data.csv')

# Créer générateur
generator = create_generator()

# Générer descriptions
descriptions = generator.generate_batch(df.head(10).to_dict('records'))

# Ajouter au dataframe
df['description'] = descriptions
df.to_csv('output.csv', index=False)
```

## 🔧 Configuration

### Changer le modèle LLM

Dans `train.py` ou `app_llm.py`, modifiez `LLM_MODEL`:

```python
# Options:
LLM_MODEL = 'gpt2'                    # Default - léger et rapide
LLM_MODEL = 'distilgpt2'              # Plus petit/plus rapide
LLM_MODEL = 'openai'                  # API OpenAI (nécessite OPENAI_API_KEY)
LLM_MODEL = 'mistral-7b'              # Modèle HuggingFace plus puissant
```

### Configurer OpenAI (optionnel)

```bash
set OPENAI_API_KEY=sk-... (Windows PowerShell)
# ou
export OPENAI_API_KEY=sk-... (Linux/macOS)
```

## 📊 Workflow

```
┌─────────────────────────────┐
│  Données brutes CSV          │
│  (Insurance claims data)     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  Preprocessing              │
│  (nettoyage, normalisation) │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  LLM Generator              │
│  (transformers library)     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  Descriptions générées      │
│  (CSV ou API Streamlit)     │
└─────────────────────────────┘
```

## 🛠️ Développement

### Ajouter un nouveau modèle LLM

Éditer `src/model.py`:

```python
def _load_model(self):
    if self.model_name == "mon_modele":
        # Charger votre modèle
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained("mon_modele")
        self.model = AutoModelForCausalLM.from_pretrained("mon_modele")
```

### Améliorer les prompts

Éditer la fonction `create_prompt()` pour générer des prompts plus spécifiques:

```python
def create_prompt(self, claim_data: Dict) -> str:
    # Personnaliser le format du prompt
    prompt = "Voici les détails d'un sinistre:\n"
    # ... votre logique
    return prompt
```

### Modifier les paramètres de génération

Dans la fonction `generate()`:

```python
output = self.model.generate(
    input_ids,
    max_length=150,          # Augmenter pour texte plus long
    num_beams=8,             # Plus de beams = meilleure qualité mais plus lent
    temperature=0.8,         # 0.5 = plus déterministe, 1.0 = plus créatif
    top_p=0.95,             # Nucleus sampling
    do_sample=True,          # Activation du sampling
)
```

## 📚 Ressources

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [GPT-2 Model Card](https://huggingface.co/gpt2)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🐛 Dépannage

### "ModuleNotFoundError: No module named 'transformers'"
```bash
pip install transformers torch
```

### "CUDA out of memory"
Le modèle est trop gros pour votre GPU. Utilisez `distilgpt2` ou CPU:
```python
generator = create_generator(model_name='distilgpt2')
```

### Les descriptions ne sont pas pertinentes
Ajustez les paramètres de génération ou utilisez un modèle plus puissant

## 📋 Améliorations Futures

- [ ] Fine-tuning sur corpus de sinistres
- [ ] Support de plusieurs langues
- [ ] Génération avec contraintes (max mots, structure fixe)
- [ ] Classement qualité descriptions
- [ ] Intégration avec APIs externes
- [ ] Caching pour accélération
- [ ] Tests unitaires

## 📄 License

Projet académique - Master 2 IA/DL

---

**Questions?** Vérifiez `GETTING_STARTED.md` pour plus de détails.
