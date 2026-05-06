# Extraction et Generation de Descriptions de Sinistres avec LLM

Projet de Deep Learning/IA generative de Master 2 - **Utilise un Language Model (LLM) pour generer des descriptions textuelles**

**Donnees:** https://www.kaggle.com/datasets/litvinenko630/insurance-claims

## Objectif

Transformer les donnees structurees de sinistres d'assurance en **descriptions textuelles coherentes et naturelles** en utilisant un Language Model. Au lieu de predire des valeurs numeriques, le systeme genere du texte explicatif.

### Exemple :

```
Donnees brutes:
  Type: Accident Auto
  Montant: EUR 5000
  Statut: Ouvert
  Cause: Collision

Description generee:
"Un accident automobile a ete declare avec un montant estime de 5000 euros. 
Le sinistre resulte d'une collision et est actuellement en cours de traitement."
```

## Structure du Projet

```
extraction-donnees-sinistres/
├── src/                          # Code production (reutilisable)
│   ├── __init__.py
│   ├── data_processor.py         # Preprocessing (nettoyage, normalisation)
│   └── model.py                  # Classe ClaimsLLMGenerator (generation LLM avec Ollama)
│
├── notebooks/                    # Exploration & experimentation
│   ├── analysis.ipynb            # EDA et exploration
│   ├── main.ipynb                # Notebook principal
│
├── data/                         # Donnees
│   ├── Insurance claims data.csv  # Dataset original
│   ├── claims_with_descriptions.csv  # Resultats generes
│   ├── X_train.csv
│   ├── X_test.csv
│
├── train.py                      # Script pour generer descriptions
├── app.py                        # Interface Streamlit basique
├── app_llm.py                    # Interface Streamlit LLM (ancienne)
├── app_llm_v2.py                 # Interface Streamlit LLM v2 (nouvelle - UTILISEZ CELLE-CI)
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
├── GETTING_STARTED.md            # Guide de demarrage
└── main.ipynb                    # Notebook demonstration
```

## Demarrage Rapide

### 1. Prerequis: Installer Ollama

Ollama fournit un acces local gratuit a des LLMs sans couts API.

**Installation:**
- Telecharger: https://ollama.ai
- Lancer Ollama: `ollama serve` (garder ce terminal ouvert)
- Dans un autre terminal, telecharger le modele: `ollama pull neural-chat`

**Verification:**
```bash
curl http://localhost:11434/api/tags
```

### 2. Installation des dependencies Python

```bash
pip install -r requirements.txt
```

**Dependencies principales:**
- `openai` - Client compatible avec Ollama API
- `streamlit` - Interface web interactive
- `pandas`, `numpy` - Data processing
- `python-dotenv` - Gestion des variables d'environnement
- `requests` - Verification Ollama

### 3. Generer des Descriptions

**Option A: Interface Web (Recommandee pour l'ecole)**
```bash
streamlit run app_llm_v2.py
```
Ouvrez: http://localhost:8501

**Sections disponibles:**
- [HOME] Accueil - Vue d'ensemble du projet et pipeline
- [TEST] Tester - Generation interactive pour 1 sinistre
- [STATS] Analyse batch - Traiter plusieurs sinistres avec metriques
- [INFO] A propos - Informations techniques et stats

**Option B: Script de traitement batch**
```bash
python train.py
```

Cela va:
- Verifier que Ollama est accessible
- Charger les donnees brutes
- Generer des descriptions pour 10 premiers sinistres
- Afficher les metriques de temps

**Output:**
```
Ollama trouve sur localhost:11434

Sinistre 1/10
  Description: "Un sinistre de type accident automobile avec un montant de EUR 5000..."
  Temps: 35.2s

Sinistre 2/10
  Description: "Declaration d'un sinistre habitation avec specialiste requis..."
  Temps: 38.1s

Temps total: 357.3s (35.7s par sinistre)
```

**Option C: Script Python personnalise**
```bash
python app.py
```

## Utilisation Programmee

### Depuis un script Python

```python
from src.model import create_generator

# Creer le generateur (Ollama par defaut)
generator = create_generator()

# Generer une description
claim_data = {
    "vehicle_age": 2.5,
    "customer_age": 45,
    "fuel_type": "Diesel",
    "transmission_type": "Manual",
    "airbags": 2,
    "ncap_rating": 4,
    "segment": "C",
    "is_esc": "Yes"
}

description = generator.generate(claim_data, max_length=300)
print(description)
```

### Traitement batch

```python
import pandas as pd
from src.model import create_generator
from src.data_processor import DataProcessor

# Charger les donnees
processor = DataProcessor()
df = processor.load_data('Insurance claims data.csv')

# Creer generateur
generator = create_generator()

# Generer descriptions
descriptions = generator.generate_batch(df.head(10).to_dict('records'))

# Ajouter au dataframe
df['description'] = descriptions
df.to_csv('sinistres_generes.csv', index=False)
```

## Configuration

### Backend LLM: Ollama

Ollama offre une API compatible OpenAI sur localhost:11434. **ZERO COUT - Execution locale gratuite.**

**Modeles disponibles dans Ollama:**

| Modele | Vitesse | Qualite | Taille | Commande |
|--------|---------|---------|--------|----------|
| neural-chat | 3x (40s) | STAR STAR STAR STAR | 4.1GB | `ollama pull neural-chat` |
| mistral | 1x (116s) | STAR STAR STAR STAR STAR | 4.1GB | `ollama pull mistral` |
| phi | 5x (25s) | STAR STAR STAR | 2.7GB | `ollama pull phi` |
| orca-mini | 4x (30s) | STAR STAR STAR | 1.7GB | `ollama pull orca-mini` |

**Changer le modele dans src/model.py:**

```python
# Dans _generate_ollama():
response = self.client.chat.completions.create(
    model="neural-chat",  # ou "mistral", "phi", etc.
    messages=[...],
    max_tokens=350,
    temperature=0.5,
    top_p=0.9
)
```

### Optimisation de vitesse

**Pour accelerer la generation:**

1. **Utiliser neural-chat ou phi** (plus rapides)
```bash
ollama pull neural-chat
```
Puis changer model="neural-chat" dans src/model.py

2. **Reduire max_tokens** (de 350 a 250)
```python
max_tokens=250  # Plus rapide, moins de contenu
```

3. **Ajuster la temperature** (plus deterministe = plus rapide)
```python
temperature=0.3  # Plus rapide (moins de variation)
```

## Pipeline Technique

```
Donnees brutes CSV
    |
    v
Preprocessing (DataProcessor)
    |
    v
Structuration en Dict
    |
    v
Creation de Prompt structure
    |
    v
Ollama API (localhost:11434)
    |
    v
Descriptions generees
    |
    v
Streamlit / CSV / Retour programmmatique
```

## Architecture Code

### src/model.py - Classe ClaimsLLMGenerator

```python
class ClaimsLLMGenerator:
    """Genere des descriptions de sinistres via Ollama."""
    
    def __init__(self, model_name: str = "ollama"):
        """Initialize avec Ollama (uniquement)."""
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
    
    def create_prompt(self, claim_data: Dict) -> str:
        """Creer un prompt structure pour le LLM."""
        # Formatage des donnees
        # Instructions pour generation
        # Retour du prompt
    
    def generate(self, claim_data: Dict) -> str:
        """Generer une description."""
        prompt = self.create_prompt(claim_data)
        response = self.client.chat.completions.create(...)
        return response.choices[0].message.content
    
    def generate_batch(self, claims: List[Dict]) -> List[str]:
        """Generer descriptions pour plusieurs sinistres."""
```

### src/data_processor.py - Classe DataProcessor

```python
class DataProcessor:
    """Charge et nettoie les donnees de sinistres."""
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Charger le CSV."""
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage et normalisation."""
    
    def analyze_quality(self, df: pd.DataFrame) -> Dict:
        """Analyser la qualite des donnees."""
```

### app_llm_v2.py - Interface Streamlit

- Page Accueil: Vue d'ensemble et statistiques
- Page Tester: Generation interactive avec selection/saisie manuelle
- Page Analyse batch: Traitement multiple avec metriques et export CSV
- Page A propos: Infos techniques et ameliorations possibles

## Ressources

- [Ollama Documentation](https://ollama.ai)
- [Ollama Models](https://ollama.ai/library)
- [OpenAI Python Client](https://github.com/openai/openai-python)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## Depannage

### "Ollama n'est pas accessible!"
Verifiez que:
1. Ollama est installe: https://ollama.ai
2. Ollama est lance: `ollama serve` dans un terminal
3. Le modele est telecharge: `ollama pull neural-chat`
4. Acces sur localhost:11434: `curl http://localhost:11434/api/tags`

### "ModuleNotFoundError: No module named 'openai'"
```bash
pip install openai requests python-dotenv
```

### La generation est trop lente (> 60s)
Utilisez un modele plus rapide:
```bash
ollama pull phi
```
Puis model="phi" dans src/model.py

Ou reduisez max_tokens:
```python
max_tokens=250  # de 350
```

### Les descriptions ne sont pas pertinentes
Ajustez le prompt dans `create_prompt()` ou augmentez temperature:
```python
temperature=0.7  # de 0.5 (plus de variation)
```

## Ameliorations Futures

- [x] Migration OpenAI → Ollama (ZERO COUT)
- [x] Interface Streamlit v2 pedagogique
- [x] Optimisation vitesse (neural-chat)
- [ ] Fine-tuning sur corpus de sinistres
- [ ] Stockage descriptions en base de donnees
- [ ] API REST pour integration
- [ ] Traitement parallelise pour batch massif
- [ ] Classement qualite descriptions (LLM-as-Judge)
- [ ] Support de plusieurs langues
- [ ] Caching embeddings pour optimisation
- [ ] Tests unitaires et CI/CD

## Credits

- Dataset: https://www.kaggle.com/datasets/litvinenko630/insurance-claims
- LLM: Ollama + Neural-chat/Mistral
- Interface: Streamlit
- Python Stack: pandas, numpy, scikit-learn, transformers