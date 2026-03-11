# INF6083 — Projet P2 : Recommandation basée sur le contenu

Devoir 2 du cours *Systèmes de recommandation* (INF6083). Implémentation d’un système de recommandation **basé sur le contenu** (content-based filtering), avec environnement local (venv), support optionnel de CUDA, et compatibilité Linux / Windows.

---

## Prérequis

- **Python** : 3.10 ou supérieur
- **Système** : Linux ou Windows
- **Optionnel** : pilote NVIDIA + CUDA pour accélération GPU (voir section CUDA)

---

## Installation (KISS)

### 1. Cloner / aller dans le projet

```bash
cd sysdereco_devoir2
```

### 2. Créer et activer un venv

**Linux / macOS :**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (cmd) :**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell) :**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

**CPU uniquement :**

```bash
pip install -r requirements.txt
```

**Avec CUDA (exemple : CUDA 12.1) :**

Installer d’abord PyTorch avec CUDA, puis le reste :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Adapter `cu121` à votre version de CUDA (voir [pytorch.org](https://pytorch.org/get-started/locally/)).

---

## Structure du projet

```
sysdereco_devoir2/
├── README.md                 # Ce fichier
├── requirements.txt          # Dépendances Python (venv)
├── config/
│   └── default.yaml          # Paramètres par défaut (chemins, seuils, etc.)
├── data/                     # Données brutes ou pré-traitées (à ajouter)
├── src/
│   ├── __init__.py
│   ├── data/                 # Chargement et prétraitement
│   │   ├── __init__.py
│   │   └── loader.py         # Structure : chargement des données
│   ├── recommender/          # Moteur de recommandation
│   │   ├── __init__.py
│   │   └── content_based.py  # Structure : similarité / recommandation
│   └── evaluation/           # Métriques et évaluation
│       ├── __init__.py
│       └── metrics.py        # Structure : précision, rappel, etc.
├── scripts/
│   ├── run_pipeline.py       # Script principal : de la donnée aux recommandations
│   └── check_env.py          # Vérification venv + CPU/CUDA
├── models/                   # Modèles sauvegardés (optionnel)
└── results/                  # Résultats (métriques, logs)
```

Chaque script et module contient en en-tête des **commentaires détaillant sa structure** et son rôle, en français.

---

## Utilisation rapide

1. **Vérifier l’environnement** (venv, PyTorch, CUDA si installé) :

   ```bash
   python scripts/check_env.py
   ```

2. **Lancer le pipeline** (une fois les données et le code en place) :

   ```bash
   python scripts/run_pipeline.py
   ```

Les chemins et options sont configurables via `config/default.yaml`.

---

## CUDA (optionnel)

- **Sans GPU** : le projet fonctionne en CPU (NumPy, scikit-learn, PyTorch CPU).
- **Avec GPU** : installer PyTorch avec le wheel CUDA adapté à votre version (voir étape 3 ci-dessus). Le code pourra utiliser `.to("cuda")` ou équivalent là où c’est pertinent.

---

## Licence

Voir le fichier `LICENSE` à la racine du projet.
