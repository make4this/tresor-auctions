# tresor-auctions

**Peut-on prévoir la qualité d'une adjudication du Trésor américain avant qu'elle ait lieu, avec uniquement des données publiques mises à jour automatiquement ?**

Projet individuel du cours *Machine Learning for Finance* (emlyon, G. Coqueret).

> État actuel : phase 3 sur 11. `--fetch` collecte les 11 tables brutes (adjudications, taux, VIX, dette, trésorerie, dealers, CFTC, FRED) dans `data/raw/`. Les séries publiées avec retard (dette, trésorerie, dealers, CFTC, FRED) portent leur date de publication (`available_at`) ; les clôtures quotidiennes reçoivent la leur au nettoyage. Le nettoyage et les modèles ne sont pas encore écrits : `--all` s'arrête donc à l'étape de nettoyage. Le détail du travail prévu est dans [PLAN.md](PLAN.md).

## Installation

Python 3.11 ou plus est nécessaire. Sur un Mac Apple Silicon, `CONDA_SUBDIR=osx-arm64` donne un Python natif même si Anaconda est installé en version Intel, et `llvm-openmp` fournit la bibliothèque `libomp` dont LightGBM a besoin.

```bash
CONDA_SUBDIR=osx-arm64 conda create -n tresor-auctions --override-channels -c conda-forge python=3.11 pip llvm-openmp
```

```bash
conda activate tresor-auctions
```

```bash
pip install -e ".[dev]"
```

```bash
cp .env.example .env
```

La clé gratuite [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) se renseigne ensuite dans `.env`, qui n'est jamais commité.

## Utilisation

```bash
python -m tresor.pipeline --help
```

## Vérifications

```bash
ruff check .
```

```bash
pytest
```

## Usage des LLM

Ce projet est réalisé avec l'aide de Claude Code. Chaque étape est consignée dans [journal_llm.md](journal_llm.md) : demande, production, erreurs trouvées et corrigées.
