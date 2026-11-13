# tresor-auctions

**Peut-on prévoir la qualité d'une adjudication du Trésor américain avant qu'elle ait lieu, avec uniquement des données publiques mises à jour automatiquement ?**

Projet individuel du cours *Machine Learning for Finance* (emlyon, G. Coqueret).

> État actuel : phase 8 sur 11. `--fetch` collecte les 11 tables brutes ; `--clean` nettoie, contrôle la qualité et construit 46 variables point-in-time ; `--backtest` lance la validation walk-forward de 5 modèles, de leur moyenne et d'une variante en écart à `mean3` (moins de 2 heures sur 8 cœurs), compare aux benchmarks `naive`, `mean3` et constante avec des intervalles de confiance, choisit le modèle sur 2014-2019 et le juge sur 2020-2026, puis produit l'interprétation (permutation et SHAP) et les figures de `reports/figures/` ; `--strategy` lance en quelques secondes l'analyse par quintiles et la stratégie illustrative à partir des prévisions du backtest. `--train` entraîne les modèles finaux sur tout l'historique (environ 35 secondes) ; `--predict` prévoit les enchères à venir et les inscrit dans le journal `data/output/predictions_log.parquet`, où une prévision est figée à partir du jour de l'enchère. Le détail du travail prévu est dans [PLAN.md](PLAN.md).

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
