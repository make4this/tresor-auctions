# tresor-auctions

**Peut-on prévoir la qualité d'une adjudication du Trésor américain avant qu'elle ait lieu, avec uniquement des données publiques mises à jour automatiquement ?**

Projet individuel du cours *Machine Learning for Finance* (emlyon business school, G. Coqueret, 2026).

> **Application en ligne : https://tresor-auctions.streamlit.app** — dépôt : https://github.com/make4this/tresor-auctions

**Usage des LLM.** Le code de ce dépôt a été écrit par Claude Code (Anthropic), sous la direction de l'étudiant, phase par phase : [PLAN.md](PLAN.md) est la consigne donnée au modèle. Après chaque phase, d'autres agents Claude, lancés séparément et en lecture seule, ont relu le travail (trois pour les phases 1 à 3, deux ensuite), et leurs remarques ont été corrigées. Tout est consigné dans [journal_llm.md](journal_llm.md) : demandes, productions, erreurs trouvées et corrigées, écarts au plan.

## Résultats

Le modèle de référence est la moyenne des prévisions de 5 modèles. Elle a été retenue par principe, pour ne pas choisir un modèle d'après ses résultats, lors de la vérification de la phase 6 ; une première version montrait déjà les résultats de chaque modèle sur 2014-2026. Elle est jugée sur 2020-2026, en validation walk-forward. Le R² est mesuré contre `mean3`, la moyenne des 3 dernières enchères de la même maturité. Les intervalles sont à 90 %, par bootstrap d'années entières.

- **Bid-to-cover** : la prévision bat `mean3`, avec un R² de +9,6 % [5,7 ; 14,5] pour les notes et bonds, le résultat le plus solide. Pour les bills, il est de +21,2 % [13,3 ; 26,2], mais ce gain est en partie dû aux crises du plafond de la dette : +17,1 % sans le meilleur trimestre ([backtest_metrics.csv](data/output/backtest_metrics.csv)).
- **Tail** (notes et bonds) : R² de +22,1 % [16,3 ; 27,8] contre `mean3`, mais de +2,8 % seulement contre une constante. Le tail reflète surtout le mouvement du marché du jour.
- **Part des dealers** (notes et bonds) : la prévision fait pire que `mean3`, avec un R² de −28,5 % [−45,9 ; −14,7] (même fichier).
- **Nouvelles sources** (dealers, CFTC, liquidité) : elles n'améliorent la prévision d'aucune cible ; pour 4 cibles sur 8, elles la dégradent un peu ([new_sources_comparison.csv](data/output/new_sources_comparison.csv)).
- **Lien avec le marché** (notes et bonds, 2009-2026) : pour les 20 % d'enchères dont le bid-to-cover est le plus en dessous de `mean3`, le taux de la maturité varie en moyenne de +2,0 pb [+1,2 ; +2,8] entre la clôture de la veille et celle du jour de l'enchère ; pour les 20 % les plus au-dessus, de −1,4 pb [−1,7 ; −1,0]. C'est une association, pas une cause ([quintiles.csv](data/output/quintiles.csv)).
- **Stratégie illustrative** : aucun gain démontré. Sur 2020-2026, elle perd 1,40 pb par trade après coût, contre 1,27 pb pour la règle « toujours vendeur », soit un écart de −0,13 pb [−0,84 ; +0,58] ([strategy_summary.csv](data/output/strategy_summary.csv)).
- Tous les chiffres du rapport et de l'oral, avec le fichier qui produit chacun, sont dans [reports/chiffres_cles.md](reports/chiffres_cles.md). Les figures sont dans [reports/figures/](reports/figures/).

Ces prévisions sont un exercice académique, pas un conseil en investissement.

## État d'avancement

Les 11 phases du plan sont faites : collecte, nettoyage, variables point-in-time, modèles, stratégie, prévisions en direct avec journal figé, application, automatisation et mise en ligne, supports pour l'oral et le rapport. Reste à faire par l'étudiant : ouvrir l'application depuis un autre appareil (critère de la phase 10), relire le code, écrire le rapport et préparer l'oral.

## Organisation du dépôt

| Dossier | Contenu |
|---|---|
| `src/tresor/sources/` | téléchargement des 11 tables publiques (un fichier par site) |
| `src/tresor/clean/` | nettoyage et contrôles de qualité |
| `src/tresor/features.py` | 46 variables point-in-time (seulement des données publiées avant l'enchère) |
| `src/tresor/models.py`, `strategy.py`, `plots.py` | validation walk-forward, interprétation, stratégie, figures |
| `src/tresor/live.py`, `check_log.py` | modèles finaux, prévisions du soir et journal figé |
| `src/tresor/models_store.py`, `refresh.py` | modèles déposés en ligne, bouton « Mettre à jour maintenant » |
| `src/tresor/pipeline.py` | ligne de commande : `python -m tresor.pipeline --help` |
| `app/` | application Streamlit, en anglais : un tableau de bord en page d'accueil et 6 pages détaillées |
| `data/raw/`, `data/clean/`, `data/output/` | données brutes, nettoyées, et résultats, mis à jour chaque soir |
| `reports/` | figures et chiffres clés pour le rapport et l'oral (`python reports/make_figures.py`) |
| `tests/` | tests automatiques (sans réseau) |
| `.github/workflows/` | passage du soir, tests, contrôle du journal figé |

## Installation

Python 3.11, la version du passage du soir. Les tests tournent aussi sous Python 3.14, celle de l'application en ligne. Ils sont faits sous Linux (GitHub Actions) et macOS. Les versions exactes de toutes les bibliothèques sont dans `requirements.txt`.

```bash
python3.11 -m venv .venv
```

```bash
source .venv/bin/activate
```

Sous Windows : `.venv\Scripts\activate`.

```bash
pip install -r requirements-dev.txt
```

```bash
pip install --no-deps -e .
```

Sur macOS, LightGBM a besoin de la bibliothèque `libomp` (`brew install libomp`). Avec Anaconda, l'environnement d'origine du projet s'obtient avec `CONDA_SUBDIR=osx-arm64 conda create -n tresor-auctions --override-channels -c conda-forge python=3.11 pip llvm-openmp`.

Aucune clé n'est nécessaire pour les tests ni pour l'application : les données sont dans le dépôt. Seul le téléchargement (`--fetch`) demande une clé gratuite [FRED](https://fred.stlouisfed.org/docs/api/api_key.html), à mettre dans `.env`, qui n'est jamais commité :

```bash
cp .env.example .env
```

Les modèles entraînés ne sont pas dans le dépôt. `python -m tresor.pipeline --train` les recrée en environ 35 secondes. Sans eux, l'application télécharge ceux du passage du soir.

## Utilisation

```bash
python -m tresor.pipeline --help
```

- `--fetch` collecte les 11 tables brutes.
- `--clean` nettoie, contrôle la qualité et construit les variables.
- `--backtest` lance la validation walk-forward : 5 modèles, leur moyenne et une variante en écart à `mean3`, contre les benchmarks `naive`, `mean3` et constante, avec l'interprétation (permutation et SHAP) et les figures. Compter environ 2 heures sur 8 cœurs.
- `--strategy` produit les quintiles et la stratégie illustrative, en quelques secondes.
- `--train` entraîne les modèles finaux. Rien n'est refait si les données n'ont pas changé.
- `--predict` prévoit les enchères à venir et les inscrit dans le journal `data/output/predictions_log.parquet`. Une prévision y est figée à partir du jour de l'enchère (date de New York).

Pour refaire les figures et `reports/chiffres_cles.md` depuis les fichiers de `data/output/`, en quelques secondes et sans relancer les modèles :

```bash
python reports/make_figures.py
```

Pour vérifier que chaque chiffre de `reports/chiffres_cles.md` correspond toujours aux fichiers :

```bash
python reports/make_figures.py --check
```

Pour lancer l'application sur sa machine :

```bash
streamlit run app/streamlit_app.py
```

## Vérifications

```bash
ruff check .
```

```bash
ruff format --check .
```

```bash
python -m pytest -q
```

## En ligne

- **Passage du soir** (`.github/workflows/update.yml`) : du lundi au vendredi à 23 h 17 UTC, GitHub Actions effectue les étapes suivantes.
  1. Il lance `python -m tresor.pipeline --fetch --clean --train --predict`. Si une source est en panne (code de sortie 3), il réessaie une fois 10 minutes plus tard, puis continue avec un avertissement. Une erreur bloquante arrête tout, et rien n'est enregistré.
  2. Il vérifie le journal figé (`python -m tresor.check_log`).
  3. Il dépose les modèles dans la release `modeles-du-soir`, sous un nom tiré de leur empreinte SHA-256.
  4. Il enregistre les données dans le dépôt.

  Le passage se lance aussi à la main : onglet Actions, « Run workflow ». La clé FRED est le secret `FRED_API_KEY` du dépôt, et le résultat de chaque source est écrit dans `data/output/fetch_status.json`.
- **Tests** (`.github/workflows/tests.yml`) : `ruff` et `pytest` à chaque envoi de code, sous Python 3.11 et 3.14.
- **Contrôle du journal figé** (`.github/workflows/journal.yml`) : un envoi fait à la main qui touche au journal est comparé à la version d'avant. Une ligne figée modifiée, supprimée ou ajoutée après coup le fait passer en rouge.
- **Application** : elle est hébergée sur Streamlit Community Cloud, avec le fichier principal `app/streamlit_app.py` et les versions de `requirements.txt`. Elle tourne sous Python 3.14. Les modèles, entraînés sous 3.11 avec les mêmes versions de numpy, scikit-learn et LightGBM, s'y relisent. Elle n'a besoin d'aucun secret : elle ne contacte jamais FRED. Le backtest ne tourne pas chaque soir : il se lance à la main avec `--backtest --strategy`.

## Données

Toutes les données viennent de sources publiques et gratuites :

- Fiscal Data et TreasuryDirect (Trésor américain) ;
- Treasury.gov ;
- la Federal Reserve Bank of New York ;
- la CFTC ;
- le CBOE ;
- FRED (Federal Reserve Bank of St. Louis).

Elles restent soumises aux conditions d'utilisation de chaque producteur. This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
