"""Page 6 : méthodologie, limites et usage des LLM."""

from __future__ import annotations

import streamlit as st

from tresor.app_ui import DISCLAIMER, REPOSITORY_URL
from tresor.live import installed_versions

st.title("Méthodologie")
st.info(DISCLAIMER)

st.markdown(
    """
### La question
Prévoir, avant chaque adjudication du Trésor américain, la qualité de la demande : le
**bid-to-cover** (montant demandé / montant émis), la **part des dealers** (ce que les primary
dealers, banques agréées tenues d'enchérir, doivent absorber), la **part des indirects** (souvent
des investisseurs étrangers) et le **tail** (taux obtenu moins le taux de marché de la veille).
Deux familles sont modélisées séparément : les notes et bonds (2 à 30 ans) et les bills
(4 à 52 semaines).

### Les sources (toutes publiques et gratuites)
- **Fiscal Data** (Trésor) : historique des adjudications depuis 2000, dette, trésorerie ;
- **TreasuryDirect** : résultats récents (Fiscal Data publie avec retard) et calendrier ;
- **Treasury.gov** : courbe des taux et taux des bills ; **CBOE** : VIX ;
- **FRED** : indice dollar, bilan de la Fed, reverse repo, compte du Trésor, spread Baa,
  inflation anticipée ; quand l'archive ALFRED existe, on utilise la date de **première
  publication** de chaque valeur, sinon un délai de publication prudent ;
- **New York Fed** : positions des primary dealers ; **CFTC** : positions des asset managers et
  des leveraged funds sur les futures de Treasuries.

### Les variables : seulement ce qui était connu avant l'enchère
Chaque valeur de marché porte la date à laquelle elle était **publiquement connue**. Pour une
enchère du jour *t*, on prend la dernière valeur publiée **strictement avant** *t*. Un contrôle
bloquant refuse toute variable qui utiliserait une information du jour même ou d'après.
46 variables : caractéristiques de l'enchère, historique de la même maturité, demande récente,
marché, liquidité et positionnement.

### La validation
- **Walk-forward** : chaque trimestre depuis 2014, entraînement sur tout le passé (depuis 2009),
  test sur le trimestre suivant ; jamais de mélange des dates.
- **Benchmark** : la moyenne des 3 dernières enchères de même maturité, plus la dernière
  enchère et une constante.
- **Modèle de référence** : la moyenne de 5 modèles (Ridge, Lasso, ElasticNet, Random Forest,
  LightGBM), retenue pour ne pas choisir un modèle d'après ses résultats.
- **Choix et jugement séparés** : un modèle « meilleur » est choisi sur 2014-2019 et jugé sur
  2020-2026 ; intervalles de confiance par bootstrap d'années entières.

### Ce que disent les résultats
- Le **bid-to-cover des notes et bonds** est le seul résultat solide : environ +10 % de R²
  contre la moyenne des 3 dernières, positif sur chaque période.
- Les **nouvelles sources** (dealers, CFTC, liquidité) n'apportent rien de mesurable.
- La **part des dealers** des notes et bonds est mal prévue depuis 2020 ; le **tail** reflète
  surtout le mouvement du marché du jour.
- Une **stratégie** fondée sur le signal (vendeur de duration si le bid-to-cover prévu est sous
  la moyenne des 3, acheteur sinon) perd 1,0 pb par trade après coûts et ne fait pas mieux que
  « toujours vendeur » (écart non significatif). La partie prévisible de la demande n'a pas
  d'effet mesurable sur le taux du jour.

### Les prévisions en direct
Les modèles sont entraînés sur tout l'historique disponible (depuis 2009) et prévoient les
enchères annoncées. Chaque prévision est inscrite dans un **journal figé** : à partir du jour de
l'enchère (heure de New York), elle n'est plus jamais modifiée, ce qui permet de juger
honnêtement les prévisions.

### Limites
- Le calendrier du Trésor n'annonce les enchères que quelques jours à l'avance, parfois avant
  leur montant (montant alors estimé).
- Le **tail** est approché : le vrai tail se mesure contre le taux « when-issued » de 13 h,
  une donnée payante ; on utilise le taux de clôture de la veille.
- Pour les séries sans archive des premières publications, on utilise les valeurs révisées
  d'aujourd'hui, avec un délai de publication prudent.
- Les crises du plafond de la dette (2015, 2021, 2023) gonflent une partie des résultats sur
  les bills.
- Le coût de 0,5 pb par trade de la stratégie est une hypothèse.

### Petit glossaire
- **pb** : point de base, 0,01 point de pourcentage de taux.
- **R² contre la moyenne des 3** : 1 − (erreur quadratique du modèle / erreur quadratique de la
  moyenne des 3 dernières enchères) ; au-dessus de 0, le modèle fait mieux.
- **Walk-forward** : on n'évalue un modèle que sur des dates postérieures à son entraînement.
- **Duration** : sensibilité d'une obligation au taux ; être « vendeur de duration », c'est
  gagner quand les taux montent.
- **P&L** : gain ou perte ; **Sharpe** : gain moyen rapporté à sa variabilité, sur un an.
- **Permutation** : hausse de l'erreur quand on brouille une variable ; **SHAP** : contribution
  de chaque variable à la prévision du modèle.

### Usage des LLM
Le code a été écrit par un LLM (Claude Code, d'Anthropic) sous la direction de l'étudiant,
phase par phase, à partir d'un plan écrit à l'avance (`PLAN.md`). Chaque phase a ensuite été
relue par deux autres agents LLM lancés séparément ; les erreurs trouvées et corrigées sont
documentées, avec les demandes et les choix, dans le journal du dépôt.
"""
)

st.markdown(
    f"**Code et journal** : [dépôt GitHub]({REPOSITORY_URL}) ·"
    f" [journal_llm.md]({REPOSITORY_URL}/blob/main/journal_llm.md) ·"
    f" [PLAN.md]({REPOSITORY_URL}/blob/main/PLAN.md)"
)
st.caption(
    "Versions utilisées par cette application : "
    + ", ".join(f"{name} {version}" for name, version in installed_versions().items())
    + "."
)
