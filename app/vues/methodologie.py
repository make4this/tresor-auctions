"""Page 6 : méthodologie, limites et usage des LLM."""

from __future__ import annotations

import streamlit as st

st.title("Méthodologie")

st.markdown(
    """
### La question
Prévoir, avant chaque adjudication du Trésor américain, la qualité de la demande : le
**bid-to-cover** (montant demandé / montant émis), la **part des dealers** (ce que les banques
primaires doivent absorber), la **part des indirects** (souvent des investisseurs étrangers) et
le **tail** (taux obtenu moins le taux de marché de la veille). Deux familles sont modélisées
séparément : les notes et bonds (2 à 30 ans) et les bills (4 à 52 semaines).

### Les sources (toutes publiques et gratuites)
- **Fiscal Data** (Trésor) : historique des adjudications depuis 2000, dette, trésorerie ;
- **TreasuryDirect** : résultats récents (Fiscal Data publie avec retard) et calendrier ;
- **Treasury.gov** : courbe des taux et taux des bills ; **CBOE** : VIX ;
- **FRED** : indice dollar, bilan de la Fed, reverse repo, compte du Trésor, spread Baa,
  inflation anticipée, avec leur **date de première publication** (ALFRED) ;
- **New York Fed** : positions des primary dealers ; **CFTC** : positions des asset managers et
  des leveraged funds sur les futures de Treasuries.

### Les variables : seulement ce qui était connu avant l'enchère
Chaque valeur de marché porte la date à laquelle elle était **publiquement connue**. Pour une
enchère du jour *t*, on prend la dernière valeur publiée **strictement avant** *t*. Un contrôle
bloquant refuse toute variable qui utiliserait une information du jour même ou d'après.
46 variables : caractéristiques de l'enchère, historique de la même maturité, demande récente,
marché, liquidité et positionnement.

### La validation
- **Walk-forward** : chaque trimestre depuis 2014, entraînement sur tout le passé, test sur le
  trimestre suivant ; jamais de mélange des dates.
- **Benchmark** : la moyenne des 3 dernières enchères de même maturité (`mean3`), plus la
  dernière enchère et une constante.
- **Modèle de référence fixé à l'avance** : la moyenne de 5 modèles (Ridge, Lasso, ElasticNet,
  Random Forest, LightGBM), pour ne pas choisir un modèle d'après ses résultats.
- **Choix et jugement séparés** : un modèle « meilleur » est choisi sur 2014-2019 et jugé sur
  2020-2026 ; intervalles de confiance par bootstrap d'années entières.

### Ce que disent les résultats
- Le **bid-to-cover des notes et bonds** est le seul résultat solide : environ +10 % de R²
  contre `mean3`, positif sur chaque période.
- Les **nouvelles sources** (dealers, CFTC, liquidité) n'apportent rien de mesurable.
- La **part des dealers** des notes et bonds est mal prévue depuis 2020 ; le **tail** reflète
  surtout le mouvement du marché du jour.
- Une **stratégie** qui vend de la duration avant les enchères prévues faibles perd de
  l'argent : la partie prévisible de la demande n'a pas d'effet mesurable sur les taux.

### Les prévisions en direct
Les modèles sont entraînés sur tout l'historique et prévoient les enchères annoncées. Chaque
prévision est inscrite dans un **journal figé** : à partir du jour de l'enchère (heure de New
York), elle n'est plus jamais modifiée, ce qui permet de juger honnêtement les prévisions.

### Limites
- Le calendrier du Trésor n'annonce les enchères que quelques jours à l'avance.
- Les crises du plafond de la dette (2015, 2021, 2023) gonflent une partie des résultats sur
  les bills.
- Le coût de 0,5 pb par trade de la stratégie est une hypothèse.

### Usage des LLM
Ce projet a été construit avec l'aide d'un LLM (Claude, d'Anthropic) utilisé comme assistant de
programmation, phase par phase. Chaque phase a été vérifiée par des relectures indépendantes,
et les erreurs trouvées sont documentées. Le détail (demandes, choix, problèmes et corrections)
est dans le fichier `journal_llm.md` du dépôt.
"""
)
