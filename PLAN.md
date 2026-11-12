# PROMPT POUR CLAUDE CODE — Projet « Prévoir les adjudications du Trésor américain » (Python)

> Copie tout ce fichier dans Claude Code (ou enregistre-le sous `PLAN.md` à la racine du projet et dis à Claude Code : « Lis PLAN.md et exécute-le phase par phase »).

---

## 0. Ton rôle et tes règles de travail

Tu es un ingénieur data / ML senior en finance de marché. Tu construis avec moi, **de zéro et en Python**, un projet individuel pour le cours *Machine Learning for Finance* (emlyon, professeur G. Coqueret). Rendu le **18 novembre 2026** : un oral de 3 slides (5 min, centré sur les données), un objet en ligne accessible par URL, et un rapport de 2 à 5 pages.

Règles impératives :

1. **Travaille phase par phase** (section 12). À la fin de chaque phase : lance le code, montre-moi la sortie réelle (nombre de lignes, aperçu, métriques), vérifie les critères d'acceptation, puis attends mon « OK » avant la phase suivante.
2. **N'invente jamais un endpoint, un nom de champ ou un chiffre.** Pour chaque API, fais d'abord un appel de test, affiche les noms de champs réels et 2-3 lignes, puis seulement code le parseur. Si une adresse de ce plan ne marche pas, dis-le et cherche la bonne dans la documentation officielle.
3. **Aucune fuite d'information future (look-ahead).** Toute variable utilisée pour prédire une enchère à la date *t* doit être disponible **avant** *t* (en tenant compte des délais de publication). C'est le point le plus important du projet.
4. **Explique ce que tu fais en français simple** : je dois comprendre et pouvoir défendre chaque ligne à l'oral. Commente le code en français, noms de variables et fonctions en anglais.
5. **Tiens un journal `journal_llm.md`** : pour chaque étape, ce que je t'ai demandé (prompt résumé), ce que tu as produit, les erreurs trouvées et corrigées. La règle 1 du prof : « LLMs are here to help you, not replace you. Mention when/how you use them. »
6. **Secrets** : la clé FRED est lue depuis une variable d'environnement (`.env` en local, jamais commité ; secrets GitHub et Streamlit en ligne). Ajoute `.env` au `.gitignore`.
7. Code propre et simple plutôt que sophistiqué : fonctions courtes, typées, testées. Pas de notebook comme livrable principal.

---

## 1. Le sujet en une phrase

Chaque semaine, le Trésor américain emprunte des dizaines de milliards de dollars par **adjudication** (enchère). Quand la demande est faible, les taux montent. **Question : peut-on prévoir la qualité d'une adjudication avant qu'elle ait lieu, avec uniquement des données publiques mises à jour automatiquement ?**

Mesures de la qualité d'une enchère (les cibles) :

| Cible | Définition | Lecture |
|---|---|---|
| `btc` (bid-to-cover) | montant demandé / montant alloué | bas = demande faible |
| `dealer_share` | part des offres compétitives allouées aux *primary dealers* | haute = demande faible (les banques obligées absorbent) |
| `indirect_share` | part allouée aux enchérisseurs indirects (surtout étrangers) | haute = forte demande étrangère |
| `tail_bp` | taux obtenu à l'enchère − taux de marché de la veille, en points de base | positif = enchère chère pour le Trésor |

Note : le vrai *tail* se mesure contre le taux *when-issued* à 13h New York (donnée payante Bloomberg/LSEG) ; ici on utilise un **proxy** avec la clôture de la veille. C'est une limite assumée.

---

## 2. Livrables finaux

1. **Une app Streamlit en ligne** (Streamlit Community Cloud, gratuit, déployé depuis GitHub) avec une URL publique.
2. **Un dataset propriétaire auto-mis à jour** : un workflow GitHub Actions tourne chaque soir de semaine, retélécharge toutes les sources, nettoie, ré-entraîne, prédit les prochaines enchères et commite les fichiers de données. L'app lit toujours la dernière version, et propose un bouton « Mettre à jour maintenant ».
3. **Un dépôt GitHub propre** : README, code, tests, journal LLM.
4. Des **figures** prêtes pour l'oral et des **chiffres** prêts pour le rapport (dossier `reports/`).

---

## 3. Stack technique

- Python 3.11+, gestion des dépendances avec `uv` (ou `pip` + `requirements.txt` si plus simple).
- Requêtes : `httpx` (avec retries et timeout ; **forcer HTTP/1.1**, certains serveurs du Trésor coupent les flux HTTP/2).
- Données : `pandas` (ou `polars`), stockage en **Parquet** (`pyarrow`).
- ML : `scikit-learn` (Ridge, Lasso, ElasticNet, RandomForest), `lightgbm`, `shap` (interprétation).
- App : `streamlit` + `plotly`.
- Config : `pydantic-settings` ou `python-dotenv` pour la clé FRED.
- Tests : `pytest`. Qualité : `ruff`.
- Automatisation : GitHub Actions (cron).

---

## 4. Structure du dépôt

```
tresor-auctions/
├── PLAN.md                    # ce fichier
├── README.md
├── journal_llm.md             # journal d'usage des LLM (règle 1)
├── pyproject.toml / requirements.txt
├── .env.example               # FRED_API_KEY=
├── .gitignore
├── src/tresor/
│   ├── config.py              # chemins, dates, listes de maturités, paramètres
│   ├── http.py                # client httpx commun (HTTP/1.1, retries, user-agent, cache)
│   ├── sources/
│   │   ├── fiscaldata.py      # adjudications, dette, trésorerie
│   │   ├── treasurydirect.py  # résultats récents + calendrier
│   │   ├── treasury_rates.py  # courbe des taux + taux des bills (CSV Treasury.gov)
│   │   ├── cboe.py            # VIX
│   │   ├── fred.py            # séries FRED (clé requise)
│   │   ├── nyfed.py           # positions des primary dealers
│   │   └── cftc.py            # Commitments of Traders (futures de Treasuries)
│   ├── clean/
│   │   ├── auctions.py        # harmonisation, classification, exclusions
│   │   ├── market.py          # séries quotidiennes / hebdomadaires alignées
│   │   └── checks.py          # contrôles de qualité des données
│   ├── features.py            # variables explicatives point-in-time
│   ├── models.py              # modèles, walk-forward, métriques
│   ├── strategy.py            # stratégie illustrative
│   ├── live.py                # prévisions des enchères à venir + journal figé
│   └── pipeline.py            # orchestration complète (CLI)
├── app/streamlit_app.py
├── data/
│   ├── raw/                   # réponses brutes des API (parquet), jamais modifiées
│   ├── clean/                 # tables nettoyées
│   └── output/                # prédictions, métriques, log des prévisions, qualité
├── reports/figures/           # figures pour l'oral et le rapport
├── tests/
└── .github/workflows/update.yml
```

Commande unique : `python -m tresor.pipeline --all` (options `--fetch`, `--clean`, `--train`, `--predict`, `--backtest`).

---

## 5. Les sources de données (toutes gratuites)

### 5.1 Déjà vérifiées (testées le 3/11/2026)

**A. Fiscal Data — historique des adjudications (sans clé)**
- `GET https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query`
- Paramètres : `page[size]=10000`, `page[number]=1..N` (lire `meta["total-pages"]`), `filter=security_type:eq:Note,auction_date:gte:2000-01-01`, `sort=auction_date`.
- Faire un appel par `security_type` : `Bill`, `Note`, `Bond` (vérifier les autres valeurs existantes, ex. `CMB`, `TIPS`, `FRN`).
- **Les valeurs manquantes sont la chaîne `"null"`** → convertir en NaN. Tout arrive en texte → typer.
- Champs utiles (114 au total) : `cusip`, `auction_date`, `announcemt_date`, `issue_date`, `maturity_date`, `security_type`, `security_term`, `original_security_term`, `reopening`, `offering_amt`, `high_yield`, `avg_med_yield`, `low_yield`, `high_discnt_rate`, `high_investment_rate`, `bid_to_cover_ratio`, `comp_accepted`, `comp_tendered`, `total_accepted`, `total_tendered`, `primary_dealer_accepted`, `direct_bidder_accepted`, `indirect_bidder_accepted`, `soma_accepted`, `inflation_index_security`, `floating_rate`, `cash_management_bill_cmb`.
- ⚠️ **Retard de publication** : le 3/11/2026, aucune enchère postérieure au 21/10 n'avait encore de résultat (les champs restaient `"null"`). D'où la source B.
- ⚠️ Les **TIPS** apparaissent avec `security_type = "Note"` ou `"Bond"` et `inflation_index_security = "Yes"`. Les **FRN** avec `floating_rate = "Yes"`.

**B. TreasuryDirect — résultats récents et calendrier (sans clé, plus frais)**
- Résultats récents : `GET https://www.treasurydirect.gov/TA_WS/securities/auctioned?format=json&type=<Bill|Note|Bond>&days=180`
- Calendrier : `GET https://www.treasurydirect.gov/TA_WS/securities/upcoming?format=json`
- L'endpoint `/search?...` a renvoyé une erreur 500 : l'éviter.
- Format camelCase. **Correspondances vérifiées** avec Fiscal Data :

| TreasuryDirect | Fiscal Data |
|---|---|
| `competitiveAuctionCloseDate` (garder les 10 premiers caractères) | `auction_date` |
| `announcementDate`, `issueDate` | `announcemt_date`, `issue_date` |
| `securityType`, `securityTerm`, `originalSecurityTerm`, `reopening` | idem en snake_case |
| `offeringAmount` | `offering_amt` |
| `highYield`, `lowYield`, `medianYield` | `high_yield`, `low_yield`, `avg_med_yield` |
| `bidToCoverRatio` | `bid_to_cover_ratio` |
| `primaryDealersCompetitiveParAwarded` | `primary_dealer_accepted` |
| `directBiddersCompetitiveParAwarded` | `direct_bidder_accepted` |
| `indirectBiddersCompetitiveParAwarded` | `indirect_bidder_accepted` |
| somme des trois lignes précédentes | `comp_accepted` (vérifié exact) |
| `inflationProtected`, `floatingRate` | `inflation_index_security`, `floating_rate` |

- Dans le calendrier, `offeringAmount` est **vide tant que le montant n'est pas annoncé** (annonce quelques jours avant). Vérifier les champs des bills (taux d'escompte : `highDiscountRate`, `highInvestmentRate`, etc.) par un appel de test.

**C. Treasury.gov — courbe des taux quotidienne (sans clé)**
- `GET https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{YEAR}/all?type=daily_treasury_yield_curve&field_tdr_date_value={YEAR}&page&_format=csv`
- Une requête par année depuis 2000. Colonnes : `Date` (format `MM/DD/YYYY`), `"1 Mo"`, `"1.5 Month"`, `"2 Mo"`, `"3 Mo"`, `"4 Mo"`, `"6 Mo"`, `"1 Yr"`, `"2 Yr"`, `"3 Yr"`, `"5 Yr"`, `"7 Yr"`, `"10 Yr"`, `"20 Yr"`, `"30 Yr"`. **Les colonnes varient selon l'année** (pas de 30 ans entre 2002 et 2006, etc.).
- Mettre en cache les années passées (elles ne changent plus) ; ne retélécharger que l'année en cours.
- Taux des bills : même URL avec `type=daily_treasury_bill_rates` (à vérifier par un appel de test).
- Ce sont les mêmes taux que les séries `DGS*` de FRED.

**D. CBOE — VIX (sans clé)**
- `GET https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv` → colonnes `DATE` (`MM/DD/YYYY`), `OPEN`, `HIGH`, `LOW`, `CLOSE`.

### 5.2 À ajouter (à vérifier par un appel de test avant de coder)

**E. FRED (clé gratuite obligatoire)** — https://fred.stlouisfed.org/docs/api/api_key.html
- `GET https://api.stlouisfed.org/fred/series/observations?series_id=<ID>&api_key=<KEY>&file_type=json&observation_start=2000-01-01`
- ⚠️ L'export sans clé `fredgraph.csv` **bloque les scripts** (erreur HTTP/2 puis time-out) : ne pas l'utiliser.
- Séries : `DTWEXBGS` (indice dollar), `WALCL` (bilan de la Fed, hebdo), `RRPONTSYD` (reverse repo), `WTREGEN` (compte du Trésor à la Fed, hebdo), `BAMLH0A0HYM2` (spread high yield), `T10YIE` (inflation anticipée 10 ans).
- Valeurs manquantes écrites `"."`.

**F. Fiscal Data — autres tables (sans clé)**
- Dette totale quotidienne : `.../v2/accounting/od/debt_to_penny`
- Trésorerie quotidienne (Daily Treasury Statement) : `.../v1/accounting/dts/operating_cash_balance`
- Même pagination et même convention `"null"`. Vérifier les noms exacts des endpoints dans https://fiscaldata.treasury.gov/api-documentation/.

**G. New York Fed — Markets Data API (sans clé)**
- Documentation : https://markets.newyorkfed.org/static/docs/markets-api.html
- Vérifié : `GET https://markets.newyorkfed.org/api/rates/secured/sofr/last/1.json` (SOFR).
- À trouver dans la doc : les **Primary Dealer Statistics** (positions nettes hebdomadaires des dealers en Treasuries, par maturité). ⚠️ Publication avec environ une semaine de décalage : utiliser la **date de publication**, pas la date de référence.

**H. CFTC — Commitments of Traders (sans clé)**
- API Socrata : https://publicreporting.cftc.gov
- Jeu de données « Traders in Financial Futures » (futures seuls). Trouver son identifiant dans le catalogue. Filtrer les contrats Treasury du CBOT (2 ans, 5 ans, 10 ans, Ultra 10 ans, T-Bond, Ultra Bond).
- Positions des *asset managers* et des *leveraged funds*. Données au mardi, publiées le vendredi : **utiliser le vendredi** comme date de disponibilité.

---

## 6. Nettoyage (cœur du projet et de l'oral)

Chaque étape écrit un compte rendu (lignes avant/après, lignes exclues et pourquoi) dans `data/output/cleaning_report.json`.

### 6.1 Adjudications
1. Concaténer Fiscal Data (historique) et TreasuryDirect (récent). Clé unique : `(cusip, auction_date)`. En cas de doublon, garder la ligne **qui a des résultats** ; à égalité, TreasuryDirect.
2. Typer : dates, nombres, booléens (`"Yes"/"No"`). Convertir `"null"`, `""` et `"."` en NaN.
3. Classer chaque titre dans `family` ∈ {`bill`, `cmb`, `nominal_coupon`, `tips`, `frn`} à partir de `security_type`, `inflation_index_security`, `floating_rate` et `cash_management_bill_cmb`.
4. **Maturité résiduelle** : la calculer depuis `security_term` (ex. « 9-Year 10-Month » → 9,83 ans), puis l'arrondir à la maturité standard la plus proche. Notes et bonds : {2, 3, 5, 7, 10, 20, 30} ans. Bills : {4, 6, 8, 13, 17, 26, 52} semaines.
   ⚠️ **Piège vérifié** : une réouverture d'un ancien 5 ans vendue lors de l'enchère 2 ans a `original_security_term = "5-Year"` mais `security_term = "2-Year"`. Utiliser `original_security_term` crée des *tails* aberrants (jusqu'à −90 pb). **Utiliser `security_term`.**
5. Exclure les émissions de moins de 1 Md$ (petites émissions hors calendrier, ex. 25 M$ en 2019 et 2021), les enchères annoncées mais jamais réalisées, et les dates futures sans calendrier.
6. Calculer les parts : `dealer_share = primary_dealer_accepted / comp_accepted` (idem direct et indirect). Vérifier que les trois parts somment à environ 1, et logguer les écarts.
7. **Bills** : le résultat est un taux d'escompte (`high_discnt_rate`) ; utiliser le taux équivalent obligataire (`high_investment_rate`) pour comparer au taux de marché. Vérifier les champs disponibles.
8. Statut de chaque ligne : `completed` (résultats publiés), `upcoming` (date ≥ aujourd'hui à New York), `missing_result` (date passée sans résultat, à signaler).

### 6.2 Données de marché
1. Taux du Trésor (format long : date, maturité, taux), VIX, FRED, dette, trésorerie, dealers NY Fed, CFTC.
2. Garder chaque série à sa fréquence d'origine, avec une colonne `available_at` = date à laquelle la valeur était **publiquement connue** (quotidien : date + 1 jour ouvré par prudence ; CFTC : vendredi ; dealers : date de publication ; FRED hebdomadaire : date de publication).
3. **Jointure point-in-time** : pour une enchère à la date *t*, prendre la dernière valeur avec `available_at < t` (`pandas.merge_asof` avec `direction="backward"` et `allow_exact_matches=False`).

### 6.3 Contrôles de qualité (`clean/checks.py`, à chaque exécution)
- Nombre de lignes par source et par an, avec une alerte en cas de baisse anormale par rapport à l'exécution précédente.
- Dernière date disponible par source (fraîcheur), avec une alerte si elle a plus de 5 jours ouvrés de retard.
- Valeurs hors bornes : bid-to-cover dans [1, 6], parts dans [0, 1], taux dans [−1, 20] %.
- Doublons de clé, trous de dates.
- Résultat écrit dans `data/output/data_quality.json`, et affiché dans l'app.

---

## 7. Variables explicatives (toutes connues avant l'enchère)

Pour chaque enchère *i* (maturité *m*, date *t*) :

- **Enchère** : maturité, famille, réouverture (0/1), log du montant, variation du montant par rapport à la précédente de même maturité, jours depuis la précédente, jour de la semaine, mois de refinancement trimestriel (février, mai, août, novembre), fin de trimestre.
- **Historique de la même maturité** : cibles des 1, 2 et 3 enchères précédentes, et leur moyenne (`btc_mean3`, etc.).
- **Demande récente toutes maturités** : moyenne du bid-to-cover relatif (divisé par sa moyenne mobile sur 12 enchères de même maturité) sur les 5 dernières enchères **strictement antérieures** à *t*.
- **Marché à t−1** : taux de la maturité, sa variation sur 5 et 20 jours, sa volatilité sur 20 jours, pente 10 ans − 2 ans, taux 3 mois, VIX et sa variation sur 5 jours, indice dollar sur 20 jours, spread high yield.
- **Offre et liquidité** : variation de la dette totale sur 30 jours, solde du compte du Trésor, reverse repo, bilan de la Fed.
- **Positionnement** : positions nettes des dealers par maturité (NY Fed), positions des asset managers et des leveraged funds (CFTC).
- **Pour les enchères à venir dont le montant n'est pas annoncé** : supposer le montant de la dernière enchère de même maturité et mettre `offering_estimated = True` (l'afficher dans l'app).

Test unitaire obligatoire : pour un échantillon d'enchères, vérifier qu'aucune variable n'utilise une date ≥ *t*.

---

## 8. Modélisation

- **Échantillon** : enchères `completed` depuis 2009 (le 7 ans reprend en 2009). Deux modèles séparés : `nominal_coupon` (notes et bonds) et `bill`.
- **Benchmarks** : `naive` (dernière enchère de même maturité) et `mean3` (moyenne des 3 dernières). **Le benchmark de référence est `mean3`.**
- **Modèles** : régression Ridge, Lasso, ElasticNet (avec standardisation), RandomForest, LightGBM.
- **Prétraitement appris sur la fenêtre d'entraînement uniquement** (dans un `Pipeline` sklearn) : écrêtage aux quantiles 1 % et 99 % (indispensable : sans lui, les modèles linéaires explosent hors échantillon), imputation par la médiane, standardisation pour les modèles linéaires. Fixer les graines (`random_state=42`).
- **Validation walk-forward** : à chaque trimestre depuis 2014-01-01, entraîner sur tout le passé (fenêtre croissante) et tester sur le trimestre suivant. Jamais de mélange aléatoire des dates. Les hyperparamètres sont choisis à l'intérieur de la fenêtre d'entraînement (`TimeSeriesSplit`).
- **Métriques par cible et par modèle** : RMSE, MAE, corrélation, **R² hors échantillon par rapport à `mean3`** (`1 − SSE_modèle / SSE_mean3`), et la **bonne direction** (le modèle dit-il correctement si l'enchère sera au-dessus ou en dessous de `mean3` ?).
- **Expérience clé pour l'oral** : comparer les performances **avec et sans les nouvelles sources** (dealers, CFTC, liquidité). Les données en plus aident-elles ?
- **Interprétation** : importance par permutation et valeurs SHAP pour le meilleur modèle.
- **Ordres de grandeur attendus** (obtenus par une première version en R ; à comparer, pas à imposer) : bid-to-cover des notes et bonds, R² hors échantillon d'environ +8 % contre `mean3` et bonne direction à 62 % ; part des dealers, R² légèrement négatif ; benchmark `naive` très mauvais (R² ≈ −60 %). Si tes résultats sont très différents, cherche une erreur (fuite d'information ou bug) avant de t'en réjouir.

---

## 9. Lien avec le marché et stratégie illustrative

1. **Analyse descriptive** : regrouper les enchères en quintiles de « surprise » (part des dealers moins `dealer_mean3`) et calculer la variation moyenne du taux de clôture entre t−1 et *t*. Attendu : les enchères les plus faibles font monter le taux d'environ +1,8 pb le jour même.
2. **Stratégie** : si le modèle prévoit `btc < btc_mean3`, se positionner vendeur de duration de la clôture t−1 à la clôture *t*, sinon acheteur. P&L = position × variation du taux (en pb) − 0,5 pb de coûts. Comparer à « toujours vendeur ». Rapporter le nombre de trades, la moyenne, le taux de réussite et le Sharpe annualisé.
3. Un résultat négatif est attendu et doit être présenté honnêtement : la partie prévisible de la demande est déjà dans les prix.

---

## 10. Prévisions en direct (`live.py`)

- Chaque exécution entraîne les modèles sur tout l'historique et prédit toutes les enchères `upcoming`.
- **Journal des prévisions** `data/output/predictions_log.parquet` : clé `(cusip, auction_date, target, model)`, avec `predicted_at` (UTC), `pred`, `bench_mean3` et `offering_estimated`.
  - Tant que la date d'aujourd'hui à New York est **antérieure** à la date de l'enchère, la prévision peut être rafraîchie (ex. quand le montant est annoncé).
  - À partir du jour de l'enchère, la ligne est **figée et ne doit plus jamais être réécrite**. Test unitaire obligatoire.
- `live_tracking` : jointure du journal avec les résultats réels dès leur publication, erreur et statut (en attente ou réalisée).

---

## 11. App Streamlit (`app/streamlit_app.py`)

Lecture des fichiers de `data/output/` avec `st.cache_data(ttl=3600)`. Pages :

1. **Prochaines enchères** : date, titre (ex. « 10 ans, réouverture »), montant (avec « estimé » si besoin), bid-to-cover prévu, moyenne des 3 dernières, écart, **signal** (demande plus faible si l'écart est ≤ −0,08, plus forte s'il est ≥ +0,08, dans la norme sinon), part des dealers prévue et tail prévu.
2. **Suivi en direct** : prévu contre réalisé (nuage de points et tableau), erreur cumulée contre `mean3`.
3. **Backtest** : tableau des métriques par cible ; courbe du gain cumulé (somme cumulée de `(réel − mean3)² − (réel − prévu)²`) ; importance des variables et SHAP ; stratégie.
4. **Explorer** : séries historiques par maturité et par cible, avec les prévisions du backtest superposées.
5. **Qualité des données** : chaque source avec sa dernière mise à jour, son nombre de lignes et ses contrôles (vert ou rouge), et le rapport de nettoyage.
6. **Méthodologie** : sources, cibles, variables, validation, limites, et la mention de l'usage des LLM.

**Bouton « Mettre à jour maintenant »** : il appelle seulement les sources légères (TreasuryDirect récent et calendrier, courbe des taux de l'année en cours), relance la préparation et la prédiction **sans ré-entraîner** (les modèles entraînés la nuit sont sauvegardés avec `joblib`), puis vide le cache. Le gros du travail reste au workflow de nuit.

Design sobre : une couleur principale (#1F4E79), une couleur d'alerte (#C8553D), des titres en français.

---

## 12. Phases d'exécution et critères d'acceptation

**Phase 1 — Squelette (30 min)** : dépôt, structure, `pyproject`, `config.py`, client HTTP, `.env.example`, `README` minimal, `journal_llm.md`.
✅ `python -m tresor.pipeline --help` fonctionne ; `ruff` est propre.

**Phase 2 — Collecte des sources vérifiées A à D** : un module par source, sauvegarde en Parquet dans `data/raw/`.
✅ Pour chaque source : nombre de lignes, plage de dates et 3 lignes d'exemple affichés. Fiscal Data contient des milliers d'enchères depuis 2000, toutes familles confondues.

**Phase 3 — Sources E à H** : d'abord un appel de test et l'affichage des champs réels, puis le parseur. Pour chaque source, sa colonne `available_at`.
✅ Même critère que la phase 2. Toute source qui ne marche pas est signalée, avec une alternative proposée.

**Phase 4 — Nettoyage et contrôles** (section 6).
✅ `cleaning_report.json` et `data_quality.json` générés. Tableau du nombre d'enchères par famille et par an. Aucune réouverture mal classée : test sur un CUSIP connu.

**Phase 5 — Variables point-in-time** (section 7).
✅ Le test anti-look-ahead passe. Taux de valeurs manquantes par variable affiché.

**Phase 6 — Modèles et walk-forward** (section 8).
✅ Tableau des métriques par famille, cible et modèle. Comparaison avec et sans nouvelles sources. Figures dans `reports/figures/`.

**Phase 7 — Analyse de marché et stratégie** (section 9).
✅ Figure des quintiles et tableau de la stratégie.

**Phase 8 — Prévisions en direct** (section 10).
✅ Le journal est créé ; le test « ligne figée » passe.

**Phase 9 — App Streamlit** (section 11).
✅ `streamlit run app/streamlit_app.py` fonctionne en local ; les 6 pages sont vérifiées une par une.

**Phase 10 — Automatisation et mise en ligne** :
- `.github/workflows/update.yml` : cron `17 23 * * 1-5` (et un déclenchement manuel `workflow_dispatch`), setup de Python, installation, `python -m tresor.pipeline --all`, commit et push de `data/` si quelque chose a changé (`permissions: contents: write`). Le secret `FRED_API_KEY` est passé en variable d'environnement.
- Déploiement sur Streamlit Community Cloud (fichier principal `app/streamlit_app.py`, clé dans les secrets Streamlit). L'app se redéploie à chaque push.

✅ Le workflow lancé à la main réussit ; l'URL publique s'ouvre depuis un autre appareil.

**Phase 11 — Supports** : script `reports/make_figures.py` ; fichier `reports/chiffres_cles.md` avec tous les chiffres du rapport et de l'oral, chacun lié au fichier qui le produit ; `journal_llm.md` complet (prompts, problèmes rencontrés et solutions).
✅ Chaque chiffre de `chiffres_cles.md` est recalculé depuis les fichiers de sortie.

---

## 13. Pièges déjà identifiés (à gérer dès le départ)

| Piège | Solution |
|---|---|
| Fiscal Data publie avec des jours de retard | Compléter par TreasuryDirect (180 derniers jours) |
| L'export `fredgraph.csv` de FRED bloque les scripts | API FRED avec clé ; taux pris sur Treasury.gov |
| Erreurs HTTP/2 sur certains serveurs | Forcer HTTP/1.1, retries avec backoff |
| `"null"` en texte dans Fiscal Data, `"."` dans FRED | Conversion systématique en NaN |
| TIPS et FRN classés en `Note` | Utiliser `inflation_index_security` et `floating_rate` |
| Réouvertures classées sur leur maturité d'origine | Utiliser la maturité résiduelle (`security_term`) |
| Petites émissions hors calendrier | Exclure les émissions de moins de 1 Md$ |
| Montant des enchères à venir pas encore annoncé | Montant de la dernière enchère de même maturité, marqué comme estimé |
| Modèles linéaires instables à cause des valeurs extrêmes | Écrêtage 1 % / 99 % appris sur la fenêtre d'entraînement |
| Données hebdomadaires publiées avec retard (CFTC, dealers) | Jointure sur la date de publication, pas sur la date de référence |
| Colonnes de la courbe des taux qui changent selon l'année | Parser par nom de colonne et accepter les colonnes absentes |
| Résultats non reproductibles | `random_state=42` partout |

---

## 14. Ce que tu me rends à la fin

1. Le lien du dépôt et l'URL de l'app.
2. Un résumé en 10 lignes des résultats, avec les chiffres clés.
3. La liste de ce que je dois vérifier ou compléter moi-même : relecture du code, la partie « ma contribution » du rapport, la répétition de l'oral.

Commence par la **phase 1**, puis montre-moi le résultat.
