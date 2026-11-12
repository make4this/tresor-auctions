# Journal d'utilisation des LLM

> Règle 1 du cours : « LLMs are here to help you, not replace you. Mention when/how you use them. »

**Outil utilisé :** Claude Code (modèle Claude Opus 5.5), dans l'application de bureau Claude.

## Phase 1 — Squelette du dépôt (9 novembre 2026)

**Ma demande (résumé) :** j'ai donné la consigne du cours et mon plan détaillé (`PLAN.md`), avec l'instruction « Lis PLAN.md et exécute-le phase par phase ». La phase 1 demande le dépôt, sa structure, `pyproject.toml`, `config.py`, le client HTTP, `.env.example`, un README minimal et ce journal.

**Ce que le LLM a produit :**

- l'arborescence de la section 4 du plan, avec un module vide par fichier à venir ;
- `src/tresor/config.py` : chemins, dates, maturités, paramètres réseau, lecture de la clé FRED ;
- `src/tresor/http.py` : client commun en HTTP/1.1, reprises automatiques, cache disque ;
- `src/tresor/pipeline.py` : la commande unique et ses options ;
- 17 tests dans `tests/`, sans appel réseau.

**Choix faits pendant cette phase :**

- `pip` plutôt que `uv` (absent de la machine), option prévue par le plan.
- Les étapes pas encore écrites échouent avec un message clair au lieu de faire semblant de réussir.
- Les messages d'erreur du client HTTP ne recopient jamais les paramètres de la requête, pour que la clé FRED ne puisse pas apparaître dans les journaux. Ajout du LLM, non demandé explicitement par le plan, en application de sa règle 6 sur les secrets.
- Le fichier `.github/workflows/update.yml` est reporté à la phase 10 : un cron qui lancerait un pipeline encore vide échouerait chaque soir.

**Erreurs trouvées et corrigées :**

| Problème | Comment il a été détecté | Correction |
|---|---|---|
| Python 3.9 sur la machine, le plan demande 3.11 ou plus | Vérification des outils avant de commencer | Environnement conda séparé `tresor-auctions` |
| Ligne de 105 caractères dans `http.py` | `ruff check` (règle E501) | Ligne reformatée |
| LightGBM ne se chargeait pas : `Library not loaded: @rpath/libomp.dylib` | Import réel et mini-entraînement, au-delà des critères de la phase | Paquet `llvm-openmp` ajouté à l'environnement |
| Python Intel exécuté en émulation sur un Mac Apple Silicon | Comparaison de `uname -m` (arm64) et de `platform.machine()` (x86_64) | Environnement recréé en natif avec `CONDA_SUBDIR=osx-arm64` |

**Résultat vérifié :**

- `python -m tresor.pipeline --help` fonctionne ;
- `ruff check .` est propre ;
- `pytest` : 17 tests réussis ;
- appel réel à Fiscal Data via le client commun : statut 200 en HTTP/1.1, 11 154 lignes annoncées par la table ;
- LightGBM, SHAP, Streamlit et les autres dépendances s'importent ;
- `.env` est bien ignoré par git.

**À relire moi-même :** `config.py`, `http.py` et `pipeline.py`, pour pouvoir expliquer chaque ligne à l'oral.

## Phase 2 — Collecte des sources A à D (9 novembre 2026)

**Ma demande (résumé) :** « OK, phase 2 » : un module par source (Fiscal Data, TreasuryDirect, courbe des taux Treasury.gov, VIX), sauvegarde en Parquet dans `data/raw/`, avec nombre de lignes, plage de dates et 3 lignes d'exemple pour chaque source.

**Méthode imposée par mon plan :** un appel de test sur chaque source, lecture des champs réels, puis seulement l'écriture des parseurs.

**Ce que le LLM a produit :**

- `sources/fiscaldata.py`, `treasurydirect.py`, `treasury_rates.py`, `cboe.py` : un module par source ;
- `sources/raw.py` : mise en table, tout reste en texte jusqu'au nettoyage ;
- `sources/collect.py` : collecte, contrôle minimal, écriture Parquet, résumé affiché ;
- l'option `--fetch` du pipeline, branchée sur cette collecte ;
- 17 tests supplémentaires (34 au total), sans appel réseau.

**Écarts entre mon plan et la réalité, trouvés par les appels de test :**

| Ce que disait le plan | Ce qui a été constaté | Décision |
|---|---|---|
| TreasuryDirect : `type=<Bill\|Note\|Bond>` suffit | `Note` et `Bond` ne renvoient ni les TIPS ni les FRN (6 + 6 enchères perdues sur 180 jours) | Interroger aussi `TIPS`, `FRN`, `CMB` et retirer les doublons |
| TreasuryDirect, bills : `highDiscountRate`, `highInvestmentRate` (à vérifier) | Ces champs n'existent pas. Les bons sont `highRate` et `investmentRate` | Correspondance notée dans `treasurydirect.py` |
| Fiscal Data : un appel par `security_type` | Seuls `Bill`, `Note` et `Bond` existent ; un appel sans filtre de type renvoie les 7 905 lignes | Un seul appel, qui ne peut oublier aucun type |
| Taux des bills depuis 2000 | Réponse vide (statut 200) pour 2000 et 2001 | Collecte à partir de 2002 |

Les correspondances entre TreasuryDirect et Fiscal Data ont été vérifiées sur 99 enchères communes (63 bills, 36 coupons) : 18 comparaisons de champs, toutes identiques sur chaque enchère. Elles serviront à la phase 4.

**Erreurs trouvées et corrigées :**

| Problème | Comment il a été détecté | Correction |
|---|---|---|
| Une réponse vide d'une année passée aurait été mise en cache pour toujours | En voyant que le serveur répond « 200 » avec un contenu vide | Une réponse vide n'est jamais mise en cache ; une année passée vide déclenche une erreur |
| Ligne de 101 caractères dans `treasury_rates.py` | `ruff check` (règle E501) | Ligne reformulée |

**Résultat vérifié :**

| Table | Lignes | Période |
|---|---|---|
| `fiscaldata_auctions` | 7 905 | 2000-01-03 → 2026-10-22 |
| `treasurydirect_recent` | 138 | 2026-05-13 → 2026-11-09 |
| `treasurydirect_upcoming` | 10 | 2026-11-10 → 2026-11-19 |
| `treasury_yield_curve` | 6 718 | 2000-01-03 → 2026-11-09 |
| `treasury_bill_rates` | 6 219 | 2002-01-02 → 2026-11-09 |
| `cboe_vix` | 9 311 | 1990-01-02 → 2026-11-06 |

`ruff check .` est propre, `pytest` donne 34 tests réussis. Une deuxième collecte passe de 34 à 12 secondes grâce aux 50 fichiers d'années terminées gardés en cache.

**Points utiles pour l'oral :**

- Dans Fiscal Data, les derniers résultats publiés datent du 15 octobre ; TreasuryDirect va jusqu'au 9 novembre. Sans cette deuxième source, il manquerait trois semaines et demie d'enchères.
- Fiscal Data a 6 lignes sans résultat : 5 enchères récentes (19 au 22 octobre 2026) et un bill à 4 semaines daté du 11 septembre 2001.

**À relire moi-même :** `sources/collect.py` et `sources/treasury_rates.py`.

## Phase 3 — Sources E à H (9 et 12 novembre 2026)

**Ma demande (résumé) :** « OK, phase 3 » : dette et trésorerie (Fiscal Data), positions des primary dealers (New York Fed), Commitments of Traders (CFTC), séries FRED. Pour chaque source, une colonne `available_at` : la date à laquelle la valeur était publiquement connue.

**Méthode :** appel de test, lecture de la documentation officielle (spécification de l'API de la New York Fed, catalogue de la CFTC, messages d'erreur de FRED), mesure des délais de publication réels, puis écriture des parseurs.

**Ce que le LLM a produit :**

- `sources/fiscaldata.py` complété (dette, trésorerie), `sources/nyfed.py`, `sources/cftc.py`, `sources/fred.py` ;
- `add_business_days` dans `sources/raw.py` : jours ouvrés américains, jours fériés fédéraux exclus ;
- 18 tests supplémentaires (52 au total), dont un test par règle de disponibilité.

**Comment `available_at` est déterminé :**

| Source | Règle | Preuve |
|---|---|---|
| Dette (`debt_to_penny`) | +2 jours ouvrés | La valeur du vendredi 6/11 n'était pas en ligne le lundi 9/11 vers 16h45 (New York) |
| Trésorerie (`operating_cash_balance`) | +1 jour ouvré | La valeur du 6/11 était en ligne ce même soir |
| Dealers (New York Fed) | +8 jours (jeudi suivant) | Page officielle (« Data are updated on Thursdays ») ; le 9/11, dernier relevé = mercredi 28/10 |
| CFTC | Vendredi de la semaine du rapport | Règle du plan ; rapports du mardi (1 051), parfois lundi ou mercredi (jours fériés) |
| FRED | Vraie date de première publication (archive ALFRED) ; à défaut, règle par série | Mesuré sur 12 714 valeurs archivées |

Pour FRED, la règle de secours (avant le début de l'archive) a été comparée aux vraies dates : elle est au moins aussi prudente que la réalité dans 99,1 à 100 % des cas pour cinq séries, et dans 93,6 % pour le spread Baa (dont 112 cas sur 204 dus à une panne de publication entre octobre 2016 et mars 2017).

**Écarts entre mon plan et la réalité :**

| Ce que disait le plan | Ce qui a été constaté | Décision |
|---|---|---|
| Quotidien : date + 1 jour ouvré | L'indice dollar `DTWEXBGS` est publié 3 à 10 jours après la date (rapport H.10 hebdomadaire) | Dates réelles ALFRED ; à défaut, mardi de la semaine suivante. La règle du plan aurait créé une fuite |
| Spread high yield `BAMLH0A0HYM2` | FRED n'en garde que 3 ans (début le 13/11/2023) | Remplacé par `BAA10Y` (spread Baa de Moody's, depuis 1986), **à valider** |
| Dealers « par maturité » | Les clés actuelles commencent en avril 2013 ; avant, d'autres clés aux tranches plus larges | Anciennes clés ajoutées : historique depuis juillet 2001. Raccord vérifié (≤ 3 ans : 57 827 M$ le 27/3/2013, 57 210 M$ le 3/4/2013) |
| Trésorerie : `close_today_bal` | Depuis avril 2022, le solde de fin de journée est dans `open_today_bal`, `close_today_bal` vaut « null » ; 16 libellés de compte différents au fil du temps | À traiter au nettoyage (phase 4) |
| Endpoint « positions des dealers » à trouver | `/api/pd/get/{série}.csv` | Trouvé dans la spécification officielle |
| Jeu CFTC à trouver | « TFF - Futures Only » = `gpe5-46if` ; filtre par code de contrat (les noms ont changé en février 2022) | 6 contrats, 5 686 rapports |

**Erreurs trouvées et corrigées :**

| Problème | Comment il a été détecté | Correction |
|---|---|---|
| New York Fed : HTTP 422 sur l'appel groupé des 16 séries | Premier lancement réel (l'appel de test ne portait que sur 3 séries) | Une requête par série, après test de ce que l'API accepte |
| FRED : HTTP 400 au-delà de 2 000 versions par requête | Appel de test sur `RRPONTSYD` | Requêtes découpées en fenêtres de 2 000 versions |
| Le mode « première publication » de FRED omet les valeurs antérieures à l'archive | Comparaison du nombre de lignes (2 025 contre 5 440 pour l'indice dollar) | Série complète + dates de première publication fusionnées |
| Une clé FRED absente aurait arrêté toute la collecte | Relecture du code | Une source sans clé échoue seule ; les autres continuent (testé) |
| Deux erreurs dans mes scripts d'exploration (`DataFrame.asof` est une méthode pandas) | Traceback | Accès par `frame["asof"]` |

**Problème rencontré :** la clé FRED a été collée à l'invite du terminal au lieu du script de saisie. Le LLM l'a ensuite écrite directement dans `.env` (ignoré par git, lisible par mon seul compte).

**Résultat vérifié (collecte du 12/11/2026, 70 secondes) :**

| Table | Lignes | Période |
|---|---|---|
| `fiscaldata_debt` | 6 741 | 2000-01-03 → 2026-11-09 |
| `fiscaldata_cash_balance` | 16 766 | 2005-10-03 → 2026-11-09 |
| `nyfed_dealer_positions` | 9 781 (16 séries) | 1998-01-28 → 2026-10-28 |
| `cftc_positions` | 5 686 (6 contrats) | 2006-06-13 → 2026-11-03 |
| `fred_series` | 27 516 (6 séries) | 2000-01-03 → 2026-11-10 |

Aucune valeur n'a une date de disponibilité antérieure à sa date. `ruff check .` est propre, `pytest` donne 52 tests réussis.

**Limites à assumer :** pendant les fermetures de l'administration américaine, la CFTC a publié en retard, et le « vendredi » est alors trop optimiste. Pour FRED, les valeurs utilisées sont les valeurs actuelles, éventuellement révisées depuis leur publication (seule la date est celle de la première publication).

**À relire moi-même :** `sources/fred.py` et `sources/nyfed.py`.

> ⚠️ Plusieurs règles de cette phase ont été corrigées après la vérification ci-dessous (CFTC, FRED, dealers). Les règles en vigueur sont celles de la section suivante.

## Vérification des phases 1 à 3 par trois agents indépendants (12 novembre 2026)

**Ma demande :** « vérifie bien tout avant (si toutes les phases sont bien effectuées avec plusieurs agents) », avant le premier commit.

**Méthode :** trois agents Claude lancés en parallèle, un par phase, en lecture seule. Chacun a relu le code et le plan, refait ses propres appels aux API et cherché des fuites d'information. Le LLM principal a revérifié chaque constat avant de corriger, puis a relancé la collecte.

**Verdict :** les critères d'acceptation des phases 1 à 3 étaient remplis. La clé FRED ne figure dans aucun fichier suivi par git. En revanche, la vérification a trouvé un problème bloquant et plusieurs problèmes importants.

**Problèmes trouvés et corrigés :**

| Gravité | Problème | Vérification | Correction |
|---|---|---|---|
| Bloquant | CFTC : le « vendredi » est trop tôt les semaines avec un jour férié (publication le lundi) et pendant les pannes : fermetures de l'administration 2018-19 et 2025, incident ION de 2023, deuil national de 2025 | Calendrier officiel 2026 (6 lundis « delayed due to a federal holiday ») et page officielle des annonces spéciales | Jour ouvré suivant si un jour férié tombe dans la semaine ; table des dates officielles pour 30 rapports retardés ; 1 524 lignes sur 5 686 changent de date |
| Important | CFTC : l'historique 2006-2010 n'a été publié que le 3/12/2010 | Annonce officielle du 3/12/2010 | Ces rapports ne sont disponibles qu'à cette date |
| Important | TreasuryDirect ne renvoie que les titres non échus, pas « 180 jours » : un bill à 4 semaines disparaît en un mois, alors que Fiscal Data a environ 4 semaines de retard | Sur 188 enchères depuis le 18/5, TreasuryDirect renvoie les 108 non échues et aucune des 78 échues | La table `treasurydirect_recent` devient cumulative ; alerte prévue en phase 4 pour une enchère récente sans résultat |
| Important | FRED : une valeur d'abord publiée vide (« . ») puis chiffrée plus tard recevait une date trop précoce (reverse repo du 9/4/2020 : chiffre publié le 25/2/2025) | Historique ALFRED de chaque cas | Date de la première publication **chiffrée** (toutes les versions, `output_type=1`) |
| Important | FRED : l'indice dollar `DTWEXBGS` n'existe que depuis le 4/2/2019 ; ses valeurs 2006-2019 n'étaient pas publiques à l'époque | 3 415 valeurs entrées d'un coup dans l'archive ce jour-là | Ces valeurs ne sont disponibles qu'au 4/2/2019 : la variable commence en pratique en 2019 |
| Important | FRED : règle « +1 jour ouvré » trop optimiste pour le spread Baa et l'inflation anticipée | Comparaison avec les vraies dates | +2 jours ouvrés : 99,80 % (Baa, hors panne 2016-17) et 99,94 % (inflation) |
| Important | Une erreur d'un type imprévu (`TypeError`) arrêtait toute la collecte | Reproduit avec un faux serveur | Chaque source est isolée ; erreur imprévue = code de sortie 1 avec le détail |
| Mineur | Une page d'erreur HTML renvoyée avec le statut 200 pouvait être mise en cache pour toujours | Reproduit | Contrôle du contenu (en-tête `Date,`, bonne année) avant mise en cache |
| Mineur | Pagination Fiscal Data triée sur un champ non unique | Tri sur deux champs testé en direct | Tri unique + contrôle des doublons |
| Mineur | Dealers : jeudi férié non géré | — | Jour ouvré suivant |
| Mineur | Journées de deuil national absentes du calendrier | — | Ajoutées (2004, 2007, 2018, 2025) |
| Mineur | La clé pouvait réapparaître si les journaux de `httpx` étaient réactivés ; `Retry-After` ignoré ; `*.part` non ignoré par git | Reproduit | Filtre qui masque la clé, attente demandée par le serveur respectée, `.gitignore` complété |
| Mineur | README : affirmait que toutes les tables avaient une date de disponibilité | Relecture | Formulation corrigée |

**Erreur introduite par le LLM pendant la correction, puis corrigée :** la nouvelle règle de l'indice dollar donnait à 83 jours fériés vides (« . ») postérieurs à 2019 une date de disponibilité antérieure à leur date. Repérée par le contrôle « jamais avant la date », corrigée par un garde-fou général et un test.

**Décisions :**

- `data/cache/` (50 fichiers d'années terminées, 868 Ko) est commité : la mise à jour automatique n'aura pas à retélécharger ces années.
- Valeurs FRED publiées le jour même (4 479 lignes) : `available_at` = la date elle-même. Ce n'est pas une fuite, car la jointure exigera `available_at` strictement antérieur au jour de l'enchère.
- Reportés en phase 10 : figer les versions des paquets, et décider si une source en panne doit bloquer toute la mise à jour du soir.

**Hypothèses non vérifiées :** fermeture d'octobre 2013 côté CFTC (aucune date officielle trouvée ; rapports d'octobre-novembre 2013 neutralisés jusqu'au 31/12/2013 par prudence) ; jour de publication des dealers quand le jeudi est férié ; publication des dealers avant 2013.

**Résultat :** `ruff check .` propre, 78 tests réussis (52 avant la vérification), collecte complète relancée sans erreur.

## Phase 4 — Nettoyage et contrôles (12 novembre 2026)

**Ma demande (résumé) :** « oui commite et phase 4 » : section 6 du plan, c'est-à-dire fusion et nettoyage des adjudications, séries de marché avec leur date de disponibilité, contrôles de qualité.

**Méthode :** le code a été rédigé pendant que les agents vérifiaient les phases 1 à 3, à partir d'une exploration des données brutes (formats de maturité, réouvertures, petites émissions, raccords du solde du Trésor), puis installé après le commit.

**Ce que le LLM a produit :**

- `clean/auctions.py` : fusion Fiscal Data et TreasuryDirect (correspondances vérifiées en phase 2), typage, familles, maturité résiduelle, exclusions, parts des enchérisseurs, taux d'enchère, statut ;
- `clean/market.py` : 55 séries au format long (taux, bills, VIX, FRED, dette, solde du Trésor, dealers raccordés, CFTC nets), et la jointure sans fuite `point_in_time` ;
- `clean/checks.py` : effectifs par année, baisse par rapport à l'exécution précédente, fraîcheur, bornes, doublons, trous ;
- `clean/run.py` et l'option `--clean` ;
- 44 tests supplémentaires (122 au total).

**Points importants pour l'oral :**

- **Réouvertures :** la maturité vient de `security_term`. Le CUSIP `912828SY7`, un 5 ans émis en 2012 et rouvert lors de l'enchère 2 ans du 26/05/2015, est classé 5Y puis 2Y (test unitaire et vérification sur les vraies données).
- **Jointure sans fuite :** pour une enchère du jour *t*, seule compte la dernière valeur publiée strictement avant *t*. Une vieille valeur publiée en retard ne remplace jamais une plus récente (cas de la panne du spread Baa en 2016-2017, testé).
- **Solde du Trésor :** trois présentations successives raccordées au million près (30/9 → 1/10/2021 et 15/4 → 18/4/2022).

**Écarts avec le plan :**

| Ce que disait le plan | Décision | Raison |
|---|---|---|
| Quotidien : `available_at` = date + 1 jour ouvré | Clôtures (taux, bills, VIX) : `available_at` = la date elle-même | Elles sont publiques le soir même. Avec la jointure stricte (< *t*), « +1 jour » ferait utiliser la clôture de *t*-2 au lieu de *t*-1, contrairement à la section 7 (« marché à t-1 ») |
| Bornes de qualité (bid-to-cover dans [1, 6], taux dans [-1, 20]) = alerte | Hors bornes = **à vérifier (orange)**, pas une erreur ; TIPS exclus de la borne des taux | Ce sont de vraies valeurs : bid-to-cover jusqu'à 10,7 sur des bills à 4 semaines en 2009-2015, taux réels des TIPS jusqu'à -1,7 % en 2021 |
| Statut `missing_result` à signaler | Ligne exclue et listée ; alerte rouge si l'enchère a moins de 30 jours | Une enchère sans résultat ne doit pas compter comme « enchère précédente » dans les variables |
| `tail_bp` (cible) | Calculé en phase 5 | Il demande la jointure avec la courbe des taux de la veille, qui relève des variables |

**Résultat vérifié :**

- 7 905 + 137 + 5 lignes en entrée ; 108 doublons Fiscal Data / TreasuryDirect résolus, au profit de la ligne avec résultats puis de TreasuryDirect.
- 13 exclusions : 12 émissions de 25 M$ hors calendrier et le bill à 4 semaines du 11/09/2001, jamais adjugé.
- 7 926 enchères gardées : 7 921 réalisées et 5 à venir. Par famille : 5 102 bills, 1 647 coupons classiques, 765 CMB, 258 TIPS, 154 FRN.
- Depuis 2009, aucune cible manquante pour les coupons classiques et les bills.
- Parts dealers + directs + indirects : 6 291 enchères contrôlées, aucune hors de [0,99 ; 1,01].
- Marché : 178 473 lignes et 55 séries, sans doublon ni date de disponibilité antérieure à la date.
- Plus grand trou : 2 jours ouvrés pour les taux, 4 pour le VIX.
- Qualité globale : orange, avec une seule remarque, les 20 bid-to-cover extrêmes ci-dessus.
- `ruff check .` est propre, 122 tests réussis. Le tableau par famille et par année est dans `data/output/auctions_by_family_year.csv`.

**À relire moi-même :** `clean/auctions.py` (fonctions `combine_sources`, `maturity_bucket`, `exclusion_reasons`) et `clean/market.py` (fonctions `known_values`, `point_in_time`).
