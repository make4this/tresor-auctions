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
- Qualité globale : orange, avec une seule remarque, les 20 bid-to-cover extrêmes ci-dessus (vert après la vérification ci-dessous, qui ne signale plus que les valeurs extrêmes récentes).
- `ruff check .` est propre, 122 tests réussis. Le tableau par famille et par année est dans `data/output/auctions_by_family_year.csv`.

**À relire moi-même :** `clean/auctions.py` (fonctions `combine_sources`, `maturity_bucket`, `exclusion_reasons`) et `clean/market.py` (fonctions `known_values`, `point_in_time`).

## Vérification de la phase 4 par deux agents indépendants (12 novembre 2026)

**Ma demande :** « Vérifie la phase 4 ».

**Méthode :** deux agents Claude en parallèle, en lecture seule. Le premier a vérifié les adjudications, en recalculant le nettoyage de son côté et en le comparant au calendrier réel du Trésor. Le second a vérifié les séries de marché et les contrôles. Le LLM principal a revérifié chaque constat sur les données avant de corriger.

**Verdict :** rien de bloquant, et **aucune fuite d'information**. La jointure a été comparée à une boucle naïve sur les 6 085 enchères réalisées depuis 2009 et les 55 séries (292 906 valeurs) : 0 écart, et aucune valeur utilisée n'est publiée le jour de l'enchère ou après. Le nettoyage recalculé par l'agent donne exactement la même table.

**Problèmes trouvés et corrigés :**

| Gravité | Problème | Vérification | Correction |
|---|---|---|---|
| Important | Les tests ne vérifiaient pas vraiment la règle « la ligne avec résultats gagne » : l'agent a cassé la règle exprès et les tests passaient toujours | Tests de mutation | 7 tests ajoutés (cette règle, type inconnu, « . », enchère récente sans résultat, CMB très court, reprise de série, durée manquante) |
| Important | CMB : des titres de 1 à 35 jours étiquetés « 4W » ; ils auraient pu servir d'« enchère précédente » à un vrai bill à 4 semaines | 150 CMB de moins de 15 jours étiquetés 4W | Pas de maturité standard au-delà de 20 % d'écart (233 CMB) ; identifiant de série famille + maturité (`series_key`) |
| Important | Reverse repo : avant 2014, FRED écrit « . » les jours sans opération (donc 0) ; la jointure réutilisait une vieille valeur, jusqu'à 356 jours | En 2009, 258 « . » sur 261 jours et aucun 0 ; depuis 2014, environ 12 « . » par an (jours fériés) | « . » = 0 les jours ouvrés (2 593 valeurs), manquant les jours fériés ; ancienneté maximale possible dans la jointure (`max_age`) |
| Important | Une date d'enchère avec une heure ou un fuseau horaire aurait fait passer la clôture du jour même | Reproduit | Ces dates sont refusées avec une erreur claire |
| Important | Une série morte restait « verte » : fraîcheur mesurée par table, pas par série | Simulations (WALCL coupé, colonne 10 ans vide) | Fraîcheur et trous contrôlés pour chacune des 55 séries, selon son rythme de publication |
| Important | La règle anti-fuite n'était qu'un chiffre du compte rendu, jamais une alerte | Reproduit | Alerte rouge si une valeur est publiée avant sa date ou sans date |
| Mineur | Octobre 2008 : trois réouvertures d'anciens 10 ans classées 7Y auraient servi d'historique au 7 ans relancé en février 2009 | 912828DV9, 912828EE6, 912828DM9 | Nouvelle série après plus de 120 jours sans enchère (`series_run`) : 7Y#1 en 2008, 7Y#2 à partir de février 2009 |
| Mineur | Une enchère absente des deux sources était invisible | — | Alerte si une série régulière n'a pas d'enchère depuis deux fois son intervalle habituel |
| Mineur | Le contrôle « les parts somment à 1 » était circulaire pour TreasuryDirect | Le champ `subtotalCompetitiveParAwardedInsideOfOfferingAmount` égale la somme des trois catégories (137/137) et Fiscal Data (103/103) | Ce champ est lu tel quel |
| Mineur | La baisse d'effectifs n'était signalée qu'une fois, jamais pour l'année en cours ni pour la table TreasuryDirect désormais cumulative | Reproduit | Référence = plus grand effectif déjà vu ; toutes les années ; TreasuryDirect inclus |
| Mineur | Statut orange permanent à cause de 20 valeurs extrêmes connues (2009-2015) | — | Seules les valeurs extrêmes des 365 derniers jours deviennent un avertissement |
| Mineur | Compte rendu des séries de marché sans les lignes écartées ; unités mélangées | — | Valeurs manquantes écartées (38 219), lignes d'autres comptes de trésorerie (11 465), colonne `unit` |

**Erreurs du LLM dans ses propres tests, corrigées :** un jeu de test contenait sans le vouloir un second trou, plus grand que celui testé ; un autre ne renseignait pas le nouveau champ TreasuryDirect.

**Non modifié, documenté :**

- La déduplication garde une ligne entière et ne complète pas ses trous avec l'autre source (vérifié sans effet : 3 drapeaux vides seulement).
- `dealer_total` est la somme des 5 tranches (bills et coupons nominaux), hors TIPS et FRN : il diffère du total officiel.
- Possible saut de niveau des dealers au changement de clés du 3/4/2013, à traiter en phase 5 : variations plutôt que niveaux, ou indicatrice.
- Les données CFTC, New York Fed et Treasury sont les versions actuelles, peut-être révisées depuis leur publication (comme FRED).
- Les CMB à 42 et 119 jours, prédécesseurs des bills à 6 et 17 semaines, ne sont pas raccordés à ces séries.

**À reprendre en phase 5 :** variables calculées à l'intérieur d'une même `series_run` ; `max_age` par série dans la jointure ; parts des enchérisseurs absentes avant avril 2008 (échantillon dès 2009) ; pas de taux de marché la veille pour les premiers bills à 8, 17 et 6 semaines ; pas de *tail* pour les FRN et les TIPS (taux réel).

**Résultat :** statut de qualité global **vert**, sans alerte. Reprises de séries détectées depuis 2009, conformes au calendrier réel : 7 ans (26/02/2009), bill à 8 semaines (16/10/2018), 20 ans (20/05/2020), bill à 17 semaines (19/10/2022), bill à 6 semaines (18/02/2025). `ruff check .` est propre, 142 tests réussis (122 avant la vérification).

## Phase 5 — Variables point-in-time (12 novembre 2026)

**Ma demande (résumé) :** « OK, phase 5 » : section 7 du plan, c'est-à-dire les variables connues avant chaque enchère, la cible *tail*, et le test anti-fuite obligatoire.

**Ce que le LLM a produit :**

- `features.py` : 47 variables réparties en 5 groupes ; la cible `tail_bp` ; la provenance de chaque valeur (date de publication de l'information utilisée) ; le contrôle `look_ahead_violations`, qui refuse d'écrire la table en cas de fuite ;
- l'étape `--clean`, qui construit désormais aussi `data/clean/features.parquet` et `data/output/features_missing.csv` ;
- 16 tests, dont le test anti-fuite sur données synthétiques et sur vraies données.

**Les 5 groupes de variables :**

| Groupe | Variables |
|---|---|
| Enchère (`auction`) | maturité, réouverture, log du montant, variation du montant, jours depuis la précédente, jour de la semaine, mois de refinancement, distance à la fin de trimestre, montant estimé |
| Historique (`history`) | les 4 cibles des 3 enchères précédentes de la même série et leur moyenne ; bid-to-cover relatif des 5 dernières enchères de la famille |
| Marché (`market`) | taux de la maturité à t-1, ses variations sur 5 et 20 jours, sa volatilité sur 20 jours ; pente 10 ans - 2 ans, taux 3 mois, VIX et sa variation, indice dollar sur 20 jours, spread Baa, inflation anticipée |
| Liquidité (`liquidity`) | variation de la dette sur 30 jours, solde du Trésor, reverse repo, bilan de la Fed |
| Positionnement (`positioning`) | positions nettes des dealers (tranche de la maturité, total, variation sur 4 semaines), positions nettes des asset managers et des leveraged funds sur le future le plus proche, rapportées à l'open interest |

Les groupes `liquidity` et `positioning` sont les « nouvelles sources » de l'expérience clé du plan (avec contre sans).

**Règles anti-fuite, à savoir défendre :**

- Une information de marché ou d'historique n'est utilisée que si elle était publiée **strictement avant** le jour de l'enchère. Les caractéristiques de l'enchère elle-même sont connues à son annonce, au plus tard le jour même (7 enchères annoncées le jour même).
- Une variable dérivée (variation, volatilité, rapport) n'est connue que quand **toute sa fenêtre** l'était.
- Les historiques se calculent à l'intérieur d'une même série (`series_run`) : une enchère du même jour n'est jamais « la précédente ».
- Une valeur de marché trop vieille est ignorée : plus de 10 jours pour une série quotidienne, 21 pour une série hebdomadaire.
- Montant pas encore annoncé : celui de la dernière enchère de la même série, marqué `offering_estimated`.

**Le test anti-fuite, sous deux formes :**

1. **Provenance :** chaque valeur de chaque variable porte la date de publication de l'information utilisée. Sur les 6 749 enchères réelles, aucune violation.
2. **Troncature :** pour un échantillon d'enchères (4 synthétiques, 8 réelles tirées au hasard depuis 2009), les variables sont recalculées après avoir effacé tout ce qui a été publié à partir du jour de l'enchère. Elles sont identiques.

**Vérification que le test sait échouer :** deux fuites ont été introduites exprès, une jointure de marché qui accepte la clôture du jour même et un historique qui compte une enchère du même jour. Elles sont détectées respectivement par 6 et 7 tests, dont les deux tests sur vraies données.

**Résultat vérifié :**

- Échantillon des modèles (réalisées depuis 2009) : 5 119 enchères (3 757 bills, 1 362 coupons) ; aucune valeur infinie.
- Valeurs manquantes : presque nulles, sauf l'indice dollar (46 à 53 %, série créée en février 2019), les positions CFTC (12 à 13 %, historique publié seulement en décembre 2010 et panne de 2013 neutralisée) et le spread Baa (2 %, panne 2016-2017).
- *Tail* médian : +0,5 pb (coupons) et +0,9 pb (bills). Les plus grands *tails* sont de vrais événements : bills à 4 semaines pendant les crises du plafond de la dette (+143 pb le 4/5/2023, +37 pb le 5/9/2017) et crise de la Silicon Valley Bank (mars 2023). Seules 19 enchères dépassent 25 pb en valeur absolue.
- Corrélation de chaque cible avec la moyenne de ses 3 dernières valeurs : 0,79 à 0,91 pour bid-to-cover et parts, 0,12 à 0,42 pour le *tail*. Le benchmark `mean3` sera donc difficile à battre.
- `ruff check .` propre, 158 tests réussis.

**Choix et limites :**

- Le contrat future « le plus proche » est le 2 ans pour les bills et le 3 ans, le 10 ans pour le 7 ans, le T-Bond pour le 20 ans et le 30 ans.
- Pas de *tail* pour les premiers bills à 8, 17 et 6 semaines : le taux de marché commence le jour même.
- Les CMB, TIPS et FRN ne sont pas modélisés.

**À relire moi-même :** `features.py` (fonctions `_windowed`, `history_features`, `market_features`, `look_ahead_violations`) et `tests/test_features.py` (fonction `truncated_inputs`).

## Vérification de la phase 5 par deux agents indépendants (12 novembre 2026)

**Ma demande :** « Vérifie la phase 5 ».

**Méthode :** deux agents Claude en parallèle, en lecture seule. Le premier a cherché toute fuite d'information. Le second a recalculé chaque variable de façon indépendante et jugé si la table est prête pour les modèles. Le LLM principal a revérifié chaque constat avant de corriger.

**Verdict :**

- **Aucune fuite.** Le premier agent a reconstruit les variables date par date pour les 5 119 enchères depuis 2009 (240 593 comparaisons, 0 écart). Après les corrections, un nouveau contrôle sur 152 enchères (les 14 maturités, débuts de série, pannes, lendemains de fériés) donne 6 992 comparaisons et 0 écart.
- **Valeurs justes.** Le second agent a recalculé les 47 variables et le *tail* avec des boucles naïves sur 118 enchères : 0 écart.

**Problèmes trouvés et corrigés :**

| Gravité | Problème | Correction |
|---|---|---|
| Important | Le test de troncature était trop étroit (8 enchères réelles, sans 10 ans ni 7 ans ; données synthétiques limitées au bill 13 semaines). Deux fuites injectées exprès sur le 10 ans et le 7 ans passaient les 16 tests | Échantillon stratifié (une enchère par maturité et par période, plus le début de chaque série depuis 2009) et coupons synthétiques : ces deux fuites sont maintenant détectées (3 et 2 tests) |
| Important | Les tests ne pouvaient pas voir une date de publication fausse : ils la lisent dans la table qu'ils contrôlent (contrôle circulaire) | Contrôle **bloquant** des délais minimaux par source (taux et VIX : 0 ; trésorerie : 1 jour ouvré ; dette : 2 ; dealers : 8 jours ; CFTC : vendredi de la semaine ; WALCL, WTREGEN : 1 jour ; indices dollar : 3 jours). Dès son premier lancement, il a trouvé une date ALFRED anormale (DTWEXB du 14/05/2018, datée du jour même) et refusé d'écrire la table : un plancher par série a été ajouté |
| Important | Indice dollar vide de 2009 à 2018 (série actuelle créée en 2019) : il serait devenu un simple indicateur « avant ou après 2019 » | Raccord avec l'ancien indice DTWEXB, publié en temps réel jusqu'à fin 2019. Archive ALFRED depuis le 18/03/2014 ; avant, règle « mercredi de la semaine suivante », prudente dans 99,86 % des cas mesurés (le mardi ne l'était que dans 98,91 %). Valeurs manquantes : 0 % au lieu de 46 à 53 % |
| Important | Des niveaux en tendance (bilan de la Fed, positions des dealers, reverse repo, solde du Trésor, ratios CFTC) sortaient de la plage d'entraînement dans 12 à 26 % des cas de la période de test | Variations (solde du Trésor et reverse repo sur 4 semaines, bilan de la Fed sur 13 semaines en %) et écarts normalisés sur 52 semaines (dealers, CFTC) |
| Important | Correspondances maturité → série, fonctions dérivées, variation du montant et demande récente non testées : une inversion « 10 ans → taux 7 ans » serait passée | 14 tests ajoutés |
| Mineur | Le 30 ans était rattaché au T-Bond, qui ne livre que des titres de 15 à 25 ans depuis 2011 | Ultra Bond |
| Mineur | `offering_estimated` vaut toujours 0 dans l'échantillon | Rangé avec les identifiants, plus parmi les variables |
| Mineur | CFTC, octobre 2013 : la neutralisation jusqu'au 31/12/2013 était inutilement prudente | Calendrier officiel du communiqué 6745-13, en retenant le vendredi de chaque semaine annoncée |
| Mineur | Tableau des valeurs manquantes limité à 2009+ | Colonnes 2014+ (période de test) ajoutées |

**Problème découvert par le test lui-même après les corrections :** la troncature sur vraies données a échoué pour l'enchère du 08/01/2009. Des valeurs de l'ancien indice dollar, vides à l'époque et complétées en 2017, rendaient « inconnue » toute fenêtre de 20 jours qui les contenait. Le même effet touchait le reverse repo de 2020, complété en 2025. Règle ajoutée : une valeur publiée plus de 30 jours après sa date n'entre pas dans les calculs sur fenêtre, puisqu'elle manquait en temps réel. Ce n'était pas une fuite, car on retirait de l'information, mais le temps réel n'était pas fidèlement reproduit. Une alerte de fraîcheur permanente sur l'ancien indice, arrêté par la Fed, a aussi été levée.

**Erreurs du LLM dans ses propres tests, corrigées :** une assertion tautologique (une valeur comparée à elle-même) et une date mal calculée dans le test du raccord des indices dollar.

**À traiter en phase 6, non modifié :**

- **Le *tail* reflète surtout le mouvement de marché du jour.** Sa corrélation avec la variation de clôture de la veille au jour même est de 0,94 (bills) et 0,89 (coupons). Prévoir le *tail* revient donc largement à prévoir ce mouvement : c'est la limite du proxy prévu par le plan, à dire à l'oral. Écrêter la cible à l'entraînement.
- **Variables très corrélées :** retards et moyenne des 3 dernières, taux 3 mois et taux de la maturité pour les bills (0,998), asset managers et leveraged funds (-0,92). Cela rend Lasso et SHAP instables.
- **Cibles manquantes :** 3 *tails* (premiers bills à 8, 17 et 6 semaines), à écarter cible par cible.
- **Spread Baa et inflation anticipée avant 2014 :** leur règle « +2 jours ouvrés » n'est pas vérifiable avant l'archive ALFRED. Prévoir un test de robustesse sans ces deux variables.
- **Groupes de l'expérience :** l'indice dollar, le spread Baa et l'inflation anticipée (FRED) restent dans le groupe de base `market`. Les « nouvelles sources » sont les dealers, la CFTC et la liquidité, comme dans la section 8 du plan.
- **CFTC :** écarts normalisés absents en 2009-2011, car l'historique publié d'un coup en décembre 2010 est exclu des fenêtres.
- **Unités :** variations de taux en points de pourcentage, *tail* en points de base, dette, dollar et bilan en fractions ; les écarts normalisés n'ont pas d'unité.

**Résultat :** 46 variables (enchère 9, historique 17, marché 11, liquidité 4, positionnement 5). Sur la période de test 2014+, valeurs manquantes nulles sauf spread Baa (2,5 à 3 %) et CFTC (3 à 3,5 %). Qualité des données : vert. `ruff check .` est propre, 170 tests réussis (158 avant la vérification).

## Phase 6 — Modèles et validation walk-forward (12 et 13 novembre 2026)

**Ma demande (résumé) :** « OK, phase 6 » : section 8 du plan, c'est-à-dire 5 modèles comparés aux benchmarks `naive` et `mean3`, validation walk-forward trimestrielle depuis 2014, expérience « avec ou sans nouvelles sources », interprétation et figures.

**Ce que le LLM a produit :**

- `models.py` : modèles en pipeline scikit-learn (écrêtage 1 %/99 % et imputation par la médiane appris sur la fenêtre d'entraînement, standardisation pour les modèles linéaires), walk-forward, métriques, expérience, interprétation (permutation et SHAP) ;
- `plots.py` : 7 figures dans `reports/figures/` ;
- l'étape `--backtest` du pipeline ;
- les tests de `tests/test_models.py`.

La première version de cette section exagérait plusieurs résultats. Elle a été réécrite après la vérification par deux agents (section suivante) ; ce qui était faux y est listé.

**Méthode :**

- Chaque trimestre de 2014 à 2026 (52 trimestres) : entraînement sur tout le passé, test sur le trimestre.
- Deux familles modélisées séparément (notes et bonds, bills), 4 cibles, 5 modèles (Ridge, Lasso, ElasticNet, Random Forest, LightGBM), 2 jeux de variables (avec ou sans les nouvelles sources).
- Hyperparamètres choisis dans la fenêtre d'entraînement par validation croisée temporelle (`TimeSeriesSplit`). `random_state=42` partout.
- **Trois benchmarks** : `naive` (dernière enchère), `mean3` (moyenne des 3 dernières, la référence du plan) et `constante` (moyenne de la cible sur la fenêtre d'entraînement, c'est-à-dire un modèle sans aucune variable).
- **Modèle de référence fixé à l'avance** : la moyenne des prévisions des 5 modèles (`moyenne_5`). Elle évite de choisir le modèle d'après ses résultats sur la période de test.
- **Choix et jugement séparés** : le « meilleur » modèle est choisi sur 2014-2019 seulement, puis jugé sur 2020-2026.
- **Intervalles de confiance à 90 %** : bootstrap par années entières (2 000 tirages), pour respecter la dépendance dans le temps. On donne aussi le R² sans le trimestre le plus favorable, pour voir si un résultat tient à un seul épisode.
- **Variante « en écart à `mean3` »** (Ridge, Lasso, ElasticNet) : le modèle prévoit la surprise (cible − `mean3`) au lieu du niveau.
- En tout : 234 840 prévisions hors échantillon.

**Résultats : R² hors échantillon contre `mean3` (%), toutes variables, moyenne des 5 modèles, intervalle à 90 %**

| Famille | Cible | 2014-2026 | 2020-2026 | Sans le meilleur trimestre | Contre la constante |
|---|---|---|---|---|---|
| Notes et bonds | Bid-to-cover | **11,6** [7,9 ; 15,4] | **9,6** [5,7 ; 14,5] | 10,0 | 81,9 |
| Notes et bonds | Part des dealers | -13,3 [-24,7 ; -2,5] | -28,5 | -15,5 | 87,9 |
| Notes et bonds | Part des indirects | 3,8 [-1,9 ; 8,8] | 5,7 | 2,9 | 82,4 |
| Notes et bonds | Tail | 21,6 [16,6 ; 26,5] | 22,1 | 20,8 | **2,5** |
| Bills | Bid-to-cover | **20,3** [16,4 ; 23,8] | **21,2** [13,3 ; 26,2] | 17,2 | 85,9 |
| Bills | Part des dealers | 8,9 [4,1 ; 13,6] | 1,5 [-4,0 ; 5,7] | 8,3 | 70,6 |
| Bills | Part des indirects | 10,4 [6,8 ; 13,9] | 6,0 [1,5 ; 10,4] | 9,6 | 72,3 |
| Bills | Tail | 21,5 [-1,2 ; 29,8] | 22,3 | **5,0** | 27,3 |

Lecture de la dernière colonne : la constante est calculée pour toute la famille, toutes maturités confondues. Elle est donc très mauvaise pour le bid-to-cover et les parts, dont le niveau dépend beaucoup de la maturité ; `mean3` est alors le vrai benchmark à battre. Pour le *tail*, dont le niveau varie peu d'une maturité à l'autre, la constante est un benchmark sérieux.

**Ce qui tient, et ce qui ne tient pas :**

- **Bid-to-cover des notes et bonds : le résultat solide.** La moyenne des 5 bat `mean3` sur chaque période, avec un intervalle loin de 0. L'année est positive 11 fois sur 13 (pire année : 2022, -3,4 %). Le gain cumulé monte régulièrement, sans saut (figure `gain_cumule_nominal_coupon_btc.png`). Bonne direction : 61,4 %.
- **Bid-to-cover des bills : un gain réel, mais gonflé par les crises du plafond de la dette.** Le meilleur trimestre est le 4e de 2015 ; sans les bills à 4 semaines, le R² tombe de 20,3 à 13,3 %. Les sauts d'octobre 2015 et d'octobre 2021 se voient sur la figure `gain_cumule_bill_btc.png`.
- **Tail : ne pas le présenter comme un succès.** Pour les notes et bonds, une simple constante bat déjà `mean3` de 19,5 % ; les modèles ne font que 2,5 % de mieux qu'elle. Le +21,6 % mesure surtout la faiblesse de `mean3`. Pour les bills, le gain tient presque entièrement au deuxième trimestre 2023 (plafond de la dette) : sans lui, il reste 5 %, et l'intervalle contient 0.
- **Part des dealers des notes et bonds : le problème venait de la spécification.** Les modèles en niveau perdent contre `mean3` (-12 à -22 %), parce que cette part a baissé de 37,5 % à 14 % : ils restent tirés vers la moyenne du passé. La variante en écart à `mean3` donne +2,9 à +3,4 % (Lasso : intervalle [1,0 ; 4,9]). Ce n'est donc pas « imprévisible », c'est un niveau qui dérive.
- **Part des indirects des notes et bonds :** impossible à distinguer de `mean3` en niveau ; la variante en écart donne +1 à +3 %.
- **Parts des bills :** bonnes en moyenne, mais le gain s'efface depuis 2020 pour la part des dealers (+1,5 %, intervalle autour de 0).

**Choix du modèle sur 2014-2019, jugé sur 2020-2026 :**

| Famille | Cible | Modèle choisi | R² au choix | R² ensuite [IC 90 %] | Moyenne des 5 ensuite |
|---|---|---|---|---|---|
| Notes et bonds | Bid-to-cover | Lasso | 9,7 | 8,1 [4,9 ; 12,1] | 9,6 |
| Notes et bonds | Part des dealers | LightGBM | -3,9 | -26,3 | -28,5 |
| Notes et bonds | Part des indirects | LightGBM | 5,6 | -0,7 | 5,7 |
| Notes et bonds | Tail | Lasso | 20,5 | 20,1 | 22,1 |
| Bills | Bid-to-cover | ElasticNet | 19,9 | 18,6 [8,9 ; 24,8] | 21,2 |
| Bills | Part des dealers | LightGBM | 16,1 | -4,5 | 1,5 |
| Bills | Part des indirects | Lasso | 14,8 | 4,2 | 6,0 |
| Bills | Tail | LightGBM | 15,2 | 20,0 | 22,3 |

La moyenne des 5 fait aussi bien ou mieux que le modèle choisi dans 7 cas sur 8. C'est elle qui servira de modèle de référence pour la stratégie (phase 7) et l'application.

**Comparaison avec les ordres de grandeur du plan (version R)**, avec un seul modèle, la moyenne des 5 :

| | Plan | Obtenu |
|---|---|---|
| Bid-to-cover des notes et bonds, R² contre `mean3` | environ +8 % | +11,6 % [7,9 ; 15,4] ; Lasso seul : +8,9 % |
| Bonne direction | 62 % | 61,4 % (Lasso seul : 59,9 %) |
| Part des dealers | légèrement négatif | -13 % en niveau ; +3 % en écart à `mean3` |
| Benchmark naïf | environ -60 % | -59,5 % |

**Expérience clé : les nouvelles sources n'apportent rien.** Pour la moyenne des 5, l'écart « avec − sans » va de -1,9 à +0,1 point selon la cible. Sur les 48 comparaisons (6 modèles × 8 cibles), aucune n'a un intervalle entièrement positif ; 17 ont un intervalle entièrement négatif (surtout les parts des bills). Lecture : les dealers, la CFTC et la liquidité ne sont pas « cassés » ; ils ne portent simplement pas d'information stable sur la surprise de l'enchère, et ajouter des variables inutiles coûte un peu de précision.

**Interprétation (permutation et SHAP, modèle choisi, entraîné avant 2022 et expliqué sur 2022-2026) :**

- Parts SHAP normalisées par modèle, puis moyennées sur les 8 cibles : historique de la série 62 %, marché 24 %, enchère 9 %, nouvelles sources 4,6 %.
- La permutation (ce qui aide vraiment hors échantillon) et SHAP (ce qui pèse dans la formule) divergent souvent. Pour le bid-to-cover des notes et bonds, les parts des dealers passées sont 2e à 4e par SHAP mais 10e à 13e par permutation, avec un apport quasi nul : elles pèsent parce que cette part a dérivé, pas parce qu'elles aident. Pour les bills, le montant (log) est 3e par SHAP mais dégrade la prévision quand on le garde (46e par permutation).
- Pour le *tail* des notes et bonds, aucune variable n'a une importance par permutation supérieure à deux écarts-types : cohérent avec un modèle proche d'une constante.

**Choix et écarts avec le plan :**

| Point | Décision | Raison |
|---|---|---|
| Hyperparamètres de la Random Forest | fixes (300 arbres, 20 observations par feuille, 33 % des variables) | une recherche par trimestre aurait multiplié le temps de calcul par 10 ; la forêt y est peu sensible |
| LightGBM | grille de 9 combinaisons (7, 15 ou 31 feuilles ; 150, 400 ou 800 arbres) choisie à chaque trimestre par `TimeSeriesSplit(3)` | la première grille de 4 combinaisons butait sur un coin ; l'élargir change peu les résultats (bid-to-cover des notes et bonds : 7,2 → 7,0 %) |
| Cible d'entraînement | écrêtée à 1 % et 99 % (évaluation sur la cible réelle) | les *tails* de crise (+143 pb) ne doivent pas dicter le modèle |
| Validation croisée interne | prétraitement appris sur toute la fenêtre d'entraînement avant la validation interne | pratique courante ; cela ne touche que le choix des hyperparamètres, jamais la période de test |
| Modèle de référence | moyenne des 5, fixée avant de voir les résultats de 2020-2026 | évite de choisir sur le test |
| Variante en écart à `mean3` | seulement pour les 3 modèles linéaires, toutes variables | test de robustesse rapide ; elle n'existe pas quand `mean3` manque (57 prévisions sur 234 840, en début de série, exclues de tous les scores) |
| Figures | Matplotlib ajouté aux dépendances | figures statiques pour l'oral et le rapport (Plotly reste pour l'application) |

**Problèmes rencontrés et corrigés :**

- scikit-learn 1.9 a supprimé le paramètre `n_alphas` de `LassoCV` et `ElasticNetCV` : remplacé par `alphas=30` ;
- Matplotlib manquait : ajouté ;
- figures : légende qui recouvrait une étiquette, décimales à l'anglaise, graduations de panneaux voisins qui se chevauchaient, étiquettes négatives écrites sur les barres, étiquettes des crises qui se touchaient : corrigé après relecture de chaque image ;
- un jeu de données de test du LLM était à l'envers (les prévisions « sans » étaient plus proches du réel que les « avec ») : corrigé ;
- un nouveau test du LLM comparait aussi les trimestres suivant un choc, qui changent normalement puisque l'enchère choquée entre dans l'entraînement : corrigé.

**Limite pratique :** le backtest complet prend moins de 2 heures sur 8 cœurs (100 minutes mesurées pour la première version, plus 17 minutes pour la grille LightGBM élargie et la variante en écart ; estimation initiale du LLM : 5 minutes). Il ne tourne donc pas chaque soir : la mise à jour automatique (phase 10) ré-entraîne seulement les modèles finaux.

**Figures** (`reports/figures/`) :

- `r2_par_modele_{nominal_coupon,bill}.png` : R² sur 2020-2026 avec intervalles, modèle choisi en bleu, moyenne des 5 en orange, même échelle pour toutes les cibles ;
- `nouvelles_sources.png` : avec ou sans, et l'intervalle de l'écart ;
- `gain_cumule_{nominal_coupon,bill}_btc.png` : gain cumulé de la moyenne des 5, en part de l'erreur de `mean3` (la courbe finit au R²), crises du plafond de la dette marquées ;
- `importance_{nominal_coupon,bill}_btc.png` : permutation et SHAP côte à côte, noms des variables en français.

**Sorties** dans `data/output/` : `backtest_predictions.parquet` (avec la colonne `constant`), `backtest_metrics.csv` (une ligne par période, avec intervalles et R² sans le meilleur trimestre), `new_sources_comparison.csv` (avec intervalles), `best_models.csv` (choix sur 2014-2019, jugement sur 2020-2026), `feature_importance.csv` (avec `shap_share`).

**À relire moi-même :** `models.py` (fonctions `walk_forward`, `score`, `bootstrap_r2`, `best_models`) et la figure `r2_par_modele_nominal_coupon.png`, pour l'oral.

## Vérification de la phase 6 par deux agents indépendants (13 novembre 2026)

**Ma demande :** « Vérifie la phase 6 ». Deux agents ont relu le travail sans rien modifier : l'un la méthode (fuites, walk-forward, tests), l'autre les résultats, leur robustesse, leur interprétation et les figures. Les agents ne lisent jamais le fichier `.env`.

**Ce qui était juste :**

- **Aucune fuite.** Rien ne laisse les données du trimestre testé entrer dans l'entraînement. Cinq tâches relancées reproduisent les prévisions enregistrées au bit près.
- Les 96 lignes de métriques recalculées indépendamment correspondent exactement ; tous les modèles d'une même cible sont jugés sur les mêmes enchères.
- Le décalage `lag1` recalculé pour 3 910 enchères de test : aucune erreur.
- Les deux jeux de variables diffèrent exactement des 9 variables de liquidité et de positionnement.

**Ce qui était faux ou exagéré dans la première version de ce journal :**

| Affirmation de la première version | Ce que la vérification a montré | Correction |
|---|---|---|
| « Bills : les gains sont nets sur les quatre cibles » | faux pour le *tail* (le gain tient au 2e trimestre 2023) et exagéré pour le bid-to-cover (crises du plafond de la dette) | R² sans le meilleur trimestre, intervalles, crises marquées sur la figure |
| *Tail* des notes et bonds : « +20 à 22 % » présenté comme un des meilleurs résultats | une constante fait déjà +19,5 % : c'est `mean3` qui est faible | benchmark `constante` ajouté ; le *tail* n'est plus présenté comme un succès |
| Part des dealers : « aucun modèle ne bat `mean3` » | les modèles en niveau ratent une baisse de 37,5 % à 14 % ; en écart à `mean3`, +3 % | variante en écart ajoutée |
| « +8,9 % (Lasso) » comparé aux « 62 % de bonne direction (LightGBM) » | deux modèles différents ; le Lasso fait 59,9 % | un seul modèle de référence, la moyenne des 5 |
| Le meilleur modèle désigné sur toute la période 2014-2026 | choisi sur la période même où on le juge : optimiste | choix sur 2014-2019, jugement sur 2020-2026 |
| Parts SHAP « 56/26/12/6 % » | somme de valeurs SHAP dans des unités différentes (le *tail* en pb pesait 66 %) | parts normalisées par modèle ; SHAP sur toutes les enchères 2022-2026 (avant : les 800 premières seulement) ; permutation affichée à côté |
| « Le R² baisse avec les nouvelles sources » | pour la plupart des cibles, l'intervalle de l'écart contient 0 | intervalles ajoutés ; message : « n'apportent rien » |

**Autres corrections :**

- Les tests laissaient passer plusieurs erreurs de walk-forward (un agent les a injectées une à une). Dix tests ont été ajoutés (18 au total pour les modèles) :
  - une enchère tombe exactement le premier jour d'un trimestre de test ;
  - brouiller tout l'après d'une date ne change aucune prévision d'avant ;
  - une valeur extrême de la cible dans le trimestre testé ne change pas l'écrêtage ;
  - une valeur extrême d'une variable ne change que sa propre prévision ;
  - variante en écart, benchmark constant, moyenne des 5, bootstrap, R² sans le meilleur trimestre, choix du modèle sur 2014-2019, égalités exclues de la direction.
- J'ai ensuite injecté 13 erreurs volontaires dans une copie du code (par exemple entraîner jusqu'au jour même, écrêter avec tout l'historique, choisir le modèle sur 2020-2026). Les 13 sont détectées, et la copie sans erreur passe tous les tests.
- La corrélation entre prévision et réalité était peu informative (le niveau varie surtout avec la maturité) : ajout de la corrélation des surprises (réel − `mean3` contre prévu − `mean3`).
- Figures refaites (voir la section Phase 6).

**Robustesse demandée en phase 5 (sans le spread Baa et l'inflation anticipée)**, faite par l'agent : les R² bougent d'environ 1 point, la conclusion ne change pas.

| Modèle | Avec | Sans |
|---|---|---|
| Notes et bonds, bid-to-cover, Lasso | 8,95 | 9,52 |
| Notes et bonds, bid-to-cover, ElasticNet | 8,89 | 9,13 |
| Bills, bid-to-cover, ElasticNet | 19,38 | 18,05 |
| Bills, bid-to-cover, Lasso | 18,80 | 18,21 |

**Limites connues, non corrigées :**

- La validation croisée interne découpe par ligne et non par date ; cela ne touche que le choix des hyperparamètres dans la fenêtre d'entraînement.
- `weekday` et `maturity_years` sont des nombres : sans effet pour les arbres, une simplification pour les modèles linéaires.
- La variation du reverse repo sort souvent de la plage d'entraînement (2014, 2021, 2023) : l'écrêtage la limite.
- Rien ne prouve ni n'exclut que les réglages de la Random Forest ou la grille LightGBM aient été choisis après avoir vu les résultats.

**Résultat vérifié :** 188 tests réussis (dont 18 pour les modèles), `ruff check .` propre.

## Phase 7 — Lien avec le marché et stratégie illustrative (13 novembre 2026)

**Ma demande :** « phase 7 » : section 9 du plan. D'abord, les enchères faibles s'accompagnent-elles d'une hausse des taux ? Ensuite, une stratégie : vendre de la duration de la clôture de t-1 à la clôture de t quand le modèle prévoit un bid-to-cover sous `mean3`, en acheter sinon.

**Ce que le LLM a produit :**

- `strategy.py` : variation du taux de clôture à clôture, quintiles de surprise par période, positions et P&L de chaque règle, synthèse par période, et tables de contrôle (par année, sensibilité au coût, autres signaux, étude d'événement, diagnostics) ;
- deux figures dans `plots.py` ;
- une étape `--strategy` dans le pipeline, qui relance la phase 7 en quelques secondes sans refaire le backtest (environ 2 heures) ;
- 18 tests (`tests/test_strategy.py`).

La première version de cette section contenait deux explications fausses. Elle a été réécrite après la vérification par deux agents (section suivante).

**Méthode :**

- **Notes et bonds seulement** : la duration d'un bill est trop courte pour un pari sur les taux.
- **Variation du taux** : taux de la maturité de l'enchère (courbe du Trésor), de la dernière clôture publiée strictement avant l'enchère (au plus 10 jours avant) à la clôture du jour de l'enchère, en points de base. Le résultat de l'enchère tombe à 13 h, donc avant la clôture du jour.
- **Surprise** : valeur réelle moins moyenne des 3 dernières enchères de la même maturité. Quintile 1 = demande la plus forte, quintile 5 = la plus faible, pour les deux mesures.
- **Signal** : la moyenne des 5 modèles, fixée à l'avance en phase 6 (prévisions hors échantillon 2014-2026).
- **P&L** = position × variation du taux − 0,5 pb, avec position +1 si vendeur (on gagne quand le taux monte) et -1 si acheteur.
- **Deux références** :
  - « toujours vendeur », demandée par le plan ;
  - « information parfaite sur le sens du bid-to-cover », qui connaît le vrai résultat à l'avance : impossible en pratique, c'est le plafond d'une règle fondée sur ce signal.
- **Sharpe** : calculé sur le P&L total de chaque jour d'enchère (deux enchères le même jour forment un seul portefeuille), annualisé par la racine du nombre de jours d'enchère par an.
- **Intervalles à 90 %** : bootstrap par années entières, comme en phase 6, y compris pour les quintiles. Avec 6 ou 7 années par sous-période, ils sont approximatifs.

**Résultat 1 : les enchères faibles s'accompagnent d'une hausse du taux le jour même** (notes et bonds 2009-2026, 1 360 enchères, `quintiles_surprise.png`)

| Quintile de surprise | 1 (forte) | 2 | 3 | 4 | 5 (faible) |
|---|---|---|---|---|---|
| Part des dealers : variation du taux (pb) | -0,7 | -0,7 | -1,0 | +0,9 | **+1,8** [1,2 ; 2,5] |
| Bid-to-cover : variation du taux (pb) | -1,4 | -1,0 | -0,3 | +1,1 | **+2,0** [1,2 ; 2,8] |

- Le plan attendait environ +1,8 pb pour les enchères les plus faibles : c'est ce qu'on obtient sur 2009-2026.
- La taille dépend de la période. Pour le quintile 5 de la part des dealers : +3,1 pb sur 2009-2013, +1,4 pb sur 2014-2019, +1,4 pb sur 2020-2026. Le sens est le même partout.
- La relation n'est pas régulière d'un quintile à l'autre (le quintile 3 est le plus bas pour les dealers).
- On parle d'association, pas de cause. La fenêtre de clôture à clôture inclut la matinée, avant le résultat de 13 h : une mauvaise nouvelle macroéconomique le matin peut à la fois faire monter les taux et affaiblir l'enchère.
- Les bornes des quintiles sont calculées sur toute la période : c'est sans danger pour une analyse descriptive faite après coup, mais ce serait interdit pour une stratégie.

**Résultat 2 : la stratégie perd de l'argent** (1 004 trades, 2014-2026, `strategie_pnl_cumule.png`)

| Règle | P&L moyen par trade, après coût [IC 90 %] | Avant coût | Écart avec « toujours vendeur » [IC 90 %] | Réussite après coût | Total | Sharpe |
|---|---|---|---|---|---|---|
| Stratégie (moyenne des 5) | **-1,00** [-1,41 ; -0,59] | -0,50 | -0,30 [-0,77 ; +0,13] | 42,8 % | -1 009 pb | -1,74 |
| Toujours vendeur | -0,70 [-1,27 ; -0,13] | -0,20 | | 44,5 % | -707 pb | -1,19 |
| Information parfaite (impossible) | +0,42 [0,15 ; 0,66] | +0,92 | +1,13 [0,52 ; 1,74] | 53,0 % | +423 pb | +0,73 |

- Chaque règle paie 502 pb de coûts (0,5 pb × 1 004 trades) : la moitié de la perte de la stratégie vient des coûts.
- Avant coût, la stratégie perd quand même 0,5 pb par trade, et c'est significatif (intervalle [-0,91 ; -0,09]). « Toujours vendeur » ne perd pas significativement avant coût.
- La stratégie ne fait **pas mieux** que « toujours vendeur » ; elle fait un peu moins bien, mais l'écart n'est pas significatif.
- Sensibilité au coût, pour la stratégie : -0,51 pb par trade sans coût, -1,51 pb avec 1 pb. L'information parfaite devient nulle avec un coût de 1 pb.
- Tous les autres signaux disponibles donnent la même conclusion : les 15 (5 modèles, moyenne des 5 et 3 variantes en écart, avec ou sans nouvelles sources) perdent de -0,71 à -1,10 pb par trade après coût, et aucun ne bat « toujours vendeur » de façon significative. Ils n'ont servi qu'à vérifier ; la règle de référence reste la moyenne des 5, choisie avant.

**Lecture, à dire à l'oral :**

- **Même en connaissant le résultat de l'enchère à l'avance, on gagne peu** : +0,9 pb par trade avant coût, +0,4 pb après. La surprise de l'enchère n'explique qu'une petite partie du mouvement de taux du jour (corrélation -0,22).
- **La partie prévue de la demande n'a pas d'effet mesurable sur le taux le jour de l'enchère**, contrairement à la partie non prévue (corrélation -0,29 avec la variation du taux).
  - Deux lectures sont possibles : elle est déjà dans les prix (la conclusion attendue par le plan), ou elle n'a simplement pas d'effet. Nos données ne permettent pas de trancher.
  - Si elle était déjà dans les prix, le taux monterait davantage avant les enchères prévues faibles ; ce n'est pas le cas (de t-6 à t-1 : +1,7 pb quand le modèle est vendeur, +1,2 pb quand il est acheteur, écart non significatif).
  - Surtout, même si la partie prévue faisait bouger le taux autant que la surprise réelle, la stratégie ne gagnerait qu'environ +0,4 pb avant coût : moins que le coût de 0,5 pb. Perdre après coûts était donc attendu dans les deux cas.
- **Pourquoi la stratégie fait un peu moins bien que « toujours vendeur »** (écart total -302 pb) :
  - 2014-2019 : -228 pb. Les taux montaient en moyenne les jours d'enchère, et la stratégie était acheteuse une fois sur deux.
  - 2020-2021 : -220 pb, dont -188 pb en 2021, année où la stratégie était acheteuse 75 % du temps alors que les taux montaient de 1,7 pb en moyenne les jours d'enchère.
  - 2022-2026 : +146 pb. La stratégie était surtout vendeuse, mais moins que « toujours vendeur », dans des années où les taux baissaient les jours d'enchère.
- **Le penchant vendeur de 2022-2025 est un biais du modèle, pas une information.** Le modèle prévoit un bid-to-cover sous `mean3` dans 67 à 81 % des cas ces années-là, alors que le vrai bid-to-cover n'est sous `mean3` que dans 42 à 54 % des cas. Ses prévisions restent tirées vers les niveaux du passé, quand les taux sont hauts et les montants émis en hausse.
- **Le comportement des taux autour des enchères a changé vers 2022** (`event_study.csv`) :

  | Période | De t-6 à t-1 | Jour de l'enchère | De t à t+3 |
  |---|---|---|---|
  | 2014-2021 | +0,5 pb | +0,7 pb | -1,4 pb |
  | 2022-2026 | +2,7 pb | -1,6 pb | +1,4 pb |

  Le taux monte avant l'enchère (le marché « fait de la place » au nouveau papier), puis redescend. Ce retour se faisait après l'enchère avant 2022, et le jour même depuis 2022 : c'est ce qui fait chuter « toujours vendeur » à partir de 2022. Ce schéma est décrit dans la littérature (concession avant l'enchère, puis retour) ; la cause du changement de 2022 n'est pas établie ici.
- **Le signal « à l'envers » n'est pas exploitable non plus.** Avant coût, il rapporte +0,5 pb par trade (significatif) ; après coût, 0,0 pb. Ce gain vient du biais vendeur dans un régime de baisse des taux les jours d'enchère, pas d'une information sur la demande.

**Choix et limites :**

| Point | Décision ou limite |
|---|---|
| P&L en points de base de taux | comme le demande le plan, à sensibilité constante (même DV01 pour chaque trade). Pondérer par la duration ne change pas la conclusion (agent : stratégie -0,88 pb, « toujours vendeur » -0,49, information parfaite +0,47) |
| Taux arrondis à 0,01 % | 9,8 % des trades ont une variation nulle ; avec le coût, ils comptent comme des pertes (réussite avant coût hors variations nulles : 47,5 %) |
| Moment de l'entrée | le signal utilise la clôture de t-1 (courbe du Trésor, VIX), publiée en fin de journée : entrer à cette clôture exacte est un peu optimiste. L'agent a refait le walk-forward avec un jour de retard : -1,06 pb par trade au lieu de -1,00 (47 positions sur 1 004 changent). Effet négligeable |
| Égalités | aucune prévision n'est égale à `mean3` ; 6 bid-to-cover réels le sont, et la règle « information parfaite » est alors acheteuse |
| Une seule règle de signal | la moyenne des 5, fixée en phase 6 ; les autres signaux ne servent qu'à vérifier |

**Résultat vérifié :** voir la section suivante.

**À relire moi-même :** `strategy.py` (fonctions `rate_changes` et `strategy_trades`) et les deux figures, pour l'oral.

## Vérification de la phase 7 par deux agents indépendants (13 novembre 2026)

**Ma demande :** « Vérifie la phase 7 ». Deux agents ont relu le travail sans rien modifier : l'un le code (moment des taux, jointures, fuites, tests), l'autre les résultats, leur robustesse, leur interprétation et les figures. Les agents ne lisent jamais le fichier `.env`.

**Ce qui était juste :**

- **Aucune fuite.** La variation du taux va bien de la dernière clôture strictement avant l'enchère à la clôture du jour ; la stratégie n'utilise que la prévision et `mean3`.
- Les deux agents ont recalculé indépendamment toutes les variations de taux (1 004 trades, 1 362 enchères pour les quintiles), les positions, les quintiles et la synthèse : écart nul partout.
- Les dates des taux sont alignées sur des événements connus (par exemple le 10 ans le 9 novembre 2016, +19 pb).
- Tous les chiffres de la première version du journal étaient exacts.

**Ce qui était faux ou exagéré dans la première version :**

| Affirmation de la première version | Ce que la vérification a montré | Correction |
|---|---|---|
| La stratégie fait pire que « toujours vendeur » parce qu'elle était surtout vendeuse en 2022-2024 | être surtout vendeur rapproche de « toujours vendeur », sans pouvoir faire pire. Le retard vient de 2014-2021 (-448 pb) ; depuis 2022, la stratégie fait mieux (+146 pb). Et l'écart n'est pas significatif | explication réécrite ; écart et intervalle ajoutés à la synthèse |
| « Conclusion du plan confirmée : la partie prévisible de la demande est déjà dans les prix » | la corrélation +0,10 vient d'un effet de période (+0,05 à l'intérieur de chaque année) ; aucune hausse plus forte avant les enchères prévues faibles ; le test n'a pas la puissance de trancher (gain maximal ≈ 0,4 pb < coût) | formulation prudente : « pas d'effet mesurable » |
| « La stratégie inverse rapporte 0,0 pb » | vrai après coût, par coïncidence : avant coût elle gagne +0,5 pb, à cause du biais vendeur | chiffre avant coût et explication ajoutés |
| « +1,8 pb » présenté comme une constante | +3,1 pb sur 2009-2013, +1,4 pb ensuite | quintiles par période |
| « Les enchères faibles font monter le taux » | la fenêtre inclut la matinée : cause non démontrée | « s'accompagnent d'une hausse » |
| « +0,9 pb avant coûts, Sharpe 0,7 » | mélange d'un chiffre avant coût et d'un Sharpe après coût | corrigé |
| Entrée à la clôture de t-1 : « cela avantage la stratégie » | sens non démontré au départ ; mesuré par l'agent : 0,05 pb par trade | chiffre dans le tableau des limites |
| « 7 erreurs injectées, toutes détectées » | l'agent en a injecté 55 : 13 qui changeraient les vrais résultats passaient (par exemple compter chaque enchère deux fois, ou évaluer le 20 ans avec le taux du 30 ans) | 10 tests ajoutés (voir plus bas) |

**Corrections du code :**

- les dates avec une heure sont refusées (même contrôle que pour les variables) ;
- une période sans trade ne fait plus planter la synthèse ;
- le Sharpe utilise la durée réelle couverte (2026 n'est pas une année complète) ;
- l'univers de la stratégie ne dépend plus du résultat de l'enchère (seule l'information parfaite l'utilise) ;
- les intervalles des quintiles sont calculés par bootstrap par année, comme le reste du projet ;
- un contrôle vérifie que les prévisions enregistrées correspondent aux variables actuelles ;
- nouvelle étape `--strategy` ;
- chaque chiffre du journal est maintenant recalculable depuis `data/output/` : `strategy_by_year.csv`, `strategy_costs.csv`, `strategy_signals.csv`, `event_study.csv`, `strategy_diagnostics.csv`, et la colonne `period` de `quintiles.csv` ;
- figures : titres plus courts, espaces des milliers, vrai signe moins dans toutes les figures du projet, ligne « information parfaite » plus visible, coûts indiqués, repère en 2022.

**Tests :** 10 tests ajoutés, 18 en tout. Ils couvrent :
- le bootstrap par année et son poids par nombre d'enchères ;
- le filtre du jeu de variables ;
- une part de vendeurs asymétrique et le total ;
- le 20 ans et le 30 ans ;
- les numéros, intervalles et parts de hausse des quintiles ;
- la limite des 10 jours ;
- le refus des heures et les périodes vides ;
- l'étude d'événement et la sensibilité au coût ;
- un test de bout en bout et le contrôle de désynchronisation.

J'ai ensuite injecté 26 erreurs dans une copie de `strategy.py`, dont les 13 qui échappaient aux tests : les 26 sont détectées, et la copie sans erreur passe tous les tests.

**Limites connues, non corrigées :**

- L'heure exacte de la courbe du Trésor et un éventuel changement de titre de référence le jour de l'enchère ne sont pas vérifiables sans données intrajournalières.
- Le coût de 0,5 pb est une hypothèse ; la sensibilité au coût est donnée.

**Résultat vérifié :** 206 tests réussis, `ruff check .` propre.

## Phase 8 — Prévisions en direct et journal figé (13 novembre 2026)

**Ma demande :** « ok phase 8 » : section 10 du plan. Entraîner les modèles sur tout l'historique, prévoir les enchères à venir, tenir un journal des prévisions où une ligne est figée à partir du jour de l'enchère, et suivre les résultats dès leur publication.

**Ce que le LLM a produit :**

- `live.py` : entraînement final et sauvegarde (joblib), prévision des enchères à venir, journal figé, suivi en direct, état de la dernière exécution ;
- les étapes `--train` et `--predict` du pipeline ;
- 35 tests (`tests/test_live.py`), dont le test obligatoire de la ligne figée.

La première version de cette section exagérait certaines garanties. Elle a été réécrite après la vérification par deux agents (section suivante).

**Méthode :**

- **Modèles finaux** : les 5 modèles, avec la même recette que le walk-forward (prétraitement dans le pipeline, cible écrêtée, mêmes réglages : alphas des modèles linéaires et grille LightGBM choisis par validation croisée temporelle, Random Forest à réglages fixes). Ils sont entraînés sur toutes les enchères réalisées depuis 2009, avec toutes les variables. Un test vérifie qu'entraîné sur les mêmes enchères, le modèle final prévoit exactement comme le walk-forward.
- **`metadata.json`** : date d'entraînement, dernière enchère, empreinte des données d'entraînement, liste des variables, versions (Python, numpy, pandas, scikit-learn, LightGBM, joblib). À la relecture, les versions et la liste des variables sont vérifiées ; en cas d'écart, la prévision s'arrête et demande `--train`.
- **Pas de ré-entraînement inutile** : si les données d'entraînement n'ont pas changé (même empreinte, mêmes versions), `--train` garde les modèles existants. Ré-entraîner donnerait les mêmes prévisions, mais pas toujours des fichiers identiques octet par octet.
- **Prévision** : pour chaque enchère à venir (notes et bonds, bills), les 4 cibles par les 5 modèles et leur moyenne, le modèle de référence fixé en phase 6.
- **Journal** `data/output/predictions_log.parquet`, clé `(cusip, auction_date, target, model)`, avec `predicted_at` (UTC), `pred`, `bench_mean3`, le montant et `offering_estimated`. Les types de colonnes sont fixés à l'écriture comme à la relecture.
- **Règle du gel** :
  - une prévision n'est écrite que pour une enchère dont la date est strictement postérieure à la date du jour à New York ; la clé d'une ligne figée, qui contient la date de l'enchère, ne peut donc correspondre à aucune nouvelle prévision ;
  - l'horloge doit avancer : un instant antérieur à la dernière prévision du journal est refusé, sinon relancer « hier » pourrait rouvrir une ligne figée aujourd'hui ;
  - avant chaque écriture, un garde-fou vérifie que chaque ligne figée est présente et identique, toutes colonnes comprises ; sinon l'écriture est annulée ;
  - une prévision inchangée est gardée telle quelle, valeur et date d'origine ;
  - aucune prévision n'est créée le jour même : le workflow tourne le soir (heure de New York), après la publication du résultat ;
  - si une autre exécution a modifié le journal pendant le calcul, l'écriture est refusée ;
  - un journal disparu alors que le suivi existe n'est pas recréé vide : il faut le restaurer depuis git ;
  - les écritures passent par un fichier temporaire.
- **Suivi** `data/output/live_tracking.parquet` : le journal joint aux résultats publiés, avec l'erreur du modèle, celle de `mean3` et le statut :
  - `realisee` : enchère réalisée, même si une cible manque, par exemple un tail non publié ;
  - `sans_resultat` : date passée sans résultat (enchère déplacée, annulée ou pas encore collectée) ;
  - `en_attente` sinon.
- **État** `data/output/live_status.json` : heure de l'exécution, date à New York, nombre d'enchères à venir, et dernière date de chaque série de marché, pour montrer la fraîcheur des données dans l'application.
- **Pipeline** : si une source échoue au téléchargement, les autres sont enregistrées et la suite (nettoyage, entraînement, prévision) tourne sur les dernières données disponibles ; le code de sortie signale quand même l'échec. Sans cela, une seule panne un soir pouvait priver une enchère de toute prévision : en 2025, 59 enchères sur 406 n'ont eu qu'un seul soir entre l'annonce et la veille.

**Résultat** (13 novembre 2026, données téléchargées le jour même) :

- `--fetch --clean --train --predict` en 114 secondes, dont 35 secondes d'entraînement (sur 8 cœurs) ; contrôle qualité vert.
- 4 enchères à venir, 96 lignes dans le journal (4 cibles × 6 modèles), toutes « en attente ».
- Après la correction du code, le journal relu puis réécrit est identique au bit près (valeurs et heures d'origine).

| Enchère | Bid-to-cover prévu (moyenne des 5) | Moyenne des 3 dernières | Écart |
|---|---|---|---|
| 13 semaines, 16 novembre | 2,792 | 2,873 | −0,082 |
| 26 semaines, 16 novembre | 2,877 | 2,813 | +0,063 |
| 6 semaines, 17 novembre | 2,709 | 2,767 | −0,058 |
| 20 ans, 18 novembre | 2,532 | 2,610 | −0,078 |

Le signal de l'application (demande plus faible si l'écart est inférieur ou égal à −0,08) devra se calculer sur les valeurs non arrondies : le 13 semaines le déclenche, le 20 ans non, alors que les deux s'affichent « −0,08 » une fois arrondis.

**Contrôle de reproductibilité :**

- Les variables des 6 745 enchères passées sont identiques avant et après le nouveau téléchargement. Le test est faible : ce jour-là, seules 3 nouvelles observations de marché sont arrivées, et aucune enchère.
- Un agent a refait le nettoyage et les variables à partir des données brutes commitées : fichiers identiques octet par octet.
- Ré-entraînés, les modèles donnent exactement les mêmes prévisions que les modèles sauvegardés et que le journal (96 sur 96).
- `backtest_predictions.parquet` reste synchronisé avec les variables actuelles.

**Choix et points ouverts :**

| Point | Décision |
|---|---|
| Modèles en direct | les 5 modèles et leur moyenne sont tous inscrits au journal ; l'application affichera la moyenne des 5 |
| Variante en écart à `mean3` | pas utilisée en direct : elle n'était qu'un test de robustesse en phase 6 |
| Stockage des modèles (8,6 Mo, dont 7,9 Mo de Random Forest) | exclus de git pour l'instant. Les commiter chaque nuit ferait grossir le dépôt d'environ 1,6 Go par an (mesuré par un agent). À décider en phase 10 ; options mesurées : un « asset » de Release GitHub ou une branche à part, sans historique qui grossit ; ou une Random Forest de 100 arbres (3,3 Mo au total, prévisions déplacées d'au plus 0,001 sur le bid-to-cover), qu'il faudrait alors noter comme écart au plan |
| Un seul programme écrit le journal | le bouton « Mettre à jour maintenant » de l'application ne devra pas écrire le journal (sur Streamlit Cloud, le fichier serait perdu au redéploiement) : il affichera une prévision provisoire. Seul le workflow de nuit écrit |
| `--all` | enchaîne toutes les étapes, backtest compris (environ 2 heures). Le workflow de nuit lancera seulement `--fetch --clean --train --predict` : c'est un écart à la section 12 du plan |
| Montant estimé | d'abord jugé inutile (le 13 novembre à midi, TreasuryDirect ne listait que des enchères annoncées, avec leur montant). Faux : le soir même, le calendrier listait 10 nouvelles enchères sans montant, jusqu'à 12 jours à l'avance ; le montant « estimé » sert donc bien (voir la phase 9) |

**À relire moi-même :** `live.py` (fonction `update_log`) et le test `test_a_row_is_frozen_from_the_auction_day_and_never_rewritten`.

## Vérification de la phase 8 par deux agents indépendants (13 novembre 2026)

**Ma demande :** « Vérifie la phase 8 ». Deux agents ont relu le travail sans rien modifier dans le projet (entraînements et prévisions dans leur dossier de travail) : l'un le code et la règle du gel, l'autre les sorties, l'exploitation et les affirmations du journal. Les agents ne lisent jamais le fichier `.env` et n'utilisent pas le réseau.

**Ce qui était juste :**

- **Aucune fuite.** Les variables des enchères à venir sont construites avec la même jointure stricte que pour le backtest ; une ligne ne peut pas être écrite le jour de l'enchère ou après, tant que l'horloge avance.
- La date de New York est correcte aux changements d'heure (mars et novembre 2026 vérifiés) et autour de minuit.
- Entraîné sur les données d'avant le 1er octobre 2026, le modèle final reproduit au bit près les 980 prévisions du dernier trimestre du walk-forward.
- Les modèles ré-entraînés, le journal et le suivi se reproduisent exactement ; `mean3` et les montants sont justes pour les 4 enchères ; les prévisions restent dans la plage du backtest pour les mêmes maturités.

**Ce qui était faux ou fragile dans la première version :**

| Problème | Conséquence | Correction |
|---|---|---|
| Le gel dépendait seulement de l'heure passée au programme | relancer avec une heure plus ancienne (rejouer une nuit ratée) réécrivait une ligne figée sans erreur ; « ne peut jamais être réécrite » était faux | heure antérieure à la dernière prévision refusée |
| Aucun type fixé pour le journal | le garde-fou bloquait à tort selon la précision des dates ou le fuseau de l'heure fournie | types fixés à l'écriture et à la relecture ; comparaison de toutes les colonnes |
| Soir sans enchère à venir | `--predict` plantait : arrivé 11 à 12 soirs par an en 2023-2024 | journal gardé tel quel |
| Deux exécutions en même temps | la dernière écrasait l'autre | écriture refusée si le journal a changé pendant le calcul ; un seul programme écrit |
| Journal disparu | recréé vide sans alerte | refus, à restaurer depuis git |
| Prévision inchangée | sa date était gardée mais sa valeur réécrite (écarts de 10⁻¹⁵), donc le fichier changeait | ligne gardée telle quelle |
| Suivi | une enchère réalisée sans tail, ou déplacée, restait « en attente » pour toujours | statut fondé sur l'enchère réalisée ; nouvel état `sans_resultat` |
| Versions des modèles | écrites mais jamais vérifiées ; des variables dans un autre ordre étaient acceptées en silence | vérifiées au chargement |
| Fichiers LightGBM | changeaient à chaque entraînement (durées de calcul enregistrées) | seul le modèle retenu est sauvegardé ; pas de ré-entraînement si les données n'ont pas changé |
| Une source en panne | arrêtait tout avant la prévision du soir | la suite tourne sur les dernières données ; l'échec reste signalé |
| Versions minimales | `scikit-learn>=1.5` alors que le code exige 1.7 ; `numpy>=1.26` alors que les modèles sont sauvegardés sous numpy 2 | corrigées ; versions exactes à fixer en phase 10 |
| Tests | 25 des 40 erreurs injectées par l'agent passaient (heure d'été, journal relu depuis le disque, soir sans enchère, horloge en arrière, recette d'entraînement…) | 22 tests ajoutés |

**Affirmations corrigées :** « hyperparamètres choisis par validation croisée » (faux pour la Random Forest) ; « environ 2 Go par an » (1,6 mesuré) ; « −0,08 » pour deux enchères dont une seule déclenche le signal ; le test des 6 745 enchères identiques, juste mais faible.

**Tests :** 35 tests pour la phase 8. J'ai injecté 32 erreurs dans une copie de `live.py`, dont les 25 qui échappaient : 31 sont détectées. La dernière (le garde-fou appelé avec la date de la veille) ne peut pas être détectée sans une seconde erreur : la construction empêche déjà de toucher une ligne figée, et le garde-fou n'est qu'une seconde sécurité. La copie sans erreur passe tous les tests.

**Notes pour la phase 9 (application), tirées de la vérification :**

- seuil du signal (0,08) à définir dans la configuration et à appliquer aux valeurs non arrondies ;
- titre de l'enchère (« 20 ans », « réouverture ») à prendre dans `auctions.parquet`, sans ajouter de colonne au journal figé ; pour les bills, « réouverture » n'apporte rien ;
- part des dealers : la moyenne des 5 fait moins bien que `mean3` sur 2020-2026 pour les notes et bonds (−28,5 %), à afficher avec cet avertissement ;
- tail : il reflète surtout le mouvement du marché du jour (corrélation 0,90 à 0,94 avec la variation du taux), à présenter comme « écart au taux de clôture de la veille » ;
- fréquence et fiabilité passées du signal, d'après le backtest 2014-2026 :
  - notes et bonds : « plus faible » 11 % du temps, avec ensuite un résultat sous `mean3` dans 77 % des cas ;
  - bills : « plus faible » 19 % du temps, 74 % des cas sous `mean3` ensuite.

**Limites connues, non corrigées :** les prévisions de la veille au soir utilisent parfois un VIX et des séries FRED de l'avant-veille (publiés plus tard que la courbe du Trésor). C'est prudent (pas de fuite), mais un peu moins frais que ce que suppose le backtest ; `live_status.json` garde les dates des données utilisées.

**Résultat vérifié :** 241 tests réussis, `ruff check .` propre.

## Phase 9 — Application Streamlit (13 novembre 2026)

**Ma demande :** « OK, phase 9 » : section 11 du plan, une application en 6 pages qui lit les fichiers de `data/output/`, avec un bouton « Mettre à jour maintenant ».

**Ce que le LLM a produit :**

- `app/streamlit_app.py` : la navigation entre les 6 pages (`app/vues/`) et la barre latérale ;
- `src/tresor/app_data.py` : la préparation des tableaux, sans Streamlit, donc testable ;
- `src/tresor/app_ui.py` : lecture des fichiers avec cache, mise en forme des graphiques, barre latérale ;
- `src/tresor/refresh.py` : la mise à jour légère du bouton ;
- `.streamlit/config.toml` : le thème (couleur principale #1F4E79) ;
- 25 tests (`tests/test_app_data.py`, `tests/test_app.py`).

**Les 6 pages :**

| Page | Contenu |
|---|---|
| Prochaines enchères | date, enchère (« 20 ans », « 10 ans, réouverture »), montant (« estimé » si pas encore annoncé), bid-to-cover prévu, moyenne des 3 dernières, écart, signal, part des dealers et tail prévus ; fiabilité passée du signal ; R² de chaque cible ; avertissements sur la part des dealers et le tail |
| Suivi en direct | nombre d'enchères suivies par statut ; prévu contre réalisé (nuage de points et tableau) ; gain cumulé contre `mean3` ; prévisions en attente |
| Backtest | métriques par période et famille avec intervalles ; gain cumulé de chaque modèle ; importance des variables (permutation et SHAP) ; quintiles et stratégie |
| Explorer | historique d'une maturité et d'une cible, avec `mean3` et les prévisions du backtest |
| Qualité des données | état de chaque source (fraîcheur, lignes, alertes), contrôles, rapport de nettoyage |
| Méthodologie | question, sources, variables, validation, résultats, limites, usage des LLM |

**Choix :**

| Point | Décision | Raison |
|---|---|---|
| Signal | seuil de 0,08 défini dans `config.py`, appliqué aux valeurs non arrondies | le 13 semaines (−0,082) déclenche « plus faible », le 20 ans (−0,078) non, alors que les deux s'affichent « −0,08 » |
| Couleurs | #1F4E79 pour l'interface et #C8553D pour l'alerte, comme le plan ; les graphiques à plusieurs séries gardent la palette des figures (#2a78d6, #eb6834) | #1F4E79 ne passe pas le contrôle de lisibilité pour distinguer des séries (phase 6) |
| Nombres | virgule décimale, espace des milliers, vrai signe moins | cohérent avec les figures du rapport |
| Cache | `st.cache_data(ttl=3600)` comme le plan, avec la date de modification du fichier dans la clé | sinon, un fichier réécrit restait invisible jusqu'à une heure |
| Bouton « Mettre à jour maintenant » | télécharge les 4 sources légères, refait la préparation et prévoit avec les modèles sauvegardés, sans réentraîner, **sur une copie des données dans un dossier temporaire** (corrigé après la vérification) ; la prévision est affichée comme **provisoire** et rien n'est écrit dans le dépôt, ni le journal figé | écart au plan, qui prévoyait de mettre à jour les données et de vider le cache : un seul programme écrit les données et le journal (le passage du soir) ; sur Streamlit Cloud, les fichiers seraient de toute façon perdus au redéploiement |
| Modèles absents | le bouton explique que la mise à jour se fait la nuit | c'est le cas sur Streamlit Cloud tant que le stockage des modèles n'est pas décidé (phase 10) |
| Titre des bills | « réouverture » non affiché | pour un bill, cela n'apporte rien au lecteur |

**Vérification :**

- **Tests sans navigateur** (outil de test de Streamlit) :
  - les 6 pages et le programme principal tournent sans erreur sur les données du dépôt ;
  - les changements de sélection fonctionnent sur les pages Explorer et Backtest ;
  - des résultats réalisés simulés s'affichent dans le suivi en direct ;
  - le bouton affiche une prévision provisoire, ou explique l'absence de modèles.
- **Vérification visuelle dans le navigateur, page par page** (écran de 1 440 × 900). Défauts trouvés et corrigés :
  - légendes qui chevauchaient les titres des graphiques ;
  - tableaux coupés (métriques, sources, prochaines enchères) ;
  - modèle affiché par défaut dans le gain cumulé : Ridge au lieu de la moyenne des 5 ;
  - deux graphiques de quintiles côte à côte avec des échelles différentes ;
  - « 1004 » au lieu de « 1 004 » ;
  - dates au format ISO ;
  - noms techniques des sources et motifs d'exclusion sans accents ;
  - texte de l'explorateur qui annonçait un historique « depuis 2009 » au lieu de 2000 ;
  - bouton « Deploy » de Streamlit visible.
- **Le vrai bouton** a été essayé :
  - Il a téléchargé les 4 sources légères et a fait apparaître 10 enchères annoncées le jour même, dont le montant n'était pas encore connu (affiché « estimé »).
  - Le journal figé n'a pas été modifié.
  - J'ai ensuite lancé `--train --predict`, comme le fera le passage de nuit. L'entraînement a été sauté (données d'entraînement inchangées). Le journal compte maintenant 14 enchères (336 lignes), et les 96 lignes d'origine sont restées identiques.

**Correction de la phase 8 :** la phrase « le cas montant pas encore annoncé n'arrive pas en pratique » était fausse. Elle venait d'une observation faite à midi ; le soir même, le calendrier listait 10 enchères sans montant, jusqu'au 25 novembre.

**Non vérifié :** l'affichage sur téléphone, et le fonctionnement sur Streamlit Cloud (phase 10).

**Résultat vérifié :** 266 tests réussis, `ruff check .` propre. Lancement : `streamlit run app/streamlit_app.py` depuis la racine du dépôt.

**À relire moi-même :** les 6 pages dans l'application, et `src/tresor/app_data.py` (fonctions `signal` et `upcoming_table`).

## Vérification de la phase 9 par deux agents indépendants (13 novembre 2026)

**Ma demande :** « Vérifie la phase 9 ». Deux agents ont relu le travail sans rien modifier (copies dans leur dossier de travail, sans réseau, sans cliquer sur le vrai bouton) : l'un le code, le bouton et les tests, l'autre tout ce qui est affiché (chiffres, phrases, français). J'ai moi-même vérifié l'affichage sur téléphone.

**Ce qui était juste :**

- Toutes les valeurs affichées, recalculées par l'agent à partir des fichiers, sont exactes : les 14 lignes de la page 1, les métriques des 3 périodes, les choix de modèles, les quintiles, la stratégie, les sources.
- Le seuil du signal s'applique bien aux valeurs non arrondies.
- Le bouton n'a jamais écrit le journal figé ni le suivi, dans aucun des 5 scénarios de l'agent ; il vérifie les modèles avant tout téléchargement et n'utilise pas FRED.

**Ce qui était faux ou fragile :**

| Problème | Correction |
|---|---|
| Page Backtest et Méthodologie : « même en connaissant le résultat, le gain reste faible : la partie prévisible de la demande n'a pas d'effet » ; le « : » présentait comme une cause deux constats distincts (phase 7) | deux phrases séparées ; la stratégie est décrite avec ses deux côtés (vendeur et acheteur) et l'écart non significatif avec « toujours vendeur » |
| Gain cumulé : « les sauts d'octobre 2015 et 2021 » écrit pour toutes les cibles ; pour le tail et les parts des bills, le saut principal est mai 2023 | phrase seulement pour les bills, avec les trois crises marquées sur le graphique |
| Page 1 : le tail présenté comme la cible la mieux prévue (+22 %) ; les bills sans réserve | le tail est retiré de la liste et commenté à part (+2,8 % contre une constante pour les notes et bonds) ; intervalles affichés ; réserve sur le plafond de la dette |
| Aucune mention « projet étudiant, pas un conseil en investissement », alors que le signal ressemble à une recommandation | mention dans la barre latérale, en page 1 et en méthodologie, avec : « le signal décrit la demande attendue, pas l'évolution des taux » |
| Prévisions à 5-12 jours présentées comme comparables au backtest | phrase ajoutée : seule la dernière prévision, figée le jour de l'enchère, est comparable |
| Fiabilité du signal ambiguë, signal « plus forte » jamais documenté | formulation exacte (« dans 77 % de ces cas, le bid-to-cover réel a bien été inférieur… ») et ligne pour « plus forte » |
| « Modèle de référence fixé à l'avance » : à l'avance de quoi ? | « retenue pour ne pas choisir un modèle d'après ses résultats » |
| Méthodologie : « relues par des relectures indépendantes » | « relue par d'autres agents LLM lancés séparément » |
| Page Qualité : plantage dès qu'un taux hors bornes serait détecté (un nombre traité comme une liste) | corrigé et testé |
| Page Qualité : règle des « 5 jours ouvrés » fausse pour Fiscal Data (30 jours) ; colonne de retard calculée depuis une autre date que celle affichée | seuils lus dans le code de contrôle ; colonnes « dernière observation » et « dernière publication » |
| Page 1 : plantage si la seule enchère à venir n'a pas encore de moyenne des 3 | corrigé et testé |
| Deux clics simultanés : fichiers temporaires partagés, fausses erreurs | le bouton travaille sur une copie dans un dossier temporaire ; verrou (« une mise à jour est déjà en cours ») ; noms de fichiers temporaires uniques partout ; `features.parquet` écrit de façon atomique |
| Le bouton modifiait 11 fichiers suivis par git | plus aucun fichier du dépôt modifié (testé : empreinte de chaque fichier de `data/` avant et après) |
| Tout `FileNotFoundError` affiché comme « modèles absents » | messages distincts : modèles absents, mise à jour déjà en cours, erreur |
| Fichier d'état abîmé : toutes les pages plantaient | lecture protégée |
| Page Suivi : parts en 0,333 au lieu de 33,3 %, gain cumulé en unités brutes | unités de la page 1, gain en % de l'erreur de la moyenne des 3 |
| Tri des enchères alphabétique (« 52 semaines » avant « 6 semaines ») | tri par maturité |
| Seuil du signal sensible aux arrondis des nombres à virgule (2,49 − 2,57 = −0,0799999…) | écart arrondi à 10⁻⁹ avant la comparaison |
| « −0 » dans des intervalles ; « enchère −1 », « Part dealers », traits d'union pour le signe moins ; « mean3 » dans le texte français ; jargon non défini | corrigés ; petit glossaire en méthodologie ; figures du rapport redessinées avec les nouveaux noms |
| Explorer : « depuis 2000 » faux pour les parts (2008) et vieilles réouvertures mal classées en 2008 | historique depuis 2009, l'échantillon des modèles |
| Version minimale de Streamlit (1.37) trop basse pour le code | `streamlit>=1.67`, la version testée |

**Sur téléphone** (vérifié par moi) : les colonnes essentielles (demande attendue, écart, prévision) sont maintenant en premier, et la barre d'outils des graphiques, qui cachait les titres, est masquée.

**Tests :** 44 tests pour l'application (25 avant), dont 4 qui font tourner le vrai bouton sans réseau sur une copie des données. J'ai injecté 43 erreurs dans une copie complète du dépôt, reprenant la plupart des 40 qui échappaient aux tests de l'agent. 41 sont détectées. Les 2 autres sont équivalentes en pratique : le tableau est de toute façon complété ensuite, et le fichier de synthèse commence par la bonne période. Au premier essai, mon outil d'injection restaurait le code mais pas les données de la copie : une erreur qui écrivait le journal faussait les essais suivants. Je l'ai corrigé, puis j'ai tout relancé ; la copie sans erreur passe tous les tests.

**Limites connues :** si les serveurs du Trésor ne répondent pas, le bouton peut attendre plusieurs minutes (délai réduit à 20 secondes par requête, mais avec des nouvelles tentatives). Le fonctionnement sur Streamlit Cloud reste à vérifier en phase 10.

**Résultat vérifié :** 285 tests réussis, `ruff check .` propre.

## Phase 10 — Automatisation et mise en ligne (13 et 14 novembre 2026)

**Ma demande :** « OK, phase 10 » : section 12 du plan. Un workflow GitHub Actions qui met à jour les données chaque soir, et l'application en ligne sur Streamlit Community Cloud.

**Ce que le LLM a produit :**

- `requirements.txt` et `requirements-dev.txt` : versions exactes, celles testées ici (Python 3.11). Le LLM a vérifié qu'elles s'installent sur Linux. Au premier essai, LightGBM 4.7 semblait introuvable : la vérification ne regardait que l'ancienne étiquette des paquets Linux (`manylinux2014`), alors que LightGBM 4.x n'est publié que pour les Linux récents (`manylinux_2_28`), ceux de GitHub et de Streamlit.
- `.github/workflows/update.yml` : le passage du soir.
- `.github/workflows/tests.yml` : `ruff` et les tests à chaque envoi de code.
- `tresor/check_log.py` : refuse d'enregistrer les données si une ligne figée du journal a changé ou disparu.
- `tresor/models_store.py` : archive des modèles déposée dans une « release » GitHub et téléchargée par l'application en ligne (extraction protégée contre les chemins piégés).
- 10 tests (295 en tout).

**Le passage du soir**, du lundi au vendredi à 23 h 17 UTC (18 h 17 ou 19 h 17 à New York) :

1. installe les versions figées ;
2. garde une copie du journal ;
3. lance `--fetch --clean --train --predict` ; une source en panne n'arrête pas la suite (règle changée après la vérification, voir plus bas) ;
4. vérifie qu'aucune ligne figée n'a changé ;
5. commite les données s'il y a du nouveau ;
6. dépose les modèles dans la release `modeles-du-soir` (l'archive est remplacée chaque soir : l'historique ne grossit pas) ;
7. signale l'échec du pipeline s'il y en a eu un.

Un seul passage à la fois (`concurrency`). La clé FRED est le secret `FRED_API_KEY` du dépôt, saisi par moi dans GitHub ; elle n'est jamais dans le code (vérifié sur tout l'historique avant la publication).

**Choix :**

| Point | Décision | Raison |
|---|---|---|
| Ce que lance le soir | `--fetch --clean --train --predict`, pas `--all` | écart à la section 12 du plan : `--all` comprend le backtest (environ 2 heures) |
| Une source en panne | la suite tourne sur les dernières données, les données sont enregistrées, et le workflow finit en échec pour que je sois prévenu (changé après la vérification : avertissement seulement, voir plus bas) | une panne d'un soir ne doit pas priver une enchère de sa prévision |
| Modèles | dans une release GitHub, pas dans git | commités chaque soir, ils feraient grossir le dépôt d'environ 1,6 Go par an |
| Dépôt | public, `make4this/tresor-auctions`, branche `main` | choix fait par moi : le professeur peut lire le code et ce journal |
| Adresse des commits | réécrite par le LLM avant la publication, après m'avoir expliqué pourquoi : l'adresse privée de mon compte GitHub remplace `pablogreco@…`, qui aurait pu attribuer mes commits à un autre compte | contenu des 14 commits vérifié identique ; une sauvegarde locale de l'ancien historique est gardée (branche locale, jamais envoyée sur GitHub) |
| Outil `gh` | téléchargé par le LLM, après mon accord, depuis la page officielle de GitHub dans `~/outils/gh`, empreinte vérifiée avec la liste publiée, sans installation dans le système | Homebrew et `gh` étaient absents ; connexion par le navigateur avec un code à usage unique, saisi par moi |
| Secret Streamlit | aucun | écart à la section 12 du plan (« clé dans les secrets Streamlit ») : l'application n'appelle jamais FRED, le bouton n'utilise que TreasuryDirect et Treasury.gov |

**Problèmes rencontrés :**

- la première commande de connexion attendait une réponse dans un terminal que je n'ai pas trouvé ; le LLM l'a relancée en affichant directement le code à saisir dans le navigateur ;
- le premier envoi du code a échoué (erreur HTTP 400 sur un envoi de 28 Mo) ; il est passé en HTTP/1.1 avec un tampon plus grand.

**Mise en ligne et vérification :**

- **Tests sur GitHub** (Linux, versions figées) : style et 295 tests réussis.
- **Passage du soir lancé à la main** le 14 novembre :
  - toutes les étapes ont tourné, dans cet ordre : journal figé vérifié, données enregistrées par le robot (commit « Données du 2026-11-14 »), modèles publiés (8,6 Mo) ;
  - le workflow a fini en échec, comme prévu, parce que l'API Fiscal Data du Trésor répondait « 503, service indisponible » pour deux sources (adjudications, trésorerie), après 5 essais chacune ; une troisième (dette) n'a répondu qu'au 5e essai. Le même essai depuis mon ordinateur donnait aussi 503 : c'était une panne du Trésor, pas un blocage de GitHub ;
  - le premier critère de la phase (« le workflow lancé à la main réussit ») n'est donc pas encore rempli. Le second (« l'adresse publique s'ouvre depuis un autre appareil ») reste à vérifier par moi.
- **Application en ligne** (https://tresor-auctions.streamlit.app), déployée par moi sur Streamlit Community Cloud avec Python 3.11 :
  - les 6 pages s'affichent avec les données du passage du soir (vérifié par le LLM dans le navigateur) ;
  - le bouton « Mettre à jour maintenant » fonctionne en ligne : il télécharge les modèles depuis la release et les sources légères, puis affiche une prévision provisoire.

**Limites connues :**

- Une application Streamlit gratuite se met en veille sans visite ; le premier visiteur doit la réveiller.
- GitHub peut suspendre les workflows programmés d'un dépôt sans activité pendant 60 jours.
- Le passage du soir réentraîne les modèles à chaque fois (environ 1 minute), car les modèles ne sont pas gardés d'un passage à l'autre sur GitHub.

**À relire moi-même :** `.github/workflows/update.yml` et l'onglet Actions du dépôt après le passage de lundi soir.

## Vérification de la phase 10 par deux agents indépendants (14 novembre 2026)

**Ma demande :** « Vérifie la phase 10 ». Deux agents ont relu le travail sans rien modifier : l'un les workflows, la release des modèles, les versions et la sécurité (lecture seule sur GitHub, essais sur une copie du dépôt) ; l'autre ce qui est publié (dépôt, README, journal, application, résultat des passages).

**Ce qui était juste :**

- Aucun secret publié : ni la clé FRED ni aucun jeton dans les 17 commits publics, les données (colonnes texte des Parquet comprises) ou les journaux des 3 passages GitHub, où la clé apparaît masquée (« *** »).
- Le journal figé a été préservé par le passage du 14 novembre (contrôle refait par l'agent sur les deux versions du fichier).
- Le passage a fait toutes ses étapes, et le robot n'a modifié que `data/` (12 fichiers).
- La réécriture de l'adresse des commits n'a changé aucun contenu (les 14 arbres comparés un à un), et les commits sont attribués à mon compte.
- L'extraction de l'archive refusait déjà les liens symboliques qui sortent du dossier.
- Le dépôt grossit d'environ 0,3 à 1 Mo par soir, soit 0,1 à 0,25 Go par an : acceptable.

**Ce qui était faux ou fragile :**

| Problème | Correction |
|---|---|
| Une erreur bloquante (nettoyage, variables, entraînement) était traitée comme une panne de source : le workflow enregistrait des données à moitié à jour | code de sortie 3 pour « des sources ont échoué, tout le reste a tourné », 1 pour une erreur bloquante ; avec 1, rien n'est enregistré (testé : une erreur imprévue au milieu du pipeline arrête tout) |
| Rouge presque chaque soir : une source sur 11 qui échoue suffisait, après seulement 15 à 20 secondes de nouvelles tentatives ; le rouge ne voulait plus rien dire | nouvel essai 10 minutes plus tard ; une panne de source donne un avertissement, pas un échec ; rouge seulement pour une erreur bloquante, un dépôt des modèles raté ou un contrôle qualité « rouge » |
| Panne invisible dans l'application : page Qualité toute verte le soir du 14 | `data/output/fetch_status.json` : résultat de chaque source, date de sa dernière réussite et erreur (clé d'API masquée, testé) ; avertissement en page Qualité et dans la barre latérale |
| Le contrôle du journal ne voyait pas une ligne ajoutée après coup pour une enchère passée | refusée ; toute prévision datée du jour de son enchère ou après (date de New York) est aussi refusée |
| Rien ne contrôlait un envoi fait à la main | nouveau workflow « Contrôle du journal figé » sur chaque envoi qui touche au journal ; il signale après coup, il n'empêche pas l'envoi |
| Un passage à cheval sur minuit à New York aurait été refusé à tort (prévision et contrôle lisaient l'heure séparément) | le contrôle prend la date du passage qui vient d'écrire le journal (`live_status.json`, s'il a moins de 6 heures) |
| 12 versions figées sur 58 bibliothèques installées | les 55 bibliothèques (60 avec les outils de test), résolues pour Linux et Python 3.11 ; installation sans résolution (`--no-deps`) puis `pip check`. La résolution faite sur Mac oubliait `watchdog`, que Streamlit n'exige que sous Linux : ajouté, et chaque version vérifiée disponible pour Linux |
| Bouton en ligne : 8,6 Mo retéléchargés à chaque clic, sans vérifier l'empreinte ; pendant le remplacement de l'archive, ou toute une journée si son dépôt échouait, l'application ne trouvait plus de modèles | archive nommée d'après son empreinte SHA-256 et annoncée par `data/output/models_release.json`, commité seulement après le dépôt ; l'application vérifie l'empreinte et garde les modèles (un seul téléchargement par version) ; l'archive de la veille reste disponible |
| Limites de taille faibles : archive lue en entier avant le contrôle, aucune limite une fois extraite (50 Mo compressés peuvent donner 50 Go) | téléchargement en flux coupé à 50 Mo ; 200 Mo au plus une fois extraite ; seulement des fichiers ordinaires, sans sous-dossier |
| Au passage d'une année, le bouton aurait écrit l'année écoulée dans le cache des taux du dépôt | ce cache est lui aussi copié dans le dossier temporaire (testé) |
| « Modèles absents ici » affiché pour toute erreur, réseau compris | « Modèles indisponibles ici pour le moment » |
| Jeton d'écriture accessible à toutes les étapes, y compris l'installation de 60 paquets ; actions désignées par une étiquette déplaçable, en Node 20 (abandonné) | `persist-credentials: false` : le jeton n'est donné qu'aux étapes d'envoi et de dépôt ; actions figées par l'empreinte de leur commit, checkout v7.0.1 et setup-python v7.1.0 (Node 24) ; `contents: read` pour les tests |
| Un lancement à la main depuis une autre branche aurait remplacé les modèles en ligne | passage seulement sur `main` |
| Un envoi refusé (quelqu'un a poussé entre-temps) perdait la soirée | jusqu'à 3 essais, avec `git pull --rebase` et un nouveau contrôle du journal à chaque fois |
| Premier passage impossible sans journal existant | copie du journal seulement s'il existe |
| 10 erreurs injectées par l'agent passaient les tests : date UTC au lieu de New York, tolérance de 10⁻⁶, lien symbolique, branche de téléchargement jamais testée, erreur au milieu du pipeline… | un test pour chacune |
| README : installation propre au Mac, sans les versions figées ; aucun résultat ; usage des LLM en une phrase, à la fin | installation pour tous les systèmes avec les versions figées ; section « Résultats », chaque chiffre avec son fichier ; plan du dépôt ; usage des LLM en tête (code écrit par Claude Code sous ma direction, `PLAN.md` est la consigne, deux agents relisent chaque phase) ; sources des données et mention exigée par FRED |
| Application sans lien vers le dépôt ; version de Python sur Streamlit consignée nulle part | liens vers le dépôt, le journal et le plan ; versions de Python et des bibliothèques affichées en méthodologie ; Python 3.11 noté dans le README |
| Journal : 8,5 Mo au lieu de 8,6 ; troisième source en difficulté (dette) omise ; qui a fait quoi pas clair (`gh`, adresse des commits, vérification des pages) ; secret Streamlit du plan non mentionné ; second critère de la phase oublié | corrigés dans la section de la phase 10 |
| Commentaires périmés (« environ 3 Mo », « sera décidé en phase 10 ») ; `.streamlit/secrets.toml` et `data/models.old/` non ignorés par git | corrigés |
| La release apparaît comme la dernière version du dépôt, avec des archives « Source code » d'un vieux commit | ses notes l'expliquent (mises à jour par le passage du soir) |

**Tests :** 328 tests (295 avant). J'ai injecté 24 erreurs dans une copie du dépôt, sur le code nouveau ou modifié. 21 ont été détectées au premier essai. Pour les 3 autres, j'ai ajouté un test chacune : un statut daté du futur accepté, un rapport de téléchargement jamais écrit par le pipeline, et la limite de téléchargement retirée (le test confondait deux messages). Les 24 erreurs sont maintenant détectées, et la copie sans erreur passe tous les tests.

**Limites connues :**

- Le contrôle d'un envoi fait à la main signale le problème, il ne l'empêche pas. Seule une protection de la branche dans les réglages de GitHub l'empêcherait ; je ne l'ai pas activée.
- Les modèles sont réentraînés chaque soir. Si les fichiers diffèrent d'un soir à l'autre, l'application les retélécharge une fois par soir (et non plus à chaque clic).
- À vérifier après quelques passages : la courbe des taux du jour doit être publiée avant 18 h 17 à New York, c'est-à-dire `market_last_dates.yield_10y` égal à la date du passage dans `live_status.json`. Sinon, la prévision utilise la courbe de la veille : c'est un retard, pas une fuite.
