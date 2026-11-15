# Chiffres clés

Tous les chiffres du rapport et de l'oral, recalculés depuis les fichiers de `data/` par `python reports/make_figures.py`. La commande `python reports/make_figures.py --check` vérifie qu'ils correspondent toujours aux fichiers. Le titre de chaque partie donne l'étape du pipeline qui produit ses fichiers.

Les chiffres marqués « du jour » changent chaque soir avec le passage automatique. Les chiffres marqués « figé » ne changent que si l'on relance `python -m tresor.pipeline --backtest --strategy`.

Données utilisées : contrôle qualité du 14/11/2026 (date de New York), dernière prévision le 14/11/2026 à 20:31 UTC.

**Petit glossaire.**

- **Modèle de référence** : la moyenne des prévisions des 5 modèles (moyenne_5), retenue par principe et non d'après ses résultats. Le meilleur modèle individuel est choisi sur 2014-2019 ; tous les modèles sont jugés sur 2020-2026.
- **mean3** : la moyenne des 3 dernières enchères de même maturité, le benchmark de référence.
- **R² contre mean3** : 1 − erreur quadratique du modèle / erreur quadratique de mean3, en %. Au-dessus de 0, le modèle fait mieux que mean3.
- **Constante** : la moyenne de la cible sur la période d'entraînement.
- **Tail** : taux obtenu à l'enchère moins le taux de marché de la veille, en points de base (pb, 0,01 point de pourcentage).
- **Variation du taux le jour de l'enchère** : de la clôture de la veille à celle du jour de l'enchère, pour la maturité de l'enchère.
- **Trade** : une position sur cette même journée, vendeuse ou acheteuse de duration.
- **Sharpe** : gain moyen rapporté à sa variabilité, sur un an.
- Intervalles à 90 %, par bootstrap d'années entières. « j » : jours calendaires.

## Données (`--fetch`, `--clean`)

| Chiffre | Valeur | Nature | Fichier | Calcul |
|---|---|---|---|---|
| Tables brutes collectées (Trésor américain, Fed de New York, FRED, CFTC, CBOE) | 11 | du jour | [data_quality.json](../data/output/data_quality.json) | nombre d'entrées de sources |
| Enchères reçues de Fiscal Data (depuis 2000) | 7 905 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.rows_in.fiscaldata |
| Résultats récents reçus de TreasuryDirect | 137 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.rows_in.treasurydirect |
| Enchères annoncées reçues de TreasuryDirect (calendrier) | 16 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.rows_in.treasurydirect_upcoming |
| Enchères présentes dans Fiscal Data et TreasuryDirect, fusionnées | 108 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.duplicates_removed |
| Enchères exclues au total | 13 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | somme des auctions.excluded.*.rows |
| dont petites émissions hors calendrier (moins de 1 Md$) | 12 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.excluded.petite_emission_hors_calendrier.rows |
| Enchères gardées après nettoyage, annoncées comprises | 7 937 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.rows_out |
| dont enchères annoncées, pas encore réalisées | 16 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.rows_out_by_status.upcoming |
| dont bills | 5 109 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.rows_out_by_family.bill |
| dont notes et bonds | 1 650 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.rows_out_by_family.nominal_coupon |
| dont TIPS, FRN et CMB (non modélisés) | 1 178 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | somme des autres familles de auctions.rows_out_by_family |
| Échantillon des modèles : bills, notes et bonds réalisés depuis 2009 | 5 119 | du jour | [features.parquet](../data/clean/features.parquet) | lignes de model_sample (status=completed, depuis 2009), family ∈ [bill, nominal_coupon] |
| dont bills dans l'échantillon des modèles | 3 757 | du jour | [features.parquet](../data/clean/features.parquet) | même calcul, family=bill |
| dont notes et bonds dans l'échantillon des modèles | 1 362 | du jour | [features.parquet](../data/clean/features.parquet) | même calcul, family=nominal_coupon |
| Variables explicatives point-in-time de chaque modèle | 46 | figé | [feature_importance.csv](../data/output/feature_importance.csv) | variables distinctes (le même nombre pour chaque famille, cible et modèle) |
| dont variables des nouvelles sources (dealers, CFTC, liquidité) | 9 | figé | [feature_importance.csv](../data/output/feature_importance.csv) | variables distinctes dont group ∈ ['liquidity', 'positioning'] |

## Délais de publication (`--clean`)

| Chiffre | Valeur | Nature | Fichier | Calcul |
|---|---|---|---|---|
| Délai médian de publication, Treasury.gov : courbe des taux | 0 j (9 sur 10 : 0 j) | du jour | [market.parquet](../data/clean/market.parquet) | available_at − date en jours, depuis 2014 ; médiane (9e décile) |
| Délai médian de publication, CBOE : VIX | 0 j (9 sur 10 : 0 j) | du jour | [market.parquet](../data/clean/market.parquet) | available_at − date en jours, depuis 2014 ; médiane (9e décile) |
| Délai médian de publication, Fiscal Data : trésorerie | 1 j (9 sur 10 : 3 j) | du jour | [market.parquet](../data/clean/market.parquet) | available_at − date en jours, depuis 2014 ; médiane (9e décile) |
| Délai médian de publication, Fiscal Data : dette | 2 j (9 sur 10 : 4 j) | du jour | [market.parquet](../data/clean/market.parquet) | available_at − date en jours, depuis 2014 ; médiane (9e décile) |
| Délai médian de publication, CFTC : positions sur les futures | 3 j (9 sur 10 : 6 j) | du jour | [market.parquet](../data/clean/market.parquet) | available_at − date en jours, depuis 2014 ; médiane (9e décile) |
| Délai médian de publication, New York Fed : positions des dealers | 8 j (9 sur 10 : 8 j) | du jour | [market.parquet](../data/clean/market.parquet) | available_at − date en jours, depuis 2014 ; médiane (9e décile) |
| Délai médian de publication, FRED : autres séries | 1 j (9 sur 10 : 3 j) | du jour | [market.parquet](../data/clean/market.parquet) | available_at − date en jours, depuis 2014 ; médiane (9e décile) |
| Délai médian de publication, FRED : indice dollar | 6 j (9 sur 10 : 1 168 j) | du jour | [market.parquet](../data/clean/market.parquet) | available_at − date en jours, depuis 2014 ; médiane (9e décile) |

## Validation walk-forward (`--backtest`)

| Chiffre | Valeur | Nature | Fichier | Calcul |
|---|---|---|---|---|
| Enchères évaluées 2014-2026, bills (bid-to-cover) | 2 906 | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2014-2026, family=bill, target=btc ; n |
| Enchères évaluées 2014-2026, notes et bonds (bid-to-cover) | 1 004 | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2014-2026, family=nominal_coupon, target=btc ; n |
| Enchères évaluées 2020-2026, bills (bid-to-cover) | 1 825 | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=bill, target=btc ; n |
| Enchères évaluées 2020-2026, notes et bonds (bid-to-cover) | 572 | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=nominal_coupon, target=btc ; n |
| R² contre mean3, notes et bonds, bid-to-cover, 2020-2026 | +9,6 % [5,7 ; 14,5] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=nominal_coupon, target=btc ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, notes et bonds, part des dealers, 2020-2026 | −28,5 % [−45,9 ; −14,7] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=nominal_coupon, target=dealer_share ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, notes et bonds, part des indirects, 2020-2026 | +5,7 % [0,3 ; 10,5] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=nominal_coupon, target=indirect_share ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, notes et bonds, tail, 2020-2026 | +22,1 % [16,3 ; 27,8] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=nominal_coupon, target=tail_bp ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, bills, bid-to-cover, 2020-2026 | +21,2 % [13,3 ; 26,2] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=bill, target=btc ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, bills, part des dealers, 2020-2026 | +1,5 % [−4,0 ; 5,7] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=bill, target=dealer_share ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, bills, part des indirects, 2020-2026 | +6,0 % [1,5 ; 10,4] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=bill, target=indirect_share ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, bills, tail, 2020-2026 | +22,3 % [−6,2 ; 31,1] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=bill, target=tail_bp ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, notes et bonds, bid-to-cover, 2014-2026 (fin de la courbe de gain cumulé) | +11,6 % [7,9 ; 15,4] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2014-2026, family=nominal_coupon, target=btc ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, bills, bid-to-cover, 2014-2026 (fin de la courbe de gain cumulé) | +20,3 % [16,4 ; 23,8] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2014-2026, family=bill, target=btc ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| Bonne direction, notes et bonds, bid-to-cover, 2020-2026 : part des enchères où le modèle dit correctement si le bid-to-cover sera au-dessus ou au-dessous de mean3 | 60 % | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=nominal_coupon, target=btc ; direction |
| R² contre une constante, notes et bonds, tail, 2020-2026 | +2,8 % | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=nominal_coupon, target=tail_bp ; r2_vs_constant |
| R² contre mean3 sans le meilleur trimestre, bills, bid-to-cover, 2020-2026 | +17,1 % | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026, family=bill, target=btc ; r2_without_best_quarter |
| R² du benchmark naïf (dernière enchère) contre mean3, notes et bonds, bid-to-cover, 2020-2026 | −54,2 % | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=naive, feature_set=toutes, period=2020-2026, family=nominal_coupon, target=btc ; r2_vs_mean3 |
| Apport des nouvelles sources au R², notes et bonds, bid-to-cover, 2014-2026 | −0,6 point [−2,1 ; +0,9] | figé | [new_sources_comparison.csv](../data/output/new_sources_comparison.csv) | family=nominal_coupon, target=btc, model=moyenne_5 ; gain [gain_ci_low ; gain_ci_high], en points de pourcentage |
| Cibles où les nouvelles sources dégradent la prévision (intervalle entièrement négatif), moyenne des 5, 2014-2026 | 4 sur 8 | figé | [new_sources_comparison.csv](../data/output/new_sources_comparison.csv) | model=moyenne_5 ; gain_ci_high < 0 |
| Cibles où les nouvelles sources améliorent la prévision (intervalle entièrement positif), moyenne des 5, 2014-2026 | 0 sur 8 | figé | [new_sources_comparison.csv](../data/output/new_sources_comparison.csv) | model=moyenne_5 ; gain_ci_low > 0 |
| Couples modèle-cible où les nouvelles sources dégradent la prévision, tous modèles, 2014-2026 | 17 sur 48 | figé | [new_sources_comparison.csv](../data/output/new_sources_comparison.csv) | toutes les lignes ; gain_ci_high < 0 |

## Lien avec le marché, notes et bonds (`--strategy`)

| Chiffre | Valeur | Nature | Fichier | Calcul |
|---|---|---|---|---|
| Enchères de l'analyse par quintiles, 2009-2026 | 1 360 | figé | [quintiles.csv](../data/output/quintiles.csv) | period=2009-2026, surprise=bid_to_cover ; somme de n |
| Variation du taux le jour de l'enchère, 20 % des enchères où la part des dealers dépasse le plus la moyenne des 3 dernières (demande faible), 2009-2026 | +1,8 pb [+1,2 ; +2,5] | figé | [quintiles.csv](../data/output/quintiles.csv) | period=2009-2026, surprise=part_des_dealers, quintile=5 ; rate_change_mean_bp [ci_low ; ci_high] |
| Variation du taux le jour de l'enchère, 20 % des enchères où la part des dealers est le plus en dessous de la moyenne des 3 dernières (demande forte), 2009-2026 | −0,7 pb [−1,1 ; −0,3] | figé | [quintiles.csv](../data/output/quintiles.csv) | period=2009-2026, surprise=part_des_dealers, quintile=1 ; rate_change_mean_bp [ci_low ; ci_high] |
| Variation du taux le jour de l'enchère, 20 % des enchères où la part des dealers dépasse le plus la moyenne des 3 dernières (demande faible), 2009-2013 | +3,1 pb [+1,5 ; +4,7] | figé | [quintiles.csv](../data/output/quintiles.csv) | period=2009-2013, surprise=part_des_dealers, quintile=5 ; rate_change_mean_bp [ci_low ; ci_high] |
| Variation du taux le jour de l'enchère, 20 % des enchères où la part des dealers dépasse le plus la moyenne des 3 dernières (demande faible), 2014-2019 | +1,4 pb [+0,3 ; +2,5] | figé | [quintiles.csv](../data/output/quintiles.csv) | period=2014-2019, surprise=part_des_dealers, quintile=5 ; rate_change_mean_bp [ci_low ; ci_high] |
| Variation du taux le jour de l'enchère, 20 % des enchères où la part des dealers dépasse le plus la moyenne des 3 dernières (demande faible), 2020-2026 | +1,4 pb [+0,1 ; +2,5] | figé | [quintiles.csv](../data/output/quintiles.csv) | period=2020-2026, surprise=part_des_dealers, quintile=5 ; rate_change_mean_bp [ci_low ; ci_high] |
| Variation du taux le jour de l'enchère, 20 % des enchères dont le bid-to-cover est le plus en dessous de la moyenne des 3 dernières (demande faible), 2009-2026 | +2,0 pb [+1,2 ; +2,8] | figé | [quintiles.csv](../data/output/quintiles.csv) | period=2009-2026, surprise=bid_to_cover, quintile=5 ; rate_change_mean_bp [ci_low ; ci_high] |
| Variation du taux le jour de l'enchère, 20 % des enchères dont le bid-to-cover est le plus au-dessus de la moyenne des 3 dernières (demande forte), 2009-2026 | −1,4 pb [−1,7 ; −1,0] | figé | [quintiles.csv](../data/output/quintiles.csv) | period=2009-2026, surprise=bid_to_cover, quintile=1 ; rate_change_mean_bp [ci_low ; ci_high] |

## Stratégie illustrative, notes et bonds (`--strategy`)

| Chiffre | Valeur | Nature | Fichier | Calcul |
|---|---|---|---|---|
| Trades (une position par enchère de notes et bonds), stratégie, 2020-2026 | 572 | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=strategie ; trades |
| Gain moyen par trade après coût, stratégie, 2020-2026 | −1,40 pb [−1,91 ; −0,86] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=strategie ; mean_bp [mean_ci_low ; mean_ci_high] |
| Gain moyen par trade après coût, toujours vendeur, 2020-2026 | −1,27 pb [−2,08 ; −0,36] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=toujours_vendeur ; mean_bp [mean_ci_low ; mean_ci_high] |
| Écart du gain moyen avec toujours vendeur, stratégie, 2020-2026 | −0,13 pb [−0,84 ; +0,58] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=strategie ; vs_vendeur_bp [vs_vendeur_ci_low ; vs_vendeur_ci_high] |
| Gain moyen par trade après coût, information parfaite sur le signe de la surprise de bid-to-cover (réel − mean3), impossible en pratique, 2020-2026 | +0,52 pb [+0,13 ; +0,87] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=information_parfaite ; mean_bp [mean_ci_low ; mean_ci_high] |
| Taux de réussite après coût (un taux inchangé compte comme une perte), stratégie, 2020-2026 | 40 % | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=strategie ; hit_rate |
| Sharpe annualisé, stratégie, 2020-2026 | −2,17 | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=strategie ; sharpe |
| Trades (une position par enchère de notes et bonds), stratégie, 2014-2026 | 1 004 | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2014-2026, rule=strategie ; trades |
| Gain moyen par trade après coût, stratégie, 2014-2026 | −1,00 pb [−1,41 ; −0,59] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2014-2026, rule=strategie ; mean_bp [mean_ci_low ; mean_ci_high] |
| Gain moyen par trade après coût, toujours vendeur, 2014-2026 | −0,70 pb [−1,27 ; −0,13] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2014-2026, rule=toujours_vendeur ; mean_bp [mean_ci_low ; mean_ci_high] |
| Écart du gain moyen avec toujours vendeur, stratégie, 2014-2026 | −0,30 pb [−0,77 ; +0,13] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2014-2026, rule=strategie ; vs_vendeur_bp [vs_vendeur_ci_low ; vs_vendeur_ci_high] |
| Gain moyen par trade après coût, information parfaite sur le signe de la surprise de bid-to-cover (réel − mean3), impossible en pratique, 2014-2026 | +0,42 pb [+0,15 ; +0,66] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2014-2026, rule=information_parfaite ; mean_bp [mean_ci_low ; mean_ci_high] |
| Taux de réussite après coût (un taux inchangé compte comme une perte), stratégie, 2014-2026 | 43 % | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2014-2026, rule=strategie ; hit_rate |
| Sharpe annualisé, stratégie, 2014-2026 | −1,74 | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2014-2026, rule=strategie ; sharpe |
| Gain total après coût, stratégie, 2014-2026 | −1 009 pb | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2014-2026, rule=strategie ; total_bp |
| Gain total après coût, toujours vendeur, 2014-2026 | −707 pb | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2014-2026, rule=toujours_vendeur ; total_bp |
| Gain total après coût, information parfaite sur le signe de la surprise de bid-to-cover (réel − mean3), impossible en pratique, 2014-2026 | +423 pb | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2014-2026, rule=information_parfaite ; total_bp |
| Corrélation entre la surprise réelle de bid-to-cover et la variation du taux, 2014-2026 | −0,22 | figé | [strategy_diagnostics.csv](../data/output/strategy_diagnostics.csv) | name=corr_surprise_move ; value |
| Corrélation entre la partie prévue du bid-to-cover et la variation du taux, 2014-2026 | +0,10 | figé | [strategy_diagnostics.csv](../data/output/strategy_diagnostics.csv) | name=corr_predicted_move ; value |
| Même corrélation, en écart à la moyenne de chaque année (sans l'effet de période), 2014-2026 | +0,04 | figé | [strategy_diagnostics.csv](../data/output/strategy_diagnostics.csv) | name=corr_predicted_move_within_year ; value |

## Prévisions en direct (`--predict`)

| Chiffre | Valeur | Nature | Fichier | Calcul |
|---|---|---|---|---|
| Enchères à venir prévues au dernier passage (bills, notes et bonds) | 14 | du jour | [live_status.json](../data/output/live_status.json) | upcoming_auctions |
| Lignes du journal des prévisions | 336 | du jour | [predictions_log.parquet](../data/output/predictions_log.parquet) | nombre de lignes |
| Enchères du journal déjà figées (jour de l'enchère atteint) | 0 | du jour | [predictions_log.parquet](../data/output/predictions_log.parquet) | enchères distinctes dont auction_date ≤ today_new_york de live_status.json |
| Enchères du journal dont le résultat est publié | 0 | du jour | [live_tracking.parquet](../data/output/live_tracking.parquet) | enchères distinctes dont status=realisee |
