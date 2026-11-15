# Chiffres clés

Tous les chiffres du rapport et de l'oral, recalculés depuis les fichiers de `data/output/` par `python reports/make_figures.py`. La commande `python reports/make_figures.py --check` vérifie qu'ils correspondent toujours aux fichiers.

Données du 14/11/2026 (date de New York). Les chiffres « du jour » changent chaque soir avec le passage automatique ; les chiffres « figé » ne changent que si l'on relance `python -m tresor.pipeline --backtest --strategy`.

Modèle de référence : la moyenne des 5 modèles (moyenne_5), fixée avant de voir les résultats. R² contre mean3 : 1 − erreur quadratique du modèle / erreur quadratique de la moyenne des 3 dernières enchères de même maturité. Choix sur 2014-2019, jugement sur 2020-2026. Intervalles à 90 %, par bootstrap d'années entières. pb : point de base.

## Données

| Chiffre | Valeur | Nature | Fichier | Calcul |
|---|---|---|---|---|
| Sources publiques collectées | 11 | du jour | [data_quality.json](../data/output/data_quality.json) | nombre d'entrées de sources |
| Enchères reçues de Fiscal Data (depuis 2000) | 7 905 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.rows_in.fiscaldata |
| Enchères présentes dans Fiscal Data et TreasuryDirect, fusionnées | 108 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.duplicates_removed |
| Petites émissions hors calendrier exclues (moins de 1 Md$) | 12 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.excluded.petite_emission_hors_calendrier.rows |
| Enchères gardées après nettoyage | 7 937 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.rows_out |
| dont bills | 5 109 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.rows_out_by_family.bill |
| dont notes et bonds | 1 650 | du jour | [cleaning_report.json](../data/output/cleaning_report.json) | auctions.rows_out_by_family.nominal_coupon |
| Variables explicatives point-in-time | 46 | figé | [features_missing.csv](../data/output/features_missing.csv) | lignes dont groupe ≠ cible |
| dont variables des nouvelles sources (dealers, CFTC, liquidité) | 9 | figé | [features_missing.csv](../data/output/features_missing.csv) | lignes dont groupe ∈ ['liquidity', 'positioning'] |

## Validation walk-forward

| Chiffre | Valeur | Nature | Fichier | Calcul |
|---|---|---|---|---|
| Enchères évaluées 2014-2026, bills | 2 906 | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, period=2014-2026, target=btc ; n |
| Enchères évaluées 2014-2026, notes et bonds | 1 004 | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, period=2014-2026, target=btc ; n |
| Enchères évaluées 2020-2026, bills | 1 825 | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, period=2020-2026, target=btc ; n |
| Enchères évaluées 2020-2026, notes et bonds | 572 | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, period=2020-2026, target=btc ; n |
| R² contre mean3, notes et bonds, bid-to-cover, 2020-2026 | 0,10 [0,06 ; 0,15] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026 ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, notes et bonds, part des dealers, 2020-2026 | −0,29 [−0,46 ; −0,15] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026 ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, notes et bonds, part des indirects, 2020-2026 | 0,06 [0,00 ; 0,11] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026 ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, notes et bonds, tail, 2020-2026 | 0,22 [0,16 ; 0,28] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026 ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, bills, bid-to-cover, 2020-2026 | 0,21 [0,13 ; 0,26] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026 ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, bills, part des dealers, 2020-2026 | 0,01 [−0,04 ; 0,06] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026 ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, bills, part des indirects, 2020-2026 | 0,06 [0,02 ; 0,10] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026 ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| R² contre mean3, bills, tail, 2020-2026 | 0,22 [−0,06 ; 0,31] | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, feature_set=toutes, period=2020-2026 ; r2_vs_mean3 [r2_ci_low ; r2_ci_high] |
| Bonne direction contre mean3, notes et bonds, bid-to-cover, 2020-2026 | 60 % | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, period=2020-2026 ; direction |
| R² contre une constante, notes et bonds, tail, 2020-2026 | 0,03 | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, period=2020-2026 ; r2_vs_constant |
| R² contre mean3 sans le meilleur trimestre, bills, bid-to-cover, 2020-2026 | 0,17 | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=moyenne_5, period=2020-2026 ; r2_without_best_quarter |
| R² du benchmark naive contre mean3, notes et bonds, bid-to-cover, 2020-2026 | −0,54 | figé | [backtest_metrics.csv](../data/output/backtest_metrics.csv) | model=naive, period=2020-2026 ; r2_vs_mean3 |
| Apport des nouvelles sources au R², notes et bonds, bid-to-cover, 2014-2026 | −0,006 [−0,021 ; +0,009] | figé | [new_sources_comparison.csv](../data/output/new_sources_comparison.csv) | family=nominal_coupon, target=btc, model=moyenne_5 ; gain [gain_ci_low ; gain_ci_high] |

## Lien avec le marché (notes et bonds, 2009-2026)

| Chiffre | Valeur | Nature | Fichier | Calcul |
|---|---|---|---|---|
| Variation du taux, 20 % des enchères à la part des dealers la plus haute (demande faible) | +1,8 pb [+1,2 ; +2,5] | figé | [quintiles.csv](../data/output/quintiles.csv) | period=2009-2026, surprise=part_des_dealers, quintile=5 ; rate_change_mean_bp [ci_low ; ci_high] |
| Variation du taux, 20 % des enchères à la part des dealers la plus basse (demande forte) | −0,7 pb [−1,1 ; −0,3] | figé | [quintiles.csv](../data/output/quintiles.csv) | period=2009-2026, surprise=part_des_dealers, quintile=1 ; rate_change_mean_bp [ci_low ; ci_high] |
| Variation du taux, 20 % des enchères au bid-to-cover le plus bas par rapport à mean3 | +2,0 pb [+1,2 ; +2,8] | figé | [quintiles.csv](../data/output/quintiles.csv) | period=2009-2026, surprise=bid_to_cover, quintile=5 ; rate_change_mean_bp [ci_low ; ci_high] |
| Variation du taux, 20 % des enchères au bid-to-cover le plus haut par rapport à mean3 | −1,4 pb [−1,7 ; −1,0] | figé | [quintiles.csv](../data/output/quintiles.csv) | period=2009-2026, surprise=bid_to_cover, quintile=1 ; rate_change_mean_bp [ci_low ; ci_high] |

## Stratégie illustrative

| Chiffre | Valeur | Nature | Fichier | Calcul |
|---|---|---|---|---|
| Trades de la stratégie, 2020-2026 | 572 | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=strategie ; trades |
| Gain moyen par trade après coût, stratégie, 2020-2026 | −1,40 pb [−1,91 ; −0,86] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=strategie ; mean_bp [mean_ci_low ; mean_ci_high] |
| Gain moyen par trade après coût, toujours vendeur, 2020-2026 | −1,27 pb [−2,08 ; −0,36] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=toujours_vendeur ; mean_bp [mean_ci_low ; mean_ci_high] |
| Écart de la stratégie avec toujours vendeur, 2020-2026 | −0,13 pb [−0,84 ; +0,58] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=strategie ; vs_vendeur_bp [vs_vendeur_ci_low ; vs_vendeur_ci_high] |
| Taux de réussite de la stratégie, 2020-2026 | 40 % | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=strategie ; hit_rate |
| Sharpe annualisé de la stratégie, 2020-2026 | −2,17 | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=strategie ; sharpe |
| Gain moyen par trade avec information parfaite sur le sens du bid-to-cover, 2020-2026 | +0,52 pb [+0,13 ; +0,87] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2020-2026, rule=information_parfaite ; mean_bp [mean_ci_low ; mean_ci_high] |
| Gain moyen par trade après coût, stratégie, 2014-2026 | −1,00 pb [−1,41 ; −0,59] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2014-2026, rule=strategie ; mean_bp [mean_ci_low ; mean_ci_high] |
| Écart de la stratégie avec toujours vendeur, 2014-2026 | −0,30 pb [−0,77 ; +0,13] | figé | [strategy_summary.csv](../data/output/strategy_summary.csv) | period=2014-2026, rule=strategie ; vs_vendeur_bp [vs_vendeur_ci_low ; vs_vendeur_ci_high] |
| Corrélation entre la surprise réelle de bid-to-cover et la variation du taux, 2014-2026 | −0,22 | figé | [strategy_diagnostics.csv](../data/output/strategy_diagnostics.csv) | name=corr_surprise_move ; value |
| Corrélation entre la partie prévue du bid-to-cover et la variation du taux, 2014-2026 | +0,10 | figé | [strategy_diagnostics.csv](../data/output/strategy_diagnostics.csv) | name=corr_predicted_move ; value |

## Prévisions en direct

| Chiffre | Valeur | Nature | Fichier | Calcul |
|---|---|---|---|---|
| Enchères à venir prévues au dernier passage | 14 | du jour | [live_status.json](../data/output/live_status.json) | upcoming_auctions |
| Lignes du journal figé | 336 | du jour | [predictions_log.parquet](../data/output/predictions_log.parquet) | nombre de lignes |
| Enchères du journal déjà figées (date de l'enchère passée) | 0 | du jour | [predictions_log.parquet](../data/output/predictions_log.parquet) | enchères distinctes dont auction_date ≤ today_new_york de live_status.json |
| Enchères du journal dont le résultat est publié | 0 | du jour | [live_tracking.parquet](../data/output/live_tracking.parquet) | enchères distinctes dont status = realisee |
