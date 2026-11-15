"""Modèles, validation walk-forward et métriques (section 8 du plan).

Chaque trimestre depuis 2014, chaque modèle est entraîné sur tout le passé (fenêtre croissante)
et testé sur le trimestre suivant. Le prétraitement (écrêtage, imputation, standardisation) est
appris sur la seule fenêtre d'entraînement, et les hyperparamètres y sont choisis par
validation croisée temporelle (TimeSeriesSplit). Les dates ne sont jamais mélangées.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tresor import features as ft
from tresor.config import CLEAN_DIR, OUTPUT_DIR, RANDOM_STATE, WALK_FORWARD_START

logger = logging.getLogger(__name__)

TEST_START = pd.Timestamp(WALK_FORWARD_START)
MODEL_NAMES = ("ridge", "lasso", "elasticnet", "random_forest", "lightgbm")
# Benchmark -> colonne des prévisions. « constante » : moyenne de la fenêtre d'entraînement.
BENCHMARKS = {"naive": "naive", "mean3": "mean3", "constante": "constant"}
REFERENCE = "mean3"  # benchmark de référence du plan
ENSEMBLE = "moyenne_5"  # modèle de référence, retenu par principe : moyenne des 5 modèles
RESIDUAL_SUFFIX = "_ecart"  # variante : le modèle prévoit l'écart de la cible à mean3
RESIDUAL_MODELS = ("ridge", "lasso", "elasticnet")
# Le modèle est choisi sur 2014-2019 et jugé sur 2020-2026, pour ne pas choisir sur le test.
PERIODS = {
    "2014-2026": (pd.Timestamp("2014-01-01"), pd.Timestamp("2026-12-31")),
    "2014-2019": (pd.Timestamp("2014-01-01"), pd.Timestamp("2019-12-31")),
    "2020-2026": (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-12-31")),
}
SELECTION_PERIOD, EVALUATION_PERIOD = "2014-2019", "2020-2026"
BOOTSTRAP_DRAWS = 2000
FEATURE_SETS = {
    "toutes": tuple(ft.FEATURE_COLUMNS),
    "sans_nouvelles_sources": tuple(
        ft.feature_groups_without(ft.FEATURE_GROUPS, ft.NEW_SOURCE_GROUPS)
    ),
}
CLIP_QUANTILES = (0.01, 0.99)  # indispensable : sans écrêtage, les modèles linéaires explosent
INNER_SPLITS = 5
# Grille élargie : avec 4 combinaisons, le coin (15 feuilles, 400 arbres) était choisi 22 fois
# sur 52 pour le bid-to-cover des notes et bonds.
LIGHTGBM_GRID = {"num_leaves": [7, 15, 31], "n_estimators": [150, 400, 800]}

PREDICTION_COLUMNS = (
    "family",
    "target",
    "model",
    "feature_set",
    "fold",
    "cusip",
    "auction_date",
    "maturity_bucket",
    "actual",
    "prediction",
    "naive",
    "mean3",
)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Écrête chaque variable entre ses quantiles appris sur l'entraînement (valeurs extrêmes)."""

    def __init__(self, lower: float = CLIP_QUANTILES[0], upper: float = CLIP_QUANTILES[1]):
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> QuantileClipper:
        values = np.asarray(X, dtype=float)
        with warnings.catch_warnings():  # une colonne entièrement vide n'a pas de quantile
            warnings.simplefilter("ignore", RuntimeWarning)
            self.low_ = np.nanquantile(values, self.lower, axis=0)
            self.high_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def make_model(name: str) -> Pipeline:
    """Un modèle complet : prétraitement appris sur l'entraînement, puis l'estimateur."""
    steps: list[tuple[str, object]] = [
        ("ecretage", QuantileClipper()),
        # keep_empty_features : une variable vide sur une fenêtre garde sa colonne.
        ("imputation", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    inner = TimeSeriesSplit(n_splits=INNER_SPLITS)
    if name == "ridge":
        steps += [
            ("standardisation", StandardScaler()),
            ("modele", RidgeCV(alphas=np.logspace(-2, 4, 25), cv=inner)),
        ]
    elif name == "lasso":
        steps += [
            ("standardisation", StandardScaler()),
            ("modele", LassoCV(alphas=30, cv=inner, max_iter=20_000, random_state=RANDOM_STATE)),
        ]
    elif name == "elasticnet":
        steps += [
            ("standardisation", StandardScaler()),
            (
                "modele",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9],
                    alphas=30,
                    cv=inner,
                    max_iter=20_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    elif name == "random_forest":
        # Hyperparamètres fixes : la forêt y est peu sensible, et une recherche par fenêtre
        # multiplierait le temps de calcul par 10 (choix documenté).
        forest = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=20,
            max_features=0.33,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
        steps += [("modele", forest)]
    elif name == "lightgbm":
        booster = LGBMRegressor(
            learning_rate=0.03,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        search = GridSearchCV(
            booster,
            LIGHTGBM_GRID,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_mean_squared_error",
        )
        steps += [("modele", search)]
    else:
        raise ValueError(f"modèle inconnu : {name}")
    return Pipeline(steps)


def quarters(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Trimestres de test, du premier trimestre de `start` à celui de `end`."""
    starts = pd.period_range(start, end, freq="Q").start_time
    return [(s, s + pd.offsets.QuarterEnd(0)) for s in starts]


def winsorized(target: pd.Series) -> pd.Series:
    """Cible d'entraînement écrêtée à 1 % et 99 % (les tails de crise ne dictent pas le modèle).
    L'évaluation se fait toujours sur la cible réelle."""
    low, high = target.quantile(list(CLIP_QUANTILES))
    return target.clip(low, high)


@dataclass(frozen=True)
class Task:
    family: str
    target: str
    model: str  # « lasso », ou « lasso_ecart » pour la variante en écart à mean3
    feature_set: str


def walk_forward(data: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Prévisions hors échantillon trimestre par trimestre, fenêtre d'entraînement croissante.

    Variante « _ecart » : le modèle apprend l'écart de la cible à mean3, puis on rajoute mean3.
    Il prévoit ainsi la surprise plutôt que le niveau, qui dérive sur 17 ans.
    """
    features = list(FEATURE_SETS[task.feature_set])
    residual = task.model.endswith(RESIDUAL_SUFFIX)
    benchmark = f"{task.target}_mean3"
    data = data.sort_values("auction_date", kind="stable")
    known = data[task.target].notna() & (data[benchmark].notna() if residual else True)
    rows = []
    for quarter_start, quarter_end in quarters(TEST_START, data["auction_date"].max()):
        train = data[(data["auction_date"] < quarter_start) & known]
        test = data[data["auction_date"].between(quarter_start, quarter_end)]
        if test.empty or len(train) < 50:
            continue
        model = make_model(task.model.removesuffix(RESIDUAL_SUFFIX))
        target = train[task.target] - train[benchmark] if residual else train[task.target]
        model.fit(train[features].to_numpy(dtype=float), winsorized(target).to_numpy())
        prediction = model.predict(test[features].to_numpy(dtype=float))
        if residual:
            prediction = prediction + test[benchmark].to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "family": task.family,
                    "target": task.target,
                    "model": task.model,
                    "feature_set": task.feature_set,
                    "fold": quarter_start,
                    "cusip": test["cusip"].to_numpy(),
                    "auction_date": test["auction_date"].to_numpy(),
                    "maturity_bucket": test["maturity_bucket"].to_numpy(),
                    "actual": test[task.target].to_numpy(),
                    "prediction": prediction,
                    "naive": test[f"{task.target}_lag1"].to_numpy(),
                    "mean3": test[benchmark].to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --- Benchmarks et modèle de référence ------------------------------------------------------


def add_constant_benchmark(predictions: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Benchmark « constante » : moyenne (écrêtée) de la cible sur la fenêtre d'entraînement.

    C'est ce que prévoit un modèle sans aucune variable : battre mean3 ne suffit pas si l'on ne
    bat pas aussi une constante.
    """
    keys = predictions[["family", "target", "fold"]].drop_duplicates()
    values = []
    for family, target, fold in keys.itertuples(index=False):
        past = sample[(sample["family"] == family) & (sample["auction_date"] < fold)][target]
        values.append((family, target, fold, float(winsorized(past.dropna()).mean())))
    constants = pd.DataFrame(values, columns=["family", "target", "fold", "constant"])
    return predictions.drop(columns="constant", errors="ignore").merge(
        constants, on=["family", "target", "fold"], how="left"
    )


def add_ensemble(predictions: pd.DataFrame) -> pd.DataFrame:
    """Modèle de référence, retenu par principe : la moyenne des prévisions des 5 modèles.

    Il évite de choisir un modèle d'après ses résultats sur la période de test.
    """
    keys = ["family", "target", "feature_set", "fold", "cusip", "auction_date"]
    members = predictions[predictions["model"].isin(MODEL_NAMES)]
    first = members.groupby(keys, sort=False).first().reset_index()
    average = members.groupby(keys, sort=False)["prediction"].mean().reset_index()
    ensemble = first.drop(columns="prediction").merge(average, on=keys).assign(model=ENSEMBLE)
    rest = predictions[predictions["model"] != ENSEMBLE]
    return pd.concat([rest, ensemble[rest.columns]], ignore_index=True)


def benchmark_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Les benchmarks présentés comme des modèles, sur exactement les mêmes enchères."""
    base = predictions[
        (predictions["model"] == MODEL_NAMES[0]) & (predictions["feature_set"] == "toutes")
    ]
    return pd.concat(
        [base.assign(model=name, prediction=base[column]) for name, column in BENCHMARKS.items()],
        ignore_index=True,
    )


# --- Métriques, périodes et intervalles de confiance ------------------------------------------


def _usable(frame: pd.DataFrame) -> pd.DataFrame:
    """Mêmes enchères pour tous les modèles : cible, prévision et benchmarks connus."""
    return frame.dropna(subset=["actual", "prediction", "mean3", "naive", "constant"])


def score(frame: pd.DataFrame) -> dict[str, float]:
    """RMSE, MAE, corrélations, R² hors échantillon (contre mean3 et la constante), direction."""
    usable = _usable(frame)
    error = usable["actual"] - usable["prediction"]
    reference_error = usable["actual"] - usable[REFERENCE]
    constant_error = usable["actual"] - usable["constant"]
    # Direction : le modèle dit-il correctement si l'enchère sera au-dessus ou en dessous de mean3 ?
    moved = np.sign(reference_error) != 0
    predicted_side = np.sign(usable["prediction"] - usable[REFERENCE])
    hits = (predicted_side == np.sign(reference_error))[moved & (predicted_side != 0)]
    # Corrélation des surprises : la prévision s'écarte-t-elle de mean3 dans le bon sens ?
    surprise = reference_error.corr(usable["prediction"] - usable[REFERENCE])
    return {
        "n": len(usable),
        "rmse": float(np.sqrt((error**2).mean())),
        "mae": float(error.abs().mean()),
        "corr": float(usable["actual"].corr(usable["prediction"])),
        "corr_surprise": float(surprise),
        "r2_vs_mean3": float(1 - (error**2).sum() / (reference_error**2).sum()),
        "r2_vs_constant": float(1 - (error**2).sum() / (constant_error**2).sum()),
        "direction": float(hits.mean()) if len(hits) else float("nan"),
    }


def _yearly_errors(frame: pd.DataFrame, reference: str = REFERENCE) -> pd.DataFrame:
    """Sommes des erreurs au carré par année : les blocs du bootstrap."""
    usable = _usable(frame)
    return (
        pd.DataFrame(
            {
                "model": (usable["actual"] - usable["prediction"]) ** 2,
                "reference": (usable["actual"] - usable[reference]) ** 2,
                "year": usable["auction_date"].dt.year,
            }
        )
        .groupby("year")[["model", "reference"]]
        .sum()
    )


def bootstrap_r2(
    frame: pd.DataFrame, other: pd.DataFrame | None = None, draws: int = BOOTSTRAP_DRAWS
) -> tuple[float, float]:
    """Intervalle de confiance à 90 % du R² contre mean3, par bootstrap d'années entières.

    Avec `other`, intervalle de l'écart de R² entre deux jeux de prévisions (mêmes années tirées).
    Rééchantillonner des années, et non des enchères, respecte la dépendance dans le temps.
    """
    first = _yearly_errors(frame)
    rng = np.random.default_rng(RANDOM_STATE)
    draw = rng.integers(0, len(first), size=(draws, len(first)))

    def r2(errors: pd.DataFrame) -> np.ndarray:
        model, reference = errors["model"].to_numpy(), errors["reference"].to_numpy()
        return 1 - model[draw].sum(axis=1) / reference[draw].sum(axis=1)

    values = r2(first)
    if other is not None:
        values = values - r2(_yearly_errors(other).reindex(first.index).fillna(0))
    low, high = np.quantile(values, [0.05, 0.95])
    return float(low), float(high)


def r2_without_best_quarter(frame: pd.DataFrame) -> float:
    """R² contre mean3 sans le trimestre où le modèle gagne le plus : le résultat tient-il à un
    seul épisode (par exemple la crise du plafond de la dette de mai 2023) ?"""
    usable = _usable(frame)
    gains = (
        pd.DataFrame(
            {
                "model": (usable["actual"] - usable["prediction"]) ** 2,
                "reference": (usable["actual"] - usable[REFERENCE]) ** 2,
                "fold": usable["fold"],
            }
        )
        .groupby("fold")[["model", "reference"]]
        .sum()
    )
    best = (gains["reference"] - gains["model"]).idxmax()
    rest = gains.drop(index=best)
    return float(1 - rest["model"].sum() / rest["reference"].sum())


def in_period(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    start, end = PERIODS[period]
    return frame[frame["auction_date"].between(start, end)]


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par période, famille, cible, modèle et jeu de variables, avec intervalle."""
    keys = ["family", "target", "model", "feature_set"]
    all_rows = pd.concat([predictions, benchmark_predictions(predictions)], ignore_index=True)
    rows = []
    for period in PERIODS:
        for group, part in in_period(all_rows, period).groupby(keys, sort=False):
            low, high = bootstrap_r2(part)
            rows.append(
                {
                    "period": period,
                    **dict(zip(keys, group, strict=True)),
                    **score(part),
                    "r2_ci_low": low,
                    "r2_ci_high": high,
                    "r2_without_best_quarter": r2_without_best_quarter(part),
                }
            )
    return pd.DataFrame(rows).sort_values(["period", *keys], ignore_index=True)


def new_sources_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Expérience clé du plan : écart de R² (avec - sans nouvelles sources) et son intervalle."""
    rows = []
    for (family, target, model), part in predictions[
        predictions["model"].isin([*MODEL_NAMES, ENSEMBLE])
    ].groupby(["family", "target", "model"]):
        with_new = part[part["feature_set"] == "toutes"]
        without = part[part["feature_set"] == "sans_nouvelles_sources"]
        if without.empty:
            continue
        low, high = bootstrap_r2(with_new, other=without)
        rows.append(
            {
                "family": family,
                "target": target,
                "model": model,
                "toutes": score(with_new)["r2_vs_mean3"],
                "sans_nouvelles_sources": score(without)["r2_vs_mean3"],
                "gain_ci_low": low,
                "gain_ci_high": high,
            }
        )
    table = pd.DataFrame(rows)
    table["gain"] = table["toutes"] - table["sans_nouvelles_sources"]
    return table


def best_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Modèle choisi sur 2014-2019 seulement, puis jugé sur 2020-2026 (jamais vu au choix).

    La moyenne des 5 modèles, modèle de référence, est donnée à côté.
    """
    full = metrics[metrics["feature_set"] == "toutes"].set_index(
        ["period", "family", "target", "model"]
    )
    rows = []
    for family, target in full.loc[SELECTION_PERIOD].index.droplevel("model").unique():
        candidates = full.loc[(SELECTION_PERIOD, family, target)].loc[list(MODEL_NAMES)]
        chosen = candidates["r2_vs_mean3"].idxmax()
        later = full.loc[(EVALUATION_PERIOD, family, target)]
        rows.append(
            {
                "family": family,
                "target": target,
                "model": chosen,
                "r2_selection": candidates.loc[chosen, "r2_vs_mean3"],
                "r2_evaluation": later.loc[chosen, "r2_vs_mean3"],
                "r2_evaluation_ci_low": later.loc[chosen, "r2_ci_low"],
                "r2_evaluation_ci_high": later.loc[chosen, "r2_ci_high"],
                "r2_vs_constant_evaluation": later.loc[chosen, "r2_vs_constant"],
                "direction_evaluation": later.loc[chosen, "direction"],
                "ensemble_r2_evaluation": later.loc[ENSEMBLE, "r2_vs_mean3"],
                "ensemble_ci_low": later.loc[ENSEMBLE, "r2_ci_low"],
                "ensemble_ci_high": later.loc[ENSEMBLE, "r2_ci_high"],
            }
        )
    return pd.DataFrame(rows)


# --- Lancement -----------------------------------------------------------------------------


def all_tasks(
    families: Sequence[str] = ft.MODELED_FAMILIES,
    targets: Sequence[str] = ft.TARGETS,
    models: Sequence[str] = MODEL_NAMES,
    feature_sets: Sequence[str] = tuple(FEATURE_SETS),
) -> list[Task]:
    tasks = [
        Task(family, target, model, feature_set)
        for family in families
        for target in targets
        for model in models
        for feature_set in feature_sets
    ]
    # Variante « écart à mean3 » : modèles linéaires, toutes variables (rapides, robustesse).
    tasks += [
        Task(family, target, f"{model}{RESIDUAL_SUFFIX}", "toutes")
        for family in families
        for target in targets
        for model in RESIDUAL_MODELS
    ]
    return tasks


def run_walk_forward(sample: pd.DataFrame, tasks: Sequence[Task], n_jobs: int = -1) -> pd.DataFrame:
    """Lance toutes les tâches en parallèle ; chaque tâche ne voit que sa famille."""
    by_family = {family: part for family, part in sample.groupby("family")}
    results = Parallel(n_jobs=n_jobs)(
        delayed(walk_forward)(by_family[task.family], task) for task in tasks
    )
    return pd.concat(results, ignore_index=True)


def finalize_backtest(
    predictions: pd.DataFrame, sample: pd.DataFrame, output_dir: Path = OUTPUT_DIR
) -> pd.DataFrame:
    """Benchmarks, modèle de référence, métriques par période, expérience et choix du modèle."""
    predictions = add_ensemble(add_constant_benchmark(predictions, sample))
    metrics = metrics_table(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "backtest_predictions.parquet", index=False)
    metrics.to_csv(output_dir / "backtest_metrics.csv", index=False)
    new_sources_comparison(predictions).to_csv(
        output_dir / "new_sources_comparison.csv", index=False
    )
    best_models(metrics).to_csv(output_dir / "best_models.csv", index=False)
    return metrics


def run_backtest(
    clean_dir: Path = CLEAN_DIR, output_dir: Path = OUTPUT_DIR, n_jobs: int = -1
) -> pd.DataFrame:
    """Walk-forward complet (environ 100 minutes sur 8 cœurs), puis toutes les synthèses."""
    sample = ft.model_sample(pd.read_parquet(clean_dir / "features.parquet"))
    predictions = run_walk_forward(sample, all_tasks(), n_jobs=n_jobs)
    logger.info("Walk-forward : %d prévisions hors échantillon", len(predictions))
    return finalize_backtest(predictions, sample, output_dir)


# --- Interprétation : permutation et SHAP pour le modèle choisi -----------------------------

# Le modèle interprété est entraîné avant 2022 et expliqué sur 2022 et après, hors échantillon.
INTERPRETATION_SPLIT = pd.Timestamp("2022-01-01")


def _shap_mean_abs(model: Pipeline, x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Valeur absolue moyenne des contributions SHAP de chaque variable, sur toute la période."""
    import shap  # import local : la bibliothèque est lente à charger

    preprocess, estimator = model[:-1], model[-1]
    if isinstance(estimator, GridSearchCV):
        estimator = estimator.best_estimator_
    background = preprocess.transform(x_train)
    explained = preprocess.transform(x_test)
    if isinstance(estimator, LGBMRegressor | RandomForestRegressor):
        values = shap.TreeExplainer(estimator).shap_values(explained)
    else:
        values = shap.LinearExplainer(estimator, background).shap_values(explained)
    return np.abs(values).mean(axis=0)


def interpret(sample: pd.DataFrame, family: str, target: str, model_name: str) -> pd.DataFrame:
    """Importance par permutation (hors échantillon) et SHAP, pour une famille et une cible.

    La permutation mesure ce qu'une variable apporte à la prévision hors échantillon ; SHAP mesure
    combien elle pèse dans la formule du modèle. Les deux peuvent diverger : seule la première dit
    si la variable aide vraiment.
    """
    from sklearn.inspection import permutation_importance

    features = list(FEATURE_SETS["toutes"])
    data = sample[(sample["family"] == family) & sample[target].notna()].sort_values(
        "auction_date", kind="stable"
    )
    train = data[data["auction_date"] < INTERPRETATION_SPLIT]
    test = data[data["auction_date"] >= INTERPRETATION_SPLIT]
    x_train, x_test = train[features].to_numpy(dtype=float), test[features].to_numpy(dtype=float)
    model = make_model(model_name).fit(x_train, winsorized(train[target]).to_numpy())
    permutation = permutation_importance(
        model,
        x_test,
        test[target].to_numpy(),
        scoring="neg_mean_squared_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
    )
    shap_values = _shap_mean_abs(model, x_train, x_test)
    return pd.DataFrame(
        {
            "family": family,
            "target": target,
            "model": model_name,
            "feature": features,
            "group": [ft._group_of(name) for name in features],
            "permutation_mean": permutation.importances_mean,
            "permutation_std": permutation.importances_std,
            "shap_mean_abs": shap_values,
            # Part de chaque variable dans le SHAP de ce modèle : comparable d'une cible à l'autre.
            "shap_share": shap_values / shap_values.sum(),
        }
    )


def run_interpretation(
    clean_dir: Path = CLEAN_DIR, output_dir: Path = OUTPUT_DIR, n_jobs: int = -1
) -> pd.DataFrame:
    """Interprète le modèle choisi sur 2014-2019 pour chaque famille et cible."""
    sample = ft.model_sample(pd.read_parquet(clean_dir / "features.parquet"))
    chosen = pd.read_csv(output_dir / "best_models.csv")
    results = Parallel(n_jobs=n_jobs)(
        delayed(interpret)(sample, row.family, row.target, row.model) for row in chosen.itertuples()
    )
    importance = pd.concat(results, ignore_index=True)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    return importance
