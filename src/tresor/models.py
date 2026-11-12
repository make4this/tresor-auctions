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
BENCHMARKS = ("naive", "mean3")
REFERENCE = "mean3"  # benchmark de référence du plan
FEATURE_SETS = {
    "toutes": tuple(ft.FEATURE_COLUMNS),
    "sans_nouvelles_sources": tuple(
        ft.feature_groups_without(ft.FEATURE_GROUPS, ft.NEW_SOURCE_GROUPS)
    ),
}
CLIP_QUANTILES = (0.01, 0.99)  # indispensable : sans écrêtage, les modèles linéaires explosent
INNER_SPLITS = 5
LIGHTGBM_GRID = {"num_leaves": [7, 15], "n_estimators": [150, 400]}

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
    model: str
    feature_set: str


def walk_forward(data: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Prévisions hors échantillon trimestre par trimestre, fenêtre d'entraînement croissante."""
    features = list(FEATURE_SETS[task.feature_set])
    data = data.sort_values("auction_date", kind="stable")
    known = data[task.target].notna()
    rows = []
    for quarter_start, quarter_end in quarters(TEST_START, data["auction_date"].max()):
        train = data[(data["auction_date"] < quarter_start) & known]
        test = data[data["auction_date"].between(quarter_start, quarter_end)]
        if test.empty or len(train) < 50:
            continue
        model = make_model(task.model)
        model.fit(train[features].to_numpy(dtype=float), winsorized(train[task.target]).to_numpy())
        prediction = model.predict(test[features].to_numpy(dtype=float))
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
                    "mean3": test[f"{task.target}_mean3"].to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def benchmark_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Les benchmarks naive et mean3 présentés comme des modèles, sur les mêmes enchères."""
    base = predictions[
        (predictions["model"] == MODEL_NAMES[0]) & (predictions["feature_set"] == "toutes")
    ]
    return pd.concat(
        [base.assign(model=name, prediction=base[name]) for name in BENCHMARKS], ignore_index=True
    )


def score(frame: pd.DataFrame) -> dict[str, float]:
    """RMSE, MAE, corrélation, R² hors échantillon contre mean3, bonne direction."""
    usable = frame.dropna(subset=["actual", "prediction", "mean3", "naive"])
    error = usable["actual"] - usable["prediction"]
    reference_error = usable["actual"] - usable[REFERENCE]
    # Direction : le modèle dit-il correctement si l'enchère sera au-dessus ou en dessous de mean3 ?
    moved = np.sign(reference_error) != 0
    predicted_side = np.sign(usable["prediction"] - usable[REFERENCE])
    hits = (predicted_side == np.sign(reference_error))[moved & (predicted_side != 0)]
    return {
        "n": len(usable),
        "rmse": float(np.sqrt((error**2).mean())),
        "mae": float(error.abs().mean()),
        "corr": float(usable["actual"].corr(usable["prediction"])),
        "r2_vs_mean3": float(1 - (error**2).sum() / (reference_error**2).sum()),
        "direction": float(hits.mean()) if len(hits) else float("nan"),
    }


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par famille, cible, modèle et jeu de variables."""
    keys = ["family", "target", "model", "feature_set"]
    all_rows = pd.concat([predictions, benchmark_predictions(predictions)], ignore_index=True)
    table = pd.DataFrame(
        [
            {**dict(zip(keys, group, strict=True)), **score(part)}
            for group, part in all_rows.groupby(keys, sort=False)
        ]
    )
    return table.sort_values(keys, ignore_index=True)


def new_sources_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Expérience clé du plan : R² contre mean3 avec et sans les nouvelles sources."""
    models = metrics[metrics["model"].isin(MODEL_NAMES)]
    table = models.pivot_table(
        index=["family", "target", "model"], columns="feature_set", values="r2_vs_mean3"
    )
    table["gain"] = table["toutes"] - table["sans_nouvelles_sources"]
    return table.reset_index()


def best_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Meilleur modèle (R² contre mean3, toutes variables) pour chaque famille et cible."""
    models = metrics[metrics["model"].isin(MODEL_NAMES) & (metrics["feature_set"] == "toutes")]
    best = models.loc[models.groupby(["family", "target"])["r2_vs_mean3"].idxmax()]
    return best[["family", "target", "model", "r2_vs_mean3", "direction"]].reset_index(drop=True)


def all_tasks(
    families: Sequence[str] = ft.MODELED_FAMILIES,
    targets: Sequence[str] = ft.TARGETS,
    models: Sequence[str] = MODEL_NAMES,
    feature_sets: Sequence[str] = tuple(FEATURE_SETS),
) -> list[Task]:
    return [
        Task(family, target, model, feature_set)
        for family in families
        for target in targets
        for model in models
        for feature_set in feature_sets
    ]


def run_walk_forward(sample: pd.DataFrame, tasks: Sequence[Task], n_jobs: int = -1) -> pd.DataFrame:
    """Lance toutes les tâches en parallèle ; chaque tâche ne voit que sa famille."""
    by_family = {family: part for family, part in sample.groupby("family")}
    results = Parallel(n_jobs=n_jobs)(
        delayed(walk_forward)(by_family[task.family], task) for task in tasks
    )
    return pd.concat(results, ignore_index=True)


def run_backtest(
    clean_dir: Path = CLEAN_DIR, output_dir: Path = OUTPUT_DIR, n_jobs: int = -1
) -> pd.DataFrame:
    """Walk-forward complet, métriques et expérience « nouvelles sources »."""
    sample = ft.model_sample(pd.read_parquet(clean_dir / "features.parquet"))
    predictions = run_walk_forward(sample, all_tasks(), n_jobs=n_jobs)
    metrics = metrics_table(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "backtest_predictions.parquet", index=False)
    metrics.to_csv(output_dir / "backtest_metrics.csv", index=False)
    new_sources_comparison(metrics).to_csv(output_dir / "new_sources_comparison.csv", index=False)
    best_models(metrics).to_csv(output_dir / "best_models.csv", index=False)
    logger.info("Walk-forward : %d prévisions hors échantillon", len(predictions))
    return metrics


# --- Interprétation : permutation et SHAP pour le meilleur modèle ---------------------------

# Le modèle interprété est entraîné avant 2022 et expliqué sur 2022 et après, hors échantillon.
INTERPRETATION_SPLIT = pd.Timestamp("2022-01-01")
SHAP_ROWS = 800


def _shap_mean_abs(model: Pipeline, x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Valeur absolue moyenne des contributions SHAP de chaque variable sur la période test."""
    import shap  # import local : la bibliothèque est lente à charger

    preprocess, estimator = model[:-1], model[-1]
    if isinstance(estimator, GridSearchCV):
        estimator = estimator.best_estimator_
    background = preprocess.transform(x_train)
    explained = preprocess.transform(x_test)[:SHAP_ROWS]
    if isinstance(estimator, LGBMRegressor | RandomForestRegressor):
        values = shap.TreeExplainer(estimator).shap_values(explained)
    else:
        values = shap.LinearExplainer(estimator, background).shap_values(explained)
    return np.abs(values).mean(axis=0)


def interpret(sample: pd.DataFrame, family: str, target: str, model_name: str) -> pd.DataFrame:
    """Importance par permutation et SHAP, pour une famille et une cible."""
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
    return pd.DataFrame(
        {
            "family": family,
            "target": target,
            "model": model_name,
            "feature": features,
            "group": [ft._group_of(name) for name in features],
            "permutation_mean": permutation.importances_mean,
            "permutation_std": permutation.importances_std,
            "shap_mean_abs": _shap_mean_abs(model, x_train, x_test),
        }
    )


def run_interpretation(
    clean_dir: Path = CLEAN_DIR, output_dir: Path = OUTPUT_DIR, n_jobs: int = -1
) -> pd.DataFrame:
    """Interprète le meilleur modèle de chaque famille et cible."""
    sample = ft.model_sample(pd.read_parquet(clean_dir / "features.parquet"))
    best = pd.read_csv(output_dir / "best_models.csv")
    results = Parallel(n_jobs=n_jobs)(
        delayed(interpret)(sample, row.family, row.target, row.model) for row in best.itertuples()
    )
    importance = pd.concat(results, ignore_index=True)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    return importance
