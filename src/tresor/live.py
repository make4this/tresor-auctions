"""Prévisions des enchères à venir et journal figé (phase 8, section 10 du plan).

- `train_models` : entraîne les 5 modèles sur tout l'historique, pour chaque famille et cible,
  et les sauvegarde (joblib) : l'application peut ainsi prévoir sans ré-entraîner.
- `predict_upcoming` : prévoit les enchères à venir avec les modèles sauvegardés.
- `update_log` : ajoute ou rafraîchit les prévisions du journal. À partir du jour de
  l'enchère (date de New York), une ligne est figée et n'est plus jamais réécrite.
- `live_tracking` : compare le journal aux résultats publiés.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from tresor import features as ft
from tresor import models as md
from tresor.config import CLEAN_DIR, MODELS_DIR, NEW_YORK_TZ, OUTPUT_DIR

logger = logging.getLogger(__name__)

FEATURE_SET = "toutes"  # le jeu de variables du modèle de référence de la phase 6
LIVE_MODELS = (*md.MODEL_NAMES, md.ENSEMBLE)
LOG_KEY = ["cusip", "auction_date", "target", "model"]
# Types fixés : le garde-fou compare des lignes relues sur disque et des lignes calculées.
LOG_SCHEMA: dict[str, str] = {
    "cusip": "string",
    "auction_date": "datetime64[us]",
    "target": "string",
    "model": "string",
    "family": "string",
    "maturity_bucket": "string",
    "predicted_at": "datetime64[us, UTC]",  # instant de la prévision, en UTC
    "pred": "float64",
    "bench_mean3": "float64",
    "offering_amt": "float64",
    "offering_estimated": "int64",  # 1 si le montant n'était pas encore annoncé
}
LOG_COLUMNS = list(LOG_SCHEMA)
NUMERIC = ("pred", "bench_mean3", "offering_amt", "offering_estimated")
# Une prévision est réécrite seulement si l'une de ces valeurs change.
COMPARED = [*NUMERIC, "family", "maturity_bucket"]
LOG_FILE = "predictions_log.parquet"
TRACKING_FILE = "live_tracking.parquet"
STATUS_FILE = "live_status.json"
METADATA_FILE = "metadata.json"
CHECKED_VERSIONS = ("numpy", "scikit-learn", "lightgbm")


# --- Écritures sûres -----------------------------------------------------------------------


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Écriture atomique : une coupure ne laisse jamais un fichier à moitié écrit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.part")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def _write_json(content: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.part")
    temporary_path.write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
    temporary_path.replace(path)


def _fingerprint(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


# --- Modèles finaux ------------------------------------------------------------------------


def _model_path(models_dir: Path, family: str, target: str, model: str) -> Path:
    return models_dir / f"{family}__{target}__{model}.joblib"


def _installed_versions() -> dict[str, str]:
    import lightgbm
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "joblib": joblib.__version__,
    }


def _fit_final(sample: pd.DataFrame, family: str, target: str, model: str) -> Pipeline:
    """Même recette que le walk-forward, sur tout l'historique disponible.

    La recherche d'hyperparamètres (LightGBM) est remplacée par le modèle retenu : mêmes
    prévisions, sans les durées de calcul qui changeraient le fichier à chaque entraînement.
    """
    data = sample[(sample["family"] == family) & sample[target].notna()]
    data = data.sort_values("auction_date", kind="stable")  # ordre utilisé par TimeSeriesSplit
    features = list(md.FEATURE_SETS[FEATURE_SET])
    pipeline = md.make_model(model).fit(
        data[features].to_numpy(dtype=float), md.winsorized(data[target]).to_numpy()
    )
    name, estimator = pipeline.steps[-1]
    if isinstance(estimator, GridSearchCV):
        pipeline.steps[-1] = (name, estimator.best_estimator_)
    return pipeline


def _sample_fingerprint(sample: pd.DataFrame) -> str:
    """Empreinte des données d'entraînement : deux entraînements comparables ont la même."""
    columns = ["cusip", "auction_date", "family", *md.FEATURE_SETS[FEATURE_SET], *ft.TARGETS]
    hashed = pd.util.hash_pandas_object(sample[columns], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def train_models(
    sample: pd.DataFrame, models_dir: Path = MODELS_DIR, n_jobs: int = -1
) -> dict[str, object]:
    """Entraîne et sauvegarde les 5 modèles par famille et cible ; renvoie les métadonnées.

    Les modèles sont écrits dans un dossier temporaire, qui remplace l'ancien d'un seul coup.
    """
    tasks = [
        (family, target, model)
        for family in ft.MODELED_FAMILIES
        for target in ft.TARGETS
        for model in md.MODEL_NAMES
    ]
    fitted = Parallel(n_jobs=n_jobs)(delayed(_fit_final)(sample, *task) for task in tasks)
    staging = models_dir.with_name(f"{models_dir.name}.part")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for task, pipeline in zip(tasks, fitted, strict=True):
        joblib.dump(pipeline, _model_path(staging, *task), compress=3)
    metadata: dict[str, object] = {
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "last_auction_date": sample["auction_date"].max().date().isoformat(),
        "sample_sha256": _sample_fingerprint(sample),
        "rows": {
            f"{family}/{target}": int((sample["family"].eq(family) & sample[target].notna()).sum())
            for family in ft.MODELED_FAMILIES
            for target in ft.TARGETS
        },
        "features": list(md.FEATURE_SETS[FEATURE_SET]),
        "models": list(md.MODEL_NAMES),
        # Un modèle sauvegardé ne se relit de façon sûre qu'avec les mêmes versions.
        "versions": _installed_versions(),
    }
    _write_json(metadata, staging / METADATA_FILE)
    old = models_dir.with_name(f"{models_dir.name}.old")
    shutil.rmtree(old, ignore_errors=True)
    if models_dir.exists():
        models_dir.rename(old)
    staging.rename(models_dir)
    shutil.rmtree(old, ignore_errors=True)
    return metadata


def load_models(models_dir: Path = MODELS_DIR) -> dict[tuple[str, str, str], Pipeline]:
    """Relit les modèles sauvegardés, après avoir vérifié versions et variables."""
    metadata_path = models_dir / METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"modèles absents dans {models_dir} (lancer --train)")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    installed = _installed_versions()
    for package in CHECKED_VERSIONS:
        saved = metadata["versions"].get(package)
        if saved != installed[package]:
            raise RuntimeError(
                f"modèles entraînés avec {package} {saved}, installé {installed[package]} :"
                " relancer --train"
            )
    if metadata["features"] != list(md.FEATURE_SETS[FEATURE_SET]):
        raise RuntimeError("les variables ont changé depuis l'entraînement : relancer --train")
    loaded = {}
    for family in ft.MODELED_FAMILIES:
        for target in ft.TARGETS:
            for model in md.MODEL_NAMES:
                path = _model_path(models_dir, family, target, model)
                if not path.exists():
                    raise FileNotFoundError(f"modèle absent : {path.name} (lancer --train)")
                loaded[(family, target, model)] = joblib.load(path)
    return loaded


# --- Prévisions ----------------------------------------------------------------------------


def _as_utc(now: pd.Timestamp) -> pd.Timestamp:
    if now.tzinfo is None:
        raise ValueError("instant sans fuseau horaire : passer un instant en UTC")
    return now.tz_convert("UTC").as_unit("us")


def new_york_today(now: pd.Timestamp) -> pd.Timestamp:
    """Date du jour à New York (sans heure), pour un instant avec fuseau horaire."""
    return _as_utc(now).tz_convert(NEW_YORK_TZ).normalize().tz_localize(None).as_unit("us")


def upcoming_rows(features: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Enchères modélisées à venir, strictement après aujourd'hui (date de New York)."""
    return features[
        features["family"].isin(ft.MODELED_FAMILIES)
        & features["status"].eq("upcoming")
        & (features["auction_date"] > today)
    ]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=LOG_SCHEMA[name]) for name in columns})


def predict_upcoming(
    rows: pd.DataFrame, models: dict[tuple[str, str, str], Pipeline]
) -> pd.DataFrame:
    """Une ligne par enchère, cible et modèle (dont la moyenne des 5), au format du journal."""
    features = list(md.FEATURE_SETS[FEATURE_SET])
    parts = []
    for family, group in rows.groupby("family"):
        x = group[features].to_numpy(dtype=float)
        for target in ft.TARGETS:
            predictions = {
                model: models[(family, target, model)].predict(x) for model in md.MODEL_NAMES
            }
            predictions[md.ENSEMBLE] = np.mean(list(predictions.values()), axis=0)
            for model, values in predictions.items():
                parts.append(
                    pd.DataFrame(
                        {
                            "cusip": group["cusip"].to_numpy(),
                            "auction_date": group["auction_date"].to_numpy(),
                            "target": target,
                            "model": model,
                            "family": family,
                            "maturity_bucket": group["maturity_bucket"].to_numpy(),
                            "pred": values,
                            "bench_mean3": group[f"{target}_mean3"].to_numpy(),
                            "offering_amt": group["offering_amt"].to_numpy(dtype=float),
                            "offering_estimated": group["offering_estimated"].to_numpy(dtype=int),
                        }
                    )
                )
    columns = [name for name in LOG_COLUMNS if name != "predicted_at"]
    if not parts:
        return _empty(columns)
    return pd.concat(parts, ignore_index=True)[columns]


# --- Journal figé ----------------------------------------------------------------------------


def normalize_log(frame: pd.DataFrame) -> pd.DataFrame:
    """Colonnes du journal, dans l'ordre et avec les types fixés ; refuse tout écart."""
    columns = set(frame.columns)
    if columns != set(LOG_COLUMNS):
        missing, extra = set(LOG_COLUMNS) - columns, columns - set(LOG_COLUMNS)
        raise ValueError(
            f"journal des prévisions : colonnes absentes {sorted(missing)},"
            f" inattendues {sorted(extra)}"
        )
    result = {}
    for name, dtype in LOG_SCHEMA.items():
        values = frame[name].reset_index(drop=True)
        if name == "auction_date":
            values = pd.to_datetime(values).astype(dtype)
        elif name == "predicted_at":
            values = pd.to_datetime(values, utc=True).astype(dtype)
        else:
            values = values.astype(dtype)
        result[name] = values
    return pd.DataFrame(result)


def _check_unique(log: pd.DataFrame) -> None:
    if log.duplicated(LOG_KEY).any():
        raise ValueError("journal des prévisions : clé en double")


def _sorted(log: pd.DataFrame) -> pd.DataFrame:
    return log.sort_values(LOG_KEY, kind="stable").reset_index(drop=True)


def _same(old: pd.DataFrame, new: pd.DataFrame) -> np.ndarray:
    """Vrai si la prévision n'a pas changé (les valeurs manquantes sont égales entre elles)."""
    same = np.ones(len(new), dtype=bool)
    for column in COMPARED:
        a, b = old[column].reset_index(drop=True), new[column].reset_index(drop=True)
        if column in NUMERIC:
            same &= np.isclose(
                a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=0, atol=1e-12,
                equal_nan=True,
            )  # fmt: skip
        else:
            a, b = a.astype("string"), b.astype("string")
            same &= ((a == b).fillna(False) | (a.isna() & b.isna())).to_numpy(dtype=bool)
    return same


def check_frozen_unchanged(old: pd.DataFrame, new: pd.DataFrame, today: pd.Timestamp) -> None:
    """Garde-fou : chaque ligne figée de l'ancien journal (enchère aujourd'hui ou passée) doit
    se retrouver à l'identique, toutes colonnes comprises, dans le nouveau."""
    old, new = normalize_log(old), normalize_log(new)
    frozen = old[old["auction_date"] <= today]
    if frozen.empty:
        return
    found = frozen[LOG_KEY].merge(new, on=LOG_KEY, how="left", indicator=True)
    if (found["_merge"] != "both").any():
        raise RuntimeError("une ligne figée du journal a disparu : écriture annulée")
    kept = normalize_log(found.drop(columns="_merge"))
    try:
        pd.testing.assert_frame_equal(_sorted(kept), _sorted(frozen), check_exact=True)
    except AssertionError as error:
        raise RuntimeError("une ligne figée du journal a changé : écriture annulée") from error


def update_log(log: pd.DataFrame, predictions: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Ajoute ou rafraîchit les prévisions, sans jamais toucher une ligne figée.

    - Une prévision n'est écrite que pour une enchère postérieure à aujourd'hui (date de New
      York) : la clé d'une ligne figée, qui contient la date de l'enchère, ne peut correspondre
      à aucune nouvelle prévision.
    - L'heure doit avancer : un instant antérieur à la dernière prévision du journal est refusé
      (sinon, relancer « hier » pourrait rouvrir une ligne figée aujourd'hui).
    - Une prévision inchangée est gardée telle quelle, avec sa date d'origine.
    """
    now = _as_utc(now)
    log = normalize_log(log)
    _check_unique(log)
    if not log.empty and now < log["predicted_at"].max():
        raise ValueError(
            "instant antérieur à la dernière prévision du journal : refusé (le gel suppose une"
            " horloge qui avance)"
        )
    today = new_york_today(now)
    fresh = normalize_log(predictions.assign(predicted_at=now))
    fresh = fresh[fresh["auction_date"] > today].reset_index(drop=True)
    _check_unique(fresh)
    if fresh.empty:
        result = log
    else:
        previous = fresh[LOG_KEY].merge(log, on=LOG_KEY, how="left", indicator=True)
        known = (previous["_merge"] == "both").to_numpy()
        changed = fresh[~(known & _same(previous, fresh))]
        replaced = log.set_index(LOG_KEY).index.isin(changed.set_index(LOG_KEY).index)
        result = pd.concat([log[~replaced], changed], ignore_index=True)
    result = _sorted(normalize_log(result))
    _check_unique(result)
    check_frozen_unchanged(log, result, today)
    return result


def read_log(path: Path) -> pd.DataFrame:
    """Journal existant, ou journal vide au bon format."""
    if not path.exists():
        return _empty(LOG_COLUMNS)
    return normalize_log(pd.read_parquet(path))


def live_tracking(log: pd.DataFrame, features: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Journal et résultats publiés : erreur du modèle et de mean3, statut de chaque prévision.

    Statut : `realisee` (enchère réalisée), `sans_resultat` (date passée sans résultat :
    enchère déplacée, annulée ou résultat pas encore collecté), `en_attente` sinon.
    """
    done = features[features["status"].eq("completed")]
    results = done.melt(
        id_vars=["cusip", "auction_date"],
        value_vars=list(ft.TARGETS),
        var_name="target",
        value_name="actual",
    )
    keys = ["cusip", "auction_date", "target"]
    tracked = normalize_log(log).merge(
        results.astype({"cusip": "string", "auction_date": "datetime64[us]", "target": "string"}),
        on=keys,
        how="left",
    )
    completed = tracked.set_index(["cusip", "auction_date"]).index.isin(
        done.astype({"cusip": "string", "auction_date": "datetime64[us]"})
        .set_index(["cusip", "auction_date"])
        .index
    )
    past = (tracked["auction_date"] < today).to_numpy()
    tracked["status"] = np.select([completed, past], ["realisee", "sans_resultat"], "en_attente")
    tracked["error"] = tracked["actual"] - tracked["pred"]
    tracked["bench_error"] = tracked["actual"] - tracked["bench_mean3"]
    return tracked


# --- Lancement -------------------------------------------------------------------------------


def _up_to_date(sample: pd.DataFrame, models_dir: Path) -> dict[str, object] | None:
    """Métadonnées des modèles s'ils ont été entraînés sur exactement ces données, avec les
    mêmes versions et variables ; sinon rien. (Ré-entraîner donnerait les mêmes prévisions,
    mais pas toujours des fichiers identiques octet par octet.)"""
    path = models_dir / METADATA_FILE
    if not path.exists():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    current = (
        metadata.get("sample_sha256") == _sample_fingerprint(sample)
        and metadata.get("features") == list(md.FEATURE_SETS[FEATURE_SET])
        and metadata.get("models") == list(md.MODEL_NAMES)
        and metadata.get("versions") == _installed_versions()
    )
    complete = all(
        _model_path(models_dir, family, target, model).exists()
        for family in ft.MODELED_FAMILIES
        for target in ft.TARGETS
        for model in md.MODEL_NAMES
    )
    return metadata if current and complete else None


def run_train(
    clean_dir: Path = CLEAN_DIR,
    models_dir: Path = MODELS_DIR,
    n_jobs: int = -1,
    force: bool = False,
) -> dict[str, object]:
    """Étape --train : modèles finaux sur toutes les enchères réalisées depuis 2009.

    Si les données d'entraînement n'ont pas changé, les modèles existants sont gardés.
    """
    sample = ft.model_sample(pd.read_parquet(clean_dir / "features.parquet"))
    existing = None if force else _up_to_date(sample, models_dir)
    if existing is not None:
        logger.info("Modèles déjà à jour (mêmes données d'entraînement) : rien à refaire")
        return existing
    metadata = train_models(sample, models_dir, n_jobs)
    logger.info("Modèles finaux entraînés jusqu'au %s", metadata["last_auction_date"])
    return metadata


def _market_dates(clean_dir: Path) -> dict[str, str]:
    """Dernière date de chaque série de marché : la fraîcheur des données de la prévision."""
    path = clean_dir / "market.parquet"
    if not path.exists():
        return {}
    last = pd.read_parquet(path, columns=["series", "date"]).groupby("series")["date"].max()
    return {series: day.date().isoformat() for series, day in last.items()}


def run_predict(
    clean_dir: Path = CLEAN_DIR,
    output_dir: Path = OUTPUT_DIR,
    models_dir: Path = MODELS_DIR,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Étape --predict : prévoit les enchères à venir, met à jour le journal et le suivi."""
    now = _as_utc(pd.Timestamp.now(tz="UTC") if now is None else now)
    today = new_york_today(now)
    log_path, tracking_path = output_dir / LOG_FILE, output_dir / TRACKING_FILE
    if not log_path.exists() and tracking_path.exists():
        raise FileNotFoundError(
            f"{LOG_FILE} absent alors que le suivi existe : restaurer le journal depuis git"
        )
    before = _fingerprint(log_path)
    features = pd.read_parquet(clean_dir / "features.parquet")
    rows = upcoming_rows(features, today)
    predictions = predict_upcoming(rows, load_models(models_dir))
    log = update_log(read_log(log_path), predictions, now)
    if _fingerprint(log_path) != before:  # une autre exécution a écrit entre-temps
        raise RuntimeError("journal modifié par une autre exécution : relancer --predict")
    _write_parquet(log, log_path)
    _write_parquet(live_tracking(log, features, today), tracking_path)
    _write_json(
        {
            "run_at": now.isoformat(),
            "today_new_york": today.date().isoformat(),
            "upcoming_auctions": int(len(rows)),
            "log_rows": int(len(log)),
            "market_last_dates": _market_dates(clean_dir),
        },
        output_dir / STATUS_FILE,
    )
    logger.info("Prévisions : %d enchères à venir, %d lignes dans le journal",
                len(rows), len(log))  # fmt: skip
    return log
