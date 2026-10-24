"""Folktables ACSIncome with explicit source/target states and source-only scaling."""
from pathlib import Path
import numpy as np
from .common import DatasetBundle, DatasetError, options_dict, source_split, standardize_train


def load(root, options=None, download=False):
    options = options_dict(options)
    source_state = options.get("source_state", "CA")
    targets = options.get("target_states")
    if not targets or isinstance(targets, str):
        raise DatasetError("ACSIncome requires an explicit nonempty target_states list; the manuscript does not identify all five states")
    states = [source_state] + list(targets)
    if len(states) != len(set(states)):
        raise DatasetError("Source and target states must be distinct")
    year = str(options.get("survey_year", "2018"))
    try:
        from folktables import ACSDataSource, ACSIncome
    except ImportError as error:
        raise DatasetError("Install the data extra: pip install '.[data]' (folktables)") from error
    source = ACSDataSource(survey_year=year, horizon=options.get("horizon", "1-Year"),
                           survey="person", root_dir=str(Path(root)))
    xs, ys, domains = [], [], []
    for state in states:
        frame = source.get_data(states=[state], download=bool(download))
        x, y, _group = ACSIncome.df_to_numpy(frame)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 10:
            raise DatasetError("ACSIncome must expose the official ten predictor columns")
        xs.append(x); ys.append(np.asarray(y, dtype=np.int64).reshape(-1))
        domains.extend([state] * len(y))
    source_n = len(ys[0])
    train, valid = source_split(np.arange(source_n), seed=int(options.get("seed", 1)),
                                valid_fraction=float(options.get("valid_fraction", 0.2)))
    y = np.concatenate(ys)
    bundle = DatasetBundle("acs_income", "tabular", y,
                           {"train": train, "valid": valid, "test": np.arange(source_n, len(y))},
                           2, x=np.concatenate(xs),
                           metadata={"source": "https://github.com/socialfoundations/folktables",
                                     "source_state": source_state, "target_states": list(targets),
                                     "survey_year": year, "domains": domains,
                                     "features": ["AGEP","COW","SCHL","MAR","OCCP","POBP","RELP","WKHP","SEX","RAC1P"],
                                     "target": "PINCP > 50000", "metric": "accuracy",
                                     "split": "source_random_train_valid__target_test",
                                     "feature_encoding": "official numeric/categorical codes, then train-only standardization"})
    return standardize_train(bundle.validate(), options.get("standardize", True))
