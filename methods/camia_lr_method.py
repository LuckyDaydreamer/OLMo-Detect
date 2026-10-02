"""
CAMIA — SUPERVISED variant (paper Appendix B; official run_ours_train_lr.py).

The paper's "with additional labeled data" attack: fit a LogisticRegression classifier
on the SAME context-aware signal features as the main CAMIA method, using labeled
member (member dev) + non-member (non-member dev) examples, then score the
eval set with predict_proba. Official reference:
    run_ours_train_lr.py::train_model_single  (LogisticRegression + MinMaxScaler; PCA variant)

Design:
  * Feature extraction is INHERITED from methods/camia_method.CAMIAMethod (_compute_sigs ->
    _signals_matrix), so the feature set is byte-identical to camia / camia_faithful.
  * Training pool  = member dev (members, y=1) + non-member dev (non-members, y=0).
  * Scoring        = clf.predict_proba(eval)[:, 1]  (higher = stronger membership).
  * We report BOTH plain LR (camia_lr_score) and LR-on-PCA (camia_lr_pca_score); the PCA
    variant is more robust when the per-domain dev split is small.

This is the SUPERVISED counterpart to the (now unsupervised) camia_faithful. It does not
touch the eval labels (fit on dev only); no y_all sign-flip leak.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_text
from model_utils import ModelBundle

try:                                   # loaded as methods.camia_lr_method (package) …
    from methods.camia_method import CAMIAMethod, _derive_con_dev_path
except ImportError:                    # … or with methods/ directly on sys.path
    from camia_method import CAMIAMethod, _derive_con_dev_path


def _fit_score_lr(con_sigs, uncon_sigs, target_sigs, n_pca: int = 10):
    """Train LR on (members=con_sigs, non-members=uncon_sigs); score target_sigs.

    Each *_sigs is a dict {signal_key: 1-D array over samples} from _signals_matrix.
    Returns (lr_scores, lr_pca_scores, feature_keys).
    """
    keys = sorted(set(con_sigs) & set(uncon_sigs) & set(target_sigs))
    if not keys:
        n = len(next(iter(target_sigs.values()))) if target_sigs else 0
        nan = np.full(n, float("nan"))
        return nan, nan, keys

    def mat(d):
        return np.column_stack([np.asarray(d[k], dtype=float) for k in keys])

    Xc, Xu, Xt = mat(con_sigs), mat(uncon_sigs), mat(target_sigs)
    X_train = np.vstack([Xc, Xu])
    y_train = np.concatenate([np.ones(len(Xc)), np.zeros(len(Xu))])

    # impute non-finite with per-column train median (fallback 0.0)
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    def imp(X):
        X = np.array(X, dtype=float)
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(med, np.where(bad)[1])
        return X

    X_train, Xt = imp(X_train), imp(Xt)

    scaler = MinMaxScaler().fit(X_train)
    Xtr_s, Xt_s = scaler.transform(X_train), scaler.transform(Xt)

    clf = LogisticRegression(max_iter=5000, solver="liblinear").fit(Xtr_s, y_train)
    lr_scores = clf.predict_proba(Xt_s)[:, 1]

    # PCA variant (robust for small dev): guard n_components
    ncomp = max(1, min(n_pca, Xtr_s.shape[1], max(1, Xtr_s.shape[0] - 1)))
    pca = PCA(n_components=ncomp).fit(Xtr_s)
    clf2 = LogisticRegression(max_iter=5000, solver="liblinear").fit(pca.transform(Xtr_s), y_train)
    lr_pca_scores = clf2.predict_proba(pca.transform(Xt_s))[:, 1]

    return lr_scores, lr_pca_scores, keys


class CAMIALRMethod(CAMIAMethod):
    name = "camia_lr"

    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict:
        if not self.con_dev_path:
            self.con_dev_path = _derive_con_dev_path(dataset.path, setting="matched")
        print(f"[camia_lr] con_dev={self.con_dev_path}  uncon_dev={self.uncon_dev_path}")

        prepared = []
        for idx, record in enumerate(dataset.records):
            text = extract_text(record)
            if text:
                prepared.append({"idx": idx,
                                 "id": extract_sample_id(record, fallback=idx),
                                 "text": text})
        if not prepared:
            return self._empty(dataset)
        texts = [p["text"] for p in prepared]

        con_texts   = self._load_texts(self.con_dev_path,   self.max_calibration_samples)
        uncon_texts = self._load_texts(self.uncon_dev_path, self.max_calibration_samples)
        if not con_texts or not uncon_texts:
            return self._empty(dataset)

        print(f"[camia_lr] member dev ({len(con_texts)}) forward passes…")
        con_sigs    = self._compute_sigs(model_bundle, con_texts,   "camia_lr_con_dev")
        print(f"[camia_lr] non-member dev ({len(uncon_texts)}) forward passes…")
        uncon_sigs  = self._compute_sigs(model_bundle, uncon_texts, "camia_lr_uncon_dev")
        print(f"[camia_lr] target ({len(texts)}) forward passes…")
        target_sigs = self._compute_sigs(model_bundle, texts,       "camia_lr_target")

        lr_scores, pca_scores, keys = _fit_score_lr(con_sigs, uncon_sigs, target_sigs)

        samples, score_list = [], []
        for i, p in enumerate(prepared):
            sc  = float(lr_scores[i])  if i < len(lr_scores)  else float("nan")
            scp = float(pca_scores[i]) if i < len(pca_scores) else float("nan")
            score_list.append(sc)
            samples.append({
                "record_index":       p["idx"],
                "sample_id":          p["id"],
                "text":               p["text"],
                "camia_lr_score":     sc,
                "camia_lr_pca_score": scp,
                "camia_score":        sc,   # alias for (method, field) readers
            })
        return {
            "method":              self.name,
            "dataset":             dataset.name,
            "dataset_path":        dataset.path,
            "dataset_data_type":   dataset.data_type,
            "con_dev_path":        self.con_dev_path,
            "uncon_dev_path":      self.uncon_dev_path,
            "n_train_members":     len(con_texts),
            "n_train_nonmembers":  len(uncon_texts),
            "n_features":          len(keys),
            "max_tokens":          self.max_tokens,
            "num_total_records":   len(dataset.records),
            "num_scored_records":  len(samples),
            "num_skipped_records": len(dataset.records) - len(samples),
            "summary": {"mean_camia_score":
                        float(np.nanmean(score_list)) if score_list else float("nan")},
            "samples": samples,
        }


def build_method(uncon_dev_path: str, con_dev_path: str = "", model_size: str = "",
                 shifted: bool = False, max_tokens: int = 512, batch_size: int = 2,
                 max_calibration_samples: int = 150, **kwargs) -> BaseMethod:
    return CAMIALRMethod(
        uncon_dev_path          = uncon_dev_path,
        con_dev_path            = con_dev_path,
        max_tokens              = max_tokens,
        batch_size              = batch_size,
        max_calibration_samples = max_calibration_samples,
    )
