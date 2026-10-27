"""MIA and a disclosed implementation of CMIA Sec.3.3 (arXiv:2506.20893v5).

The original CMIA metric is a response-rate gap, NOT a membership ROC AUC.
cmia_model_auc_extension is an additional model-discrimination statistic.
Selection and classifier fitting use attack_cal; final evaluation uses test.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def binary_auc(positive, negative):
    positive, negative = np.asarray(positive), np.asarray(negative)
    if not len(positive) or not len(negative):
        raise ValueError('AUC needs nonempty positive and negative samples')
    values = np.r_[positive, negative]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite attack scores')
    return float(roc_auc_score(np.r_[np.ones(len(positive)), np.zeros(len(negative))], values))


def log_true_probability(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z[np.arange(len(z)), labels] - np.log(np.exp(z).sum(axis=1))


def mia_metrics(member_cal, nonmember_cal, member_test, nonmember_test):
    """Higher score means member; direction calibrated BEFORE evaluating test."""
    sign = 1 if binary_auc(member_cal, nonmember_cal) >= .5 else -1
    threshold = float(np.quantile(sign * np.asarray(nonmember_cal), .99))
    return {
        'mia_auc_raw': binary_auc(member_test, nonmember_test),
        'mia_auc': binary_auc(sign * np.asarray(member_test), sign * np.asarray(nonmember_test)),
        'mia_direction': sign,
        'mia_tpr_at_calibrated_1pct_fpr': float(np.mean(sign * np.asarray(member_test) > threshold)),
        'mia_actual_test_fpr': float(np.mean(sign * np.asarray(nonmember_test) > threshold)),
        'mia_eval_members': len(member_test),
        'mia_eval_nonmembers': len(nonmember_test),
    }


def fit_neighbor(logits, labels, neighbor, retained):
    keep = np.isin(labels, retained)
    x = np.asarray(logits)[keep, neighbor].reshape(-1, 1)
    y = (np.asarray(labels)[keep] == neighbor).astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError('CMIA needs at least two retained classes with calibration samples')
    clf = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight='balanced', solver='lbfgs', C=1.0, max_iter=1000, random_state=0))
    clf.fit(x, y)
    return clf


def cmia_evaluate(target_cal, target_test, references, cal_labels, test_labels, forget, selection_references=None):
    """references: [(cal_logits, test_logits), ...] from retained-only retrains.

    Macro over individual forgotten classes for a multi-class request is our
    disclosed extension. Neighbors are chosen using reference calibration only.
    Logistic-regression choice is explicit; not claimed bitwise official code.
    """
    cal_labels, test_labels = np.asarray(cal_labels), np.asarray(test_labels)
    retained = [c for c in range(target_cal.shape[1]) if c not in forget]
    if not references:
        raise ValueError('CMIA requires independent retrained reference models')
    ref_clfs = {r: [fit_neighbor(zc, cal_labels, r, retained) for zc, _ in references]
                for r in retained}
    selection_references = references if selection_references is None else selection_references
    selection_clfs = {r: [fit_neighbor(zc, cal_labels, r, retained) for zc, _ in selection_references]
                      for r in retained}
    per_class = []
    for f in forget:
        cm, tm = cal_labels == f, test_labels == f
        def reference_rate(r):
            return np.mean([np.mean(clf.predict(zc[cm, r:r+1]))
                            for clf, (zc, _) in zip(selection_clfs[r], selection_references)])
        neighbor = max(retained, key=lambda r: (reference_rate(r), -r))
        target_clf = fit_neighbor(target_cal, cal_labels, neighbor, retained)
        pcal = target_clf.predict_proba(target_cal[cm, neighbor:neighbor+1])[:, 1]
        ptest = target_clf.predict_proba(target_test[tm, neighbor:neighbor+1])[:, 1]
        rcal, rtest, rates = [], [], []
        for clf, (zc, zt) in zip(ref_clfs[neighbor], references):
            rcal.append(clf.predict_proba(zc[cm, neighbor:neighbor+1])[:, 1])
            rtest.append(clf.predict_proba(zt[tm, neighbor:neighbor+1])[:, 1])
            rates.append(float(np.mean(clf.predict(zt[tm, neighbor:neighbor+1])) * 100))
        # Match sample counts: average reference probability per query image.
        rcal, rtest = np.mean(rcal, axis=0), np.mean(rtest, axis=0)
        direction = 1 if binary_auc(pcal, rcal) >= .5 else -1
        response = float(np.mean(target_clf.predict(target_test[tm, neighbor:neighbor+1])) * 100)
        ref_rate = float(np.mean(rates))
        per_class.append({
            'forget_class': int(f), 'neighbor_class': int(neighbor),
            'cmia_response_pct': response, 'cmia_reference_pct': ref_rate,
            'cmia_gap_pp': abs(response - ref_rate),
            'cmia_model_auc_extension': binary_auc(direction * ptest, direction * rtest),
            'cmia_auc_direction': direction,
            'reference_models': len(references),
        })
    keys = ('cmia_response_pct', 'cmia_reference_pct', 'cmia_gap_pp', 'cmia_model_auc_extension')
    return {**{k: float(np.mean([x[k] for x in per_class])) for k in keys}, 'per_class': per_class}


def unavailable_mia():
    """No genuine nonmembers exist in the all_shared data protocol."""
    return {**{k: None for k in ('mia_auc_raw', 'mia_auc', 'mia_direction',
            'mia_tpr_at_calibrated_1pct_fpr', 'mia_actual_test_fpr')},
            'mia_eval_members': 0, 'mia_eval_nonmembers': 0,
            'mia_status': 'unavailable_no_held_out_nonmembers'}
