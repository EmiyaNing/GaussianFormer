"""Decision rules on a fixed occupancy probability and conditional semantics."""
import numpy as np

TEMPERATURES = (.5, .75, 1., 1.5, 2.)
RISK_RATIOS = (0., .25, .5, .75, 1.)


def log_odds(o):
    o = np.asarray(o, dtype=np.float64)
    with np.errstate(divide='ignore'):
        return np.log(o) - np.log1p(-o)


def temperature(q, value):
    q = np.asarray(q, dtype=np.float64)
    if value == 1:
        return q
    with np.errstate(divide='ignore'):
        a = np.log(q) / value
    a -= a.max(-1, keepdims=True)
    a = np.exp(a)
    return a / a.sum(-1, keepdims=True)


def temperature_preserving_winner(q, value):
    """Retain the original winner if float rounding creates a maximum tie.

    A positive temperature is monotone in exact arithmetic. At machine
    precision two distinct values can round to the same transformed value.
    We keep the original maximizer among that tied maximum set, and reject
    every strict inversion instead of silently relaxing the rank check.
    """
    q=np.asarray(q)
    winner=q.argmax(-1)
    transformed=temperature(q,value)
    retained=np.take_along_axis(transformed,winner[...,None],axis=-1)[...,0]
    np.testing.assert_array_equal(retained,transformed.max(-1))
    ties=int((transformed.argmax(-1)!=winner).sum())
    return transformed,winner,ties


def risk_score(o, qmax, ratio):
    """C_FN=C_FP=1, C_class=ratio: foreground iff score >= 0.

    The foreground action risk is (1-o)+ratio*o*(1-qmax);
    the empty action risk is o. Ratio1 is joint MAP, ratio0 hierarchical.
    A learned per-winner intercept is fitted separately on training scenes.
    """
    qmax = np.asarray(qmax, dtype=np.float64)
    assert 0 <= ratio <= 1
    with np.errstate(divide='ignore'):
        return log_odds(o) + np.log((1-ratio) + ratio*qmax)


def predict(o, q, ratio=1., threshold=0.):
    fg = np.asarray(q).argmax(-1)
    score = risk_score(o, np.asarray(q).max(-1), ratio)
    return np.where(score >= threshold, fg, 17)


def confusion(y, pred):
    return np.bincount(np.asarray(y, dtype=np.int64)*18 + pred,
                       minlength=324).reshape(18, 18)


def event_counts(y, q, o):
    fg = q.argmax(-1)
    joint = predict(o, q)
    hierarchical = predict(o, q, ratio=0)
    gap = (hierarchical != 17) & (joint == 17)
    return dict(gap=int(gap.sum()), gap_true_foreground=int((gap & (y<17)).sum()),
        gap_true_empty=int((gap & (y==17)).sum()),
        gap_correct_winner=int((gap & (y==fg) & (y<17)).sum()),
        gap_wrong_winner=int((gap & (y!=fg) & (y<17)).sum()),
        foreground=int((y<17).sum()), empty=int((y==17).sum()),
        native_missed=int(((y<17)&(joint==17)).sum()),
        native_recoverable=int(((y<17)&(fg==y)&(joint==17)).sum()),
        saturated_zero=int((o==0).sum()), saturated_one=int((o==1).sum()))
