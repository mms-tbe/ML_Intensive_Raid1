import numpy as np

from scripts.preprocessing import select_threshold


def test_select_threshold_meets_recall_and_maximizes_precision():
    # 5 позитивов (proba высокая), 5 негативов; один негатив "шумный" на 0.6
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.95, 0.9, 0.85, 0.8, 0.55, 0.6, 0.3, 0.2, 0.1, 0.05])
    thr = select_threshold(y, p, min_recall=0.80)
    pred = (p >= thr).astype(int)
    recall = pred[y == 1].mean()
    assert recall >= 0.80
    # при recall>=0.8 порог не должен падать ниже 0.55 (иначе зря ловим негатив 0.6)
    assert 0.55 <= thr <= 0.8


def test_select_threshold_fallback_when_target_unreachable():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.4, 0.3, 0.2, 0.1])
    thr = select_threshold(y, p, min_recall=0.99)
    assert (p >= thr)[y == 1].mean() >= 0.99
