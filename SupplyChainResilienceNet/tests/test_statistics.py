from scrn.statistics import holm_adjust,paired_comparison


def test_holm_non_decreasing_order():
    vals=[.03,.002,.07]
    adjusted=holm_adjust(vals)
    assert all(b>=a for a,b in zip(vals,adjusted))
    assert adjusted[1]<=adjusted[0]<=adjusted[2]


def test_paired_results():
    result=paired_comparison([.8,.81,.79,.82,.83],[.75,.76,.74,.78,.79],bootstrap=100)
    assert result["n_pairs"]==5
    assert result["mean_paired_difference"]>0
    assert 0<=result["wilcoxon_p_unadjusted"]<=1
