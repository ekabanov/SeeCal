from portion_prior_audit import portion_estimates


def test_portion_estimators_are_deterministic():
    values = portion_estimates([(0.75, 150), (0.25, 50)])
    assert values["median_portion_div_share"] == 200
    assert values["sum_item_portions"] == 200
    assert values["dominant_portion_div_share"] == 200
    assert values["least_squares_portion_fit"] == 200
    assert values["share_weighted_portion"] == 125


def test_sum_item_portions_does_not_average_or_share_scale_portion_kinds():
    values = portion_estimates([(0.9, 112), (0.1, 897)])

    assert values["sum_item_portions"] == 1009
    assert values["median_portion_div_share"] != 1009


def test_portion_estimators_reject_missing_values():
    assert portion_estimates([]) == {}
    assert portion_estimates([(0, 100), (1, 0)]) == {}
