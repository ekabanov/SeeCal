from teacher_labeling.evaluate_bakeoff import labels_match


def test_labels_match_normalizes_taxonomy_and_spelling():
    assert labels_match("bread-wholemeal", "wholemeal bread")
    assert labels_match("spaetzle", "spätzle")
    assert labels_match(
        "yaourt-yahourt-yogourt-ou-yoghourt-natural",
        "yogurt",
    )
    assert labels_match("tomato-raw", "tomato slices")


def test_labels_match_rejects_unrelated_or_generic_only_overlap():
    assert not labels_match("tomato sauce", "mushroom sauce")
    assert not labels_match("bread wholemeal", "red cabbage")
