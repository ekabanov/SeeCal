from scale_ordering_audit import audit


def test_ordering_audit_uses_median_view_prediction_per_size():
    rows = []
    for size, predictions in (
        ("small", (10, 12, 100)),
        ("average", (20, 22, 24)),
        ("big", (30, 32, 34)),
    ):
        rows.extend(
            {
                "id": f"fpb:test:rice_{size}_RGB_IMG_{index}",
                "p50_g": prediction,
            }
            for index, prediction in enumerate(predictions)
        )

    result = audit(rows)

    assert result["ordered_triads"] == 1
    assert result["complete_triads"] == 1
    assert result["pairwise_correct"] == 3
    assert result["failed_families"] == {}
