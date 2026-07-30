from factored_pipeline.visible_labels import (
    filter_visible_prediction,
    visible_component_weights,
)


def test_visible_policy_removes_hidden_recipe_metadata_and_tiny_components():
    result = visible_component_weights(
        [
            ("chicken", 100),
            ("rice", 80),
            ("olive oil", 10),
            ("salt", 1),
            ("garlic", 1),
            ("parsley", 1),
            ("tiny garnish", 1),
        ]
    )
    assert result == [("chicken", 100.0), ("rice", 80.0)]


def test_visible_policy_keeps_largest_item_when_everything_is_filtered():
    assert visible_component_weights([("salt", 1), ("pepper", 0.5)]) == [
        ("salt", 1.0)
    ]


def test_visible_policy_caps_component_count():
    result = visible_component_weights(
        [(f"item {index}", 10 - index / 10) for index in range(12)]
    )
    assert len(result) == 8


def test_visible_prediction_filter_removes_hidden_items_and_renormalizes():
    result = filter_visible_prediction(
        {
            "not_food": False,
            "container": "plate",
            "items": [
                {"name": "rice", "share_pct": 60},
                {"name": "chicken", "share_pct": 30},
                {"name": "olive oil", "share_pct": 10},
            ],
        }
    )
    assert [item["name"] for item in result["items"]] == ["rice", "chicken"]
    assert sum(item["share_pct"] for item in result["items"]) == 100
