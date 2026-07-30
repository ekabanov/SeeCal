import json

from factored_pipeline.resolver import SQLiteNutritionResolver
from factored_pipeline.eval_taxonomy import EvaluationTaxonomy
from factored_pipeline.scoring import score_dish, summarize_dishes
from make_fdc_db import build_database
from score_harness import assemble_audit, compare_assemblies
from test_fdc_db import make_fdc_fixture


def write_taxonomy(path):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "taxonomy_version": "test_v1",
                "matching": {
                    "soft_threshold": 0.72,
                    "hard_threshold": 0.42,
                },
                "entries": {
                    "cucumber": {"fdc_id": 1, "category": "Vegetables"},
                    "cheeseburger": {"fdc_id": 2, "category": "Fast Foods"},
                },
            }
        )
    )
    return path


def test_v2_family_corrects_known_fdc_category_failures(tmp_path):
    path = tmp_path / "taxonomy-v2.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "taxonomy_version": "eval_taxonomy_v2",
                "matching": {"soft_threshold": 0.72, "hard_threshold": 0.42},
                "entries": {
                    "garden salad": {
                        "fdc_id": None,
                        "category": "Fats and Oils",
                        "family": "vegetable",
                    },
                    "mixed greens": {
                        "fdc_id": None,
                        "category": "Vegetables",
                        "family": "vegetable",
                    },
                    "berries": {
                        "fdc_id": None,
                        "category": "Beverages",
                        "family": "fruit",
                    },
                    "grapes": {
                        "fdc_id": None,
                        "category": "Fruits",
                        "family": "fruit",
                    },
                },
            }
        )
    )
    taxonomy = EvaluationTaxonomy(path)
    assert taxonomy.match_kind("garden salad", "mixed greens")[0] == "soft"
    assert taxonomy.match_kind("berries", "grapes")[0] == "soft"


def test_hard_mismatch_density_and_atwater_are_separate(tmp_path):
    database = tmp_path / "nutrition.sqlite"
    build_database([make_fdc_fixture(tmp_path / "source")], database)
    resolver = SQLiteNutritionResolver(database)
    ground_truth = {
        "total_calories": 20,
        "items": [
            {
                "name": "cucumber",
                "estimated_grams": 100,
                "calories": 20,
                "protein_g": 1,
                "fat_g": 0,
                "carbs_g": 4,
            }
        ],
    }
    prediction = {
        "total_calories": 150,
        "items": [
            {
                "name": "cheeseburger",
                "estimated_grams": 100,
                "calories": 150,
                "protein_g": 1,
                "fat_g": 14,
                "carbs_g": 1,
            }
        ],
    }
    row = score_dish(ground_truth, prediction, resolver)
    resolver.close()
    assert row["hmr"] == 1
    assert row["dvr"] == 1
    assert row["air"] == 0
    summary = summarize_dishes([row])
    assert summary["tier1_clean_dishes"] == 0
    assert summary["conditional_kcal_mae"] is None


def test_factored_item_from_profile_has_zero_internal_violations(tmp_path):
    database = tmp_path / "nutrition.sqlite"
    build_database([make_fdc_fixture(tmp_path / "source")], database)
    resolver = SQLiteNutritionResolver(database)
    profile = resolver.resolve("cucumber").profile
    prediction = {
        "total_calories": profile.kcal_per_100g,
        "items": [
            {
                "name": "cucumber",
                "estimated_grams": 100,
                "calories": profile.kcal_per_100g,
                "protein_g": profile.protein_per_100g,
                "fat_g": profile.fat_per_100g,
                "carbs_g": profile.carbs_per_100g,
            }
        ],
    }
    row = score_dish(prediction, prediction, resolver)
    resolver.close()
    assert row["hmr"] == 0
    assert row["idr"] == 1
    assert row["idp"] == 1
    assert row["dvr"] == 0
    assert row["air"] == 0
    assert row["tier1_clean"]


def test_frozen_taxonomy_decouples_hmr_from_runtime_resolver(tmp_path):
    database = tmp_path / "nutrition.sqlite"
    build_database([make_fdc_fixture(tmp_path / "source")], database)
    resolver = SQLiteNutritionResolver(database)
    taxonomy = EvaluationTaxonomy(write_taxonomy(tmp_path / "taxonomy.json"))
    cucumber_profile = resolver.resolve("cucumber").profile
    resolver.resolve = lambda name: type("Resolution", (), {"profile": cucumber_profile, "rung": "mutated"})()
    ground_truth = {
        "items": [{"name": "cucumber", "estimated_grams": 100}],
    }
    prediction = {
        "items": [{"name": "cheeseburger", "estimated_grams": 100}],
    }
    row = score_dish(
        ground_truth,
        prediction,
        resolver,
        taxonomy=taxonomy,
    )
    resolver.close()
    assert row["hmr"] == 1


def test_exact_match_is_not_consumed_by_larger_soft_match(tmp_path):
    database = tmp_path / "nutrition.sqlite"
    build_database([make_fdc_fixture(tmp_path / "source")], database)
    resolver = SQLiteNutritionResolver(database)
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "taxonomy_version": "test_v1",
                "matching": {"soft_threshold": 0.72, "hard_threshold": 0.42},
                "entries": {
                    "sausage": {"fdc_id": 1, "category": "processed meat"},
                    "bacon": {"fdc_id": 2, "category": "processed meat"},
                },
            }
        )
    )
    row = score_dish(
        {
            "items": [
                {"name": "sausage", "estimated_grams": 65, "share_pct": 65},
                {"name": "bacon", "estimated_grams": 35, "share_pct": 35},
            ]
        },
        {"items": [{"name": "bacon", "share_pct": 100}]},
        resolver,
        taxonomy=EvaluationTaxonomy(taxonomy_path),
    )
    resolver.close()
    assert row["idr"] == 0.35
    assert row["idp"] == 1
    assert row["hmr"] == 0.65


def test_assemble_audit_uses_scale_mass_and_database_macros(tmp_path):
    database = tmp_path / "nutrition.sqlite"
    build_database([make_fdc_fixture(tmp_path / "source")], database)
    identify = tmp_path / "identify.json"
    identify.write_text(
        json.dumps(
            {
                "paired_results": [
                    {
                        "id": "dish_1",
                        "group_id": "dish_1",
                        "status": "ok",
                        "ground_truth": {
                            "total_calories": 18.1,
                            "items": [
                                {
                                    "name": "cucumber",
                                    "estimated_grams": 100,
                                }
                            ],
                        },
                        "prediction": {
                            "not_food": False,
                            "container": "plate",
                            "items": [
                                {"name": "cucumber", "share_pct": 100}
                            ],
                        },
                    }
                ]
            }
        )
    )
    scale = tmp_path / "scale.json"
    scale.write_text(
        json.dumps(
            {
                "paired": [
                    {
                        "id": "nutrition5k:dish_1:overhead",
                        "p10_g": 80,
                        "p50_g": 100,
                        "p90_g": 120,
                    }
                ]
            }
        )
    )
    result = assemble_audit(
        identify,
        scale,
        database,
        write_taxonomy(tmp_path / "taxonomy.json"),
    )
    assert result["complete_predictions"] == 1
    assert result["group_hmr_mean"] == 0
    row = result["paired"][0]
    assert row["prediction"]["items"][0]["estimated_grams"] == 100
    assert row["air"] == 0
    assert row["dvr"] == 0


def test_assembly_comparison_uses_only_mutually_eligible_groups(tmp_path):
    def write_result(path, errors, incomplete=()):
        rows = [
            {
                "id": record_id,
                "group_id": group_id,
                "complete": record_id not in incomplete,
                "tier1_clean": True,
                "calorie_absolute_error": error,
            }
            for record_id, group_id, error in errors
        ]
        path.write_text(
            json.dumps(
                {
                    "predictions": len(rows),
                    "groups": len({row["group_id"] for row in rows}),
                    "complete_predictions": sum(row["complete"] for row in rows),
                    "complete_groups": len(
                        {
                            row["group_id"]
                            for row in rows
                            if row["complete"]
                        }
                    ),
                    "paired": rows,
                }
            )
        )
        return path

    left = write_result(
        tmp_path / "left.json",
        [("a1", "a", 10), ("a2", "a", 30), ("b1", "b", 50)],
    )
    right = write_result(
        tmp_path / "right.json",
        [("a1", "a", 20), ("a2", "a", 20), ("b1", "b", 1)],
        incomplete={"b1"},
    )
    result = compare_assemblies(left, right, bootstrap_samples=100)
    assert result["shared_eligible_predictions"] == 2
    assert result["shared_eligible_groups"] == 1
    assert result["left_conditional_kcal_mae"] == 20
    assert result["right_conditional_kcal_mae"] == 20
