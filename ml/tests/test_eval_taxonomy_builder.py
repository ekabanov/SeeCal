import json

from make_eval_taxonomy import build_taxonomy
from make_fdc_db import build_database
from test_fdc_db import make_fdc_fixture


def test_reviewed_overrides_replace_resolver_mapping_and_record_provenance(tmp_path):
    database = tmp_path / "nutrition.sqlite"
    build_database([make_fdc_fixture(tmp_path / "source")], database)
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "taxonomy_version": "base",
                "entries": {
                    "regional cucumber plate": {
                        "fdc_id": None,
                        "category": None,
                        "family": None,
                    }
                },
            }
        )
    )
    overrides = tmp_path / "overrides.tsv"
    overrides.write_text(
        "name\tfdc_id\tcategory\tfamily\treview_note\n"
        "regional cucumber plate\t1\tVegetables\tvegetable\treviewed fixture\n"
    )

    result = build_taxonomy(
        database=database,
        json_paths=[],
        jsonl_paths=[],
        base_taxonomy_paths=[base],
        override_paths=[overrides],
        version="eval_taxonomy_v3",
    )

    assert result["entries"]["regional cucumber plate"] == {
        "fdc_id": 1,
        "category": "Vegetables",
        "family": "vegetable",
    }
    assert result["provenance"]["reviewed_override_count"] == 1
    assert result["provenance"]["reviewed_overrides"][0]["sha256"]
