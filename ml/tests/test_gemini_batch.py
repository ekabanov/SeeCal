import base64
from decimal import Decimal
import json
from pathlib import Path

from PIL import Image
import pytest

from teacher_labeling.budget import BudgetPolicy, ModelPrice, PricingCatalog
from teacher_labeling.gemini_batch import (
    BatchPreparationError,
    build_batch_request,
    prepare_batch_files,
)


def _record(image_path: Path, image_id: int = 1) -> dict:
    return {
        "schema_version": 1,
        "source": {
            "dataset": "food-recognition-benchmark-2022",
            "release": "v2.1",
            "license": "CC-BY-4.0",
            "image_id": image_id,
            "file_name": image_path.name,
            "annotations_sha256": "a" * 64,
        },
        "image": {
            "path": str(image_path),
            "width": 40,
            "height": 20,
        },
        "source_labels": [{"category_id": 1, "name": "rice"}],
        "source_annotation_count": 1,
        "loss_mask": {
            "semantic_food_fields": True,
            "calories": False,
            "grams": False,
            "macros": False,
            "hidden_ingredients": False,
        },
    }


def _image(path: Path) -> None:
    Image.new("RGB", (40, 20), "orange").save(path)


def _policy(cap: str = "20") -> BudgetPolicy:
    return BudgetPolicy(
        run_id="test",
        enabled_providers=frozenset({"google"}),
        provider_caps={"google": Decimal(cap), "openai": Decimal("0")},
        automatic_total_cap=Decimal(cap),
        authorized_total_cap=Decimal("25"),
        usage_buffer=Decimal("0.20"),
    )


def _pricing() -> PricingCatalog:
    prices = {
        model: ModelPrice(
            provider="google",
            mode="batch",
            input_per_million_usd=input_price,
            output_per_million_usd=output_price,
        )
        for model, input_price, output_price in (
            ("gemini-3.5-flash", Decimal("0.75"), Decimal("4.50")),
            ("gemini-3.5-flash-lite", Decimal("0.15"), Decimal("1.25")),
        )
    }
    return PricingCatalog(
        prices=prices,
        snapshot_hash="pricing-hash",
        source_url="https://example.test/pricing",
        retrieved_at="2026-07-28",
    )


def test_build_request_embeds_normalized_image_and_forbids_numeric_output(tmp_path):
    path = tmp_path / "image.png"
    _image(path)
    request, metadata = build_batch_request(_record(path), max_edge=16)

    config = request["request"]["generationConfig"]
    assert config["thinkingConfig"]["thinkingLevel"] == "MINIMAL"
    assert config["responseMimeType"] == "application/json"
    image_part = request["request"]["contents"][0]["parts"][0]["inlineData"]
    assert image_part["mimeType"] == "image/jpeg"
    assert base64.b64decode(image_part["data"]).startswith(b"\xff\xd8")
    assert metadata["image"]["normalized_width"] == 16
    schema_text = json.dumps(config["responseJsonSchema"])
    assert "calories" not in schema_text
    assert "grams" not in schema_text
    assert "macros" not in schema_text


def test_build_request_rejects_unmasked_numeric_fields(tmp_path):
    path = tmp_path / "image.png"
    _image(path)
    record = _record(path)
    record["loss_mask"]["calories"] = True

    with pytest.raises(BatchPreparationError, match="mask every"):
        build_batch_request(record)


def test_prepare_chunks_and_prices_both_models_without_paid_calls(tmp_path):
    manifest = tmp_path / "pilot.jsonl"
    records = []
    for image_id in range(1, 4):
        path = tmp_path / f"{image_id}.png"
        _image(path)
        records.append(_record(path, image_id))
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    plan_path = prepare_batch_files(
        manifest_path=manifest,
        output_dir=tmp_path / "batches",
        policy=_policy(),
        pricing=_pricing(),
        max_records_per_batch=2,
        record_limit=2,
        record_offset=1,
    )

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(plan["batches"]) == 1
    assert plan["record_limit"] == 2
    assert plan["record_offset"] == 1
    assert plan["paid_calls_submitted"] is False
    assert set(plan["models"]) == {
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    }
    assert plan["combined_buffered_estimate_usd"] == "0.007380"
    assert (plan_path.parent / "requests-001.jsonl").is_file()


def test_prepare_refuses_plan_above_automatic_cap(tmp_path):
    path = tmp_path / "image.png"
    _image(path)
    manifest = tmp_path / "pilot.jsonl"
    manifest.write_text(json.dumps(_record(path)) + "\n", encoding="utf-8")

    with pytest.raises(BatchPreparationError, match="automatic cap"):
        prepare_batch_files(
            manifest_path=manifest,
            output_dir=tmp_path / "batches",
            policy=_policy("0.000001"),
            pricing=_pricing(),
        )
