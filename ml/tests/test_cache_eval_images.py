import json

from PIL import Image

from cache_eval_images import cache


def test_cache_resizes_and_rewrites_image_references(tmp_path):
    image = tmp_path / "large.jpg"
    Image.new("RGB", (400, 200), "red").save(image)
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "one",
                "images": ["large.jpg"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "large.jpg"},
                            {"type": "text", "text": "prompt"},
                        ],
                    }
                ],
            }
        )
        + "\n"
    )
    output = tmp_path / "output.jsonl"
    result = cache(
        source,
        output,
        cache_dir=tmp_path / "cache",
        ml_root=tmp_path,
        max_edge=100,
    )
    record = json.loads(output.read_text())
    cached = tmp_path / record["images"][0]
    assert Image.open(cached).size == (100, 50)
    assert record["messages"][0]["content"][0]["image"] == record["images"][0]
    assert result == {
        "records": 1,
        "images_created": 1,
        "images_reused": 0,
        "max_edge": 100,
    }


def test_cache_supports_scale_image_path_records(tmp_path):
    image = tmp_path / "large.jpg"
    Image.new("RGB", (200, 100), "green").save(image)
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "scale:one",
                "group_id": "scale:one",
                "image_path": "large.jpg",
                "source": "test",
                "total_mass_g": 123.0,
            }
        )
        + "\n"
    )
    output = tmp_path / "output.jsonl"

    result = cache(
        source,
        output,
        cache_dir=tmp_path / "cache",
        ml_root=tmp_path,
        max_edge=64,
    )
    record = json.loads(output.read_text())

    assert "images" not in record
    assert Image.open(tmp_path / record["image_path"]).size == (64, 32)
    assert result == {
        "records": 1,
        "images_created": 1,
        "images_reused": 0,
        "max_edge": 64,
    }


def test_cache_can_reuse_images_already_below_limit(tmp_path):
    image = tmp_path / "small.jpg"
    Image.new("RGB", (32, 24), "blue").save(image)
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"id": "one", "images": ["small.jpg"]}) + "\n"
    )
    output = tmp_path / "output.jsonl"

    result = cache(
        source,
        output,
        cache_dir=tmp_path / "cache",
        ml_root=tmp_path,
        max_edge=64,
        reuse_small_originals=True,
    )
    record = json.loads(output.read_text())

    assert record["images"] == ["small.jpg"]
    assert result == {
        "records": 1,
        "images_created": 0,
        "images_reused": 1,
        "max_edge": 64,
    }
