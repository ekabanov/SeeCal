import json

import numpy as np
from PIL import Image
import pytest

from download_foodseg103 import verify_file
from make_identify_data import foodseg_records


def test_foodseg_mask_pixels_become_coarse_shares(tmp_path):
    root = tmp_path / "foodseg"
    images = root / "train" / "images"
    masks = root / "train" / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    Image.new("RGB", (10, 10), "white").save(images / "00000001.jpg")
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, :7] = 1
    mask[:, 7:] = 2
    Image.fromarray(mask).save(masks / "00000001.png")
    (root / "metadata.json").write_text(
        json.dumps({"categories": ["background", "rice", "beans"]})
    )
    records = foodseg_records(root, "train", ml_root=tmp_path)
    payload = json.loads(records[0]["messages"][-1]["content"][0]["text"])
    assert payload["items"] == [
        {"name": "rice", "portion_units": 14},
        {"name": "beans", "portion_units": 6},
    ]


def test_foodseg_source_checksum_is_enforced(tmp_path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"pinned source")
    verify_file(
        source,
        "caeef37a76ff7885d011af508bf4cfe89c190cbc68391e57c4e5d61b95149acb",
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_file(source, "0" * 64)
