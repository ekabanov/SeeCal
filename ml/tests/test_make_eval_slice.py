import json

from make_eval_slice import build_slice, filter_by_group_manifests


def test_slice_keeps_whole_groups_and_is_deterministic(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"id": f"{group}:{view}", "group_id": group}
        for group in ("a", "b", "c")
        for view in (1, 2)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    assert build_slice(source, first, groups=2, seed="fixed") == {
        "groups": 2,
        "records": 4,
    }
    build_slice(source, second, groups=2, seed="fixed")
    assert first.read_text() == second.read_text()
    chosen = [json.loads(line) for line in first.read_text().splitlines()]
    assert len({row["group_id"] for row in chosen}) == 2


def test_filter_by_group_manifests_can_join_dataset_specific_prefixes(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"id": "view-a", "group_id": "nv-real:dish_1"}),
                json.dumps({"id": "view-b", "group_id": "nv-real:dish_1"}),
                json.dumps({"id": "view-c", "group_id": "nv-real:dish_2"}),
            ]
        )
        + "\n"
    )
    groups = tmp_path / "groups.jsonl"
    groups.write_text(
        json.dumps(
            {
                "id": "scale:view-a",
                "group_id": "nutritionverse-real-v2:dish_1",
            }
        )
        + "\n"
    )
    output = tmp_path / "output.jsonl"

    result = filter_by_group_manifests(
        source,
        output,
        group_manifest_paths=[groups],
        group_tail=True,
    )

    assert result == {
        "groups": 1,
        "records": 2,
        "source_group_manifests": 1,
    }
    assert {
        json.loads(line)["id"] for line in output.read_text().splitlines()
    } == {"view-a", "view-b"}


def test_filter_can_keep_one_deterministic_record_per_group(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"id": f"{group}:{view}", "group_id": group}
        for group in ("a", "b")
        for view in (1, 2, 3)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    groups = tmp_path / "groups.jsonl"
    groups.write_text(
        "".join(
            json.dumps({"id": group, "group_id": group}) + "\n"
            for group in ("a", "b")
        )
    )
    output = tmp_path / "output.jsonl"

    result = filter_by_group_manifests(
        source,
        output,
        group_manifest_paths=[groups],
        one_per_group=True,
        seed="fixed",
    )

    assert result["groups"] == 2
    assert result["records"] == 2
