import sqlite3

import pytest

from prepared_alias_audit import _affected_mae, _copy_with_aliases


def test_candidate_alias_copy_never_mutates_source_database(tmp_path):
    source = tmp_path / "source.sqlite"
    candidate = tmp_path / "candidate.sqlite"
    connection = sqlite3.connect(source)
    connection.executescript(
        """
        CREATE TABLE foods (fdc_id INTEGER PRIMARY KEY);
        CREATE TABLE aliases (
            normalized_alias TEXT NOT NULL,
            fdc_id INTEGER NOT NULL,
            priority INTEGER NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY(normalized_alias, fdc_id)
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO foods VALUES (1);
        INSERT INTO foods VALUES (2);
        INSERT INTO aliases VALUES ('fish', 1, 200, 'current');
        """
    )
    connection.commit()
    connection.close()

    _copy_with_aliases(
        source,
        candidate,
        [{"name": "fish", "candidate_fdc_id": 2}],
    )

    source_connection = sqlite3.connect(source)
    candidate_connection = sqlite3.connect(candidate)
    try:
        assert source_connection.execute(
            "SELECT COUNT(*) FROM aliases"
        ).fetchone()[0] == 1
        assert candidate_connection.execute(
            """
            SELECT fdc_id, priority, source
            FROM aliases
            WHERE normalized_alias = 'fish'
            ORDER BY priority DESC
            """
        ).fetchall() == [
            (2, 300, "semantics_first_prepared_v1"),
            (1, 200, "current"),
        ]
    finally:
        source_connection.close()
        candidate_connection.close()


def test_affected_mae_uses_equal_group_weighting():
    result = {
        "paired": {
            "true_mass_exact_shares": [
                {
                    "group_id": "multi",
                    "complete": True,
                    "calorie_absolute_error": 10,
                },
                {
                    "group_id": "multi",
                    "complete": True,
                    "calorie_absolute_error": 30,
                },
                {
                    "group_id": "single",
                    "complete": True,
                    "calorie_absolute_error": 40,
                },
                {
                    "group_id": "ignored",
                    "complete": True,
                    "calorie_absolute_error": 100,
                },
            ]
        }
    }
    assert _affected_mae(result, {"multi", "single"}) == pytest.approx(30)
