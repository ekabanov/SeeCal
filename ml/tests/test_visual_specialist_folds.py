import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visual_specialist.folds import fold_for_group


def test_fold_assignment_is_deterministic_and_in_range():
    first = fold_for_group("dish_123", folds=5, seed=7)
    assert first == fold_for_group("dish_123", folds=5, seed=7)
    assert 0 <= first < 5
