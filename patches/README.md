# mlx-vlm local patches (historical — for the legacy .venv only)

These are the hand-applied fixes to **mlx-vlm 0.4.0 @ commit 718f69e9** installed in
`.venv/` (Python 3.14). They correspond to CLAUDE.md issues 14–19 and were verified
against pristine upstream on 2026-07-26 (all bugs real at that commit; batch smoke
test passed).

- `*.py.patched` — full patched files as installed in `.venv/lib/python3.14/site-packages/mlx_vlm/`
  (`datasets.py`, `sft_trainer.py` live under `trainer/`).
- `*.diff` — unified diffs vs pristine 718f69e9.

**Status: obsolete for new training.** Upstream mlx-vlm 0.6.7 (July 2026) fixes this
entire bug family (issue #824 / PR #826 / v0.6.4 `completion_mask`). New training uses
`.venv-vlm067/` with an unpatched install — but note two migration traps:
1. 0.6.7 reads images ONLY from a top-level `"images"` JSONL field (content-embedded
   image entries are ignored → silent text-only training). Dataset must include it.
2. LoRA scale convention changed to alpha/rank in v0.5.0 (rank16/alpha32: 32 → 2).

Keep this directory so `.venv/` can be reconstructed if the old stack is ever needed
(e.g. to reproduce adapters_v4). To reapply: `pip install git+https://github.com/Blaizzy/mlx-vlm.git@718f69e9`
then copy the `.py.patched` files over the installed ones.
