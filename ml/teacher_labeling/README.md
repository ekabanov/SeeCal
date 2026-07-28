# Teacher-labeling pilot

All commands run from `ml/`. Paid calls are not part of `prepare`; submission
will be a separate command guarded by the append-only budget ledger.

```bash
# One-time, isolated environment (do not modify .venv):
python3 -m venv .venv-teacher
.venv-teacher/bin/pip install -r teacher_labeling/requirements.lock

# Snapshot public terms/provenance (already captured once on this machine):
.venv/bin/python -m teacher_labeling.snapshot_sources

# After extracting FRB 2022 v2.1, validate it and select the 5k pilot:
.venv/bin/python -m teacher_labeling.food_recognition_2022 \
  --annotations datasets/food_recognition_2022/train/annotations.json \
  --images-root datasets/food_recognition_2022/train/images \
  --out datasets/food_recognition_2022/manifests/pilot-5000.jsonl \
  --expected

# The downloaded 2022 mirror on this machine is v2.0 in Deep Lake 2.2.2
# format. Decode it only with the isolated Python 3.9 reader:
/usr/bin/python3 -m venv .venv-frb
.venv-frb/bin/pip install -r teacher_labeling/requirements-frb-legacy.lock
.venv-frb/bin/python -m teacher_labeling.deeplake_export metadata \
  --dataset datasets/food_recognition_2022/source_archive/hub/train \
  --out datasets/food_recognition_2022/derived/train-v2.0-coco.json \
  --archive-sha256 00afdf97a392e6baaf6e1bd13c5487917227b8e4030847f804c79ab165aa7879

# Build reviewable Gemini JSONL chunks; this makes no API calls:
.venv/bin/python -m teacher_labeling.gemini_batch \
  --manifest datasets/food_recognition_2022/manifests/pilot-5000.jsonl \
  --output-dir runs/teacher-labeling/gemini-5k-pilot/prepared

# A paid batch requires the explicit acknowledgement flag and is one job only:
.venv-teacher/bin/python -m teacher_labeling.gemini_submit \
  --plan runs/teacher-labeling/gemini-5k-pilot/bakeoff-325/batch-plan.json \
  --model gemini-3.5-flash-lite \
  --confirm-paid-submit

# Inspect effective caps and settled/open spend:
.venv/bin/python -m teacher_labeling.cli status
```

The committed run config enables Google only. It narrows the broader secret
authorization to a $25 run cap with a $5 safety reserve, so automation stops at
$20 committed spend. The OpenAI key, if present in the secret file for other
work, is ignored by this run.
