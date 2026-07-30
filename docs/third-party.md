# Third-party license audit

This repository's own code is MIT-licensed (see `LICENSE`). It depends on the
following third-party projects, models, datasets, and data services. Code
dependencies are not vendored in this repo — Python packages come from PyPI via
`ml/setup.sh`, Swift packages are resolved by SwiftPM from the URLs in
`ios/SeeCal/Package.swift`, and the model/dataset are downloaded by the user.

Checked 2026-07-26 against each project's GitHub-reported license.

## Python (ml/, pinned in `ml/setup.sh`)

| Package | Version pinned | License | Notes |
|---|---|---|---|
| [mlx](https://github.com/ml-explore/mlx) | 0.32.0 | MIT | Apple's array framework |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | 0.31.3 | MIT | used only for `mlx_lm.fuse` experiments (attic/), not the trained/served path |
| [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | 0.6.7 | MIT | training + inference for Qwen3.5 |
| [transformers](https://github.com/huggingface/transformers) | 5.14.1 | Apache-2.0 | processor/tokenizer, pulled in by mlx-vlm |
| [datasets](https://github.com/huggingface/datasets) | 5.0.0 | Apache-2.0 | JSONL loading inside mlx-vlm's trainer |
| [torch](https://github.com/pytorch/pytorch) | 2.13.0 | BSD-3-Clause-style (custom, see repo `LICENSE`) | required at import time by the Qwen3VL processor; not used for compute |
| [torchvision](https://github.com/pytorch/vision) | 0.28.0 | BSD-3-Clause | same as above |
| pytest | latest at install time | MIT | test runner only |

## Swift (ios/, resolved via `ios/SeeCal/Package.swift` / `Package.resolved`)

| Package | License | Notes |
|---|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | MIT | on-device inference runtime |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | MIT | Qwen3.5 VLM support + `LoRAContainer` adapter loading |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | Apache-2.0 | transitive, via mlx-swift-lm |
| [swift-jinja](https://github.com/huggingface/swift-jinja) | Apache-2.0 | transitive — chat template rendering |
| [swift-collections](https://github.com/apple/swift-collections) | Apache-2.0 | transitive |
| [swift-numerics](https://github.com/apple/swift-numerics) | Apache-2.0 | transitive |
| [swift-crypto](https://github.com/apple/swift-crypto) | Apache-2.0 | transitive |
| [swift-asn1](https://github.com/apple/swift-asn1) | Apache-2.0 | transitive |
| [yyjson](https://github.com/ibireme/yyjson) | MIT | transitive, via swift-transformers |

GRDB is not used anywhere in `ios/` (checked — no import, no dependency
entry); nothing to audit there.

## Tooling

| Tool | License | Notes |
|---|---|---|
| [XcodeGen](https://github.com/yonaskolb/XcodeGen) | MIT | generates `ios/App/SeeCal.xcodeproj` at build time (`scripts/build.sh`); not committed, not a runtime dependency |

## Runtime data service

| Service | License / terms | Use and attribution |
|---|---|---|
| [Open Food Facts](https://world.openfoodfacts.org) product API | Database: [Open Database License 1.0 (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/); individual database contents: Database Contents License; product images: CC BY-SA (SeeCal does not request or display them) | Barcode lookup requests only product name, serving, ingredients text and nutrition. Responses are cached locally and the selected package-label values are snapshotted into the user's private meal log. The Settings data-sources card and each barcode result attribute Open Food Facts and link its [terms of use](https://world.openfoodfacts.org/terms-of-use). |

The product database is community-contributed and carries no assurance of
accuracy, completeness, or reliability. SeeCal therefore keeps all imported
nutrition editable, preserves its Open Food Facts provenance, and treats missing
required nutrition as incomplete instead of inventing zero values.

Checked 2026-07-28 against the Open Food Facts API and license documentation.

## Model and dataset (not code, not redistributed)

| Asset | License | Notes |
|---|---|---|
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (base) and [mlx-community/Qwen3.5-4B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) (MLX conversion used here) | Apache-2.0 | downloaded by the user via `ml/download_model.sh`; not committed |
| [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) | CC BY 4.0 | downloaded by the user via `ml/download_dataset.sh`; never committed. Attribution: Thames, Q. et al., "Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food," CVPR 2021 |
| [FoodSeg103](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1) via the pinned [Hugging Face mirror](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103) | Apache-2.0 | offline weak-label source for IDENTIFY only; `ml/download_foodseg103.py` pins revision `34e1208e14bc3595d544fc8c3f3c6673253fd9ef` and verifies all four source-object SHA-256 hashes. Images/masks are not committed or shipped |
| Food Recognition Benchmark 2022 v2.0 Deep Lake mirror (FRB) | CC BY 4.0 under the original AIcrowd 2022 release | user-provided 4.9 GB archive, SHA-256 `00afdf97a392e6baaf6e1bd13c5487917227b8e4030847f804c79ab165aa7879`; verified as 39,962 images / 76,491 annotations / 498 classes. A leakage-safe 5,000-image pilot and teacher-accepted visible names are used for IDENTIFY representation supervision only; FRB supplies no mass or nutrition labels and is never bundled |
| [NutritionVerse-Real v2](https://www.kaggle.com/datasets/nutritionverse/nutritionverse-real) | CC BY-NC-SA 4.0 | used in the explicitly non-commercial SCALE training track after the project owner confirmed that SeeCal will remain free and open source. The official Train split may be subdivided for training/validation; official Val remains a frozen OOD test. A separate permissive-only manifest excludes it. The pinned archive is downloaded by `ml/download_nutritionverse_real.sh`, remains gitignored, and is never bundled in the app |
| [Food Portion Benchmark](https://huggingface.co/datasets/issai/Food_Portion_Benchmark) | CC BY-NC 4.0 | used only in the explicitly non-commercial SCALE training track. `ml/make_fpb_scale_data.py` derives total image mass by summing its per-object gram labels and keeps related views in one group. The local download is pinned to revision `53fcacf4b9dbe24c1c6ffa5a2cdb9d8c502e482f`, remains gitignored, and is never bundled in the app. Attribution: Sanatbyek, A. et al., “A multitask deep learning model for food scene recognition and portion estimation—the Food Portion Benchmark (FPB) dataset,” IEEE Access, 2025 |
| [USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets/) Foundation 2026-04, SR Legacy 2018-04, and FNDDS 2021–2023 | U.S. government data; public domain under 17 U.S.C. §105 | downloaded by `ml/download_fdc.sh`; a pruned SQLite derivative is used by the factored pipeline but is not selected by the shipping v8 app path |
| Fine-tuned LoRA adapter weights produced by this pipeline (`ml/adapters_v5/` etc.) | inherits Qwen3.5's Apache-2.0 terms as a derivative of the base model | not committed to this repository |

## Summary

No copyleft (GPL/AGPL/LGPL) code dependencies were found anywhere in the stack.
Every direct and transitive code dependency checked is MIT or Apache-2.0,
except PyTorch's own custom BSD-style license (permissive, non-copyleft).
Open Food Facts runtime data remains subject to its separate ODbL/DbCL terms;
those terms do not relicense this repository's own MIT-licensed code.
NutritionVerse-Real and Food Portion Benchmark are isolated in an optional
non-commercial training track. Their source data is not shipped, but whether
trained weights are an adaptation for ShareAlike purposes is not settled here;
any public release of weights trained with NutritionVerse should preserve
attribution, identify modifications, use compatible terms, and receive a
license review. This is a project boundary, not legal advice.
