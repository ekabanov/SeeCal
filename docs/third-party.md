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
| Fine-tuned LoRA adapter weights produced by this pipeline (`ml/adapters_v5/` etc.) | inherits Qwen3.5's Apache-2.0 terms as a derivative of the base model | not committed to this repository |

## Summary

No copyleft (GPL/AGPL/LGPL) code dependencies were found anywhere in the stack.
Every direct and transitive code dependency checked is MIT or Apache-2.0,
except PyTorch's own custom BSD-style license (permissive, non-copyleft).
Open Food Facts runtime data remains subject to its separate ODbL/DbCL terms;
those terms do not relicense this repository's own MIT-licensed code.
