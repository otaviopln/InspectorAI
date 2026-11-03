# Legacy code archive

This directory preserves historical InspectorAI implementations that are no
longer part of the supported application or training workflow.

## Archived package components

- `inspector/config.py`: configuration for the former heuristic classifier.
- `inspector/core/`: rule-based analyzer, basic R.C.F.O. formatter, and
  fixed clarification-question generator.
- `inspector/tests/`: tests associated with the retired components and an
  older duplicate dataset-preparation regression test.

## Archived scripts

- `scripts/train_colab.py`: previous trainer that reads legacy generated
  datasets and performs its own train/validation split.
- `scripts/dataset_builder.py`: former synthetic/mock dataset generator.
- `scripts/expand_dataset.py` and `expand_dataset_60k.py`: previous
  dataset-expansion experiments.
- `scripts/build_alpaca_dataset.py` and
  `scripts/build_openassistant_dataset.py`: earlier external-dataset
  conversion experiments.

## Important

The archived files are retained for historical reference only. They are not
maintained or tested as part of the current application. Their imports, paths,
dependencies, dataset schemas, and generated outputs may no longer match the
current project.

Use the supported workflow from the project root:

```powershell
python scripts/prepare_dataset.py
python scripts/train.py
python -m inspector
python inspector_gui.py
```
