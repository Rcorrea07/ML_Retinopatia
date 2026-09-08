# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Binary image classifier for diabetic retinopathy detection (fundus/retina photos) using transfer learning with EfficientNetB0 (Keras/TensorFlow).

The project is split in two halves that must stay in their roles:

- `src/retinopatia/` — all the real logic, as importable Python modules. This is what gets edited.
- `notebooks/Retinopatia_0_1.ipynb` — a **thin orchestrator** for Google Colab. Each code cell is one or two calls into the package. Do not move logic back into the notebook; when a cell would grow beyond a call plus its arguments, the logic belongs in a module.

All code, comments, docstrings, and print output are in Portuguese — preserve that convention when editing, including new module and function names.

## Execution environment

Editing happens locally (VS Code); **execution happens entirely inside Google Colab** because of the GPU.

- There is no local virtualenv, TensorFlow, or CUDA install, and none is wanted. Do not try to `pip install` the dependencies or run the pipeline locally, and do not assume any of the imports (`tensorflow`, `cv2`, `sklearn`, …) resolve on this machine.
- `requirements.txt` is documentation of what Colab provides — not a local install target.
- The code reaches Colab through git: the notebook's first cell clones/pulls `https://github.com/Rcorrea07/ML_Retinopatia.git` into `/content/ML_Retinopatia` and puts its `src/` on `sys.path`. `%autoreload 2` is enabled so a `git pull` picks up edits without restarting the runtime. **Changes only reach Colab once pushed** — mention that when finishing work the user will run there.
- Dataset comes from Drive: `/content/drive/MyDrive/Datasets/diabetic-retinopathy-detection.zip`. Extraction, the Otsu-cropped image cache, and the split CSVs are all cached (under `/content/...` and `/content/drive/MyDrive/ProjetoRetinopatia/splits`, keyed by `SEED`), and each step skips itself if its output already exists.

### Local verification

Only static checks are possible locally — the pipeline itself can be verified only by running the notebook in Colab:

```bash
python3 -m py_compile src/retinopatia/*.py
```

There is no test suite, linter config, or build step.

## Module map

| Module | Responsibility |
|---|---|
| `configuracao.py` | Every constant: seed, Colab paths, image size, batch size, augmentation dict, learning rates, epochs, callback patiences, checkpoint filenames, 0.90 sensitivity target |
| `utilitarios.py` | `configurar_sementes()` — seeds + best-effort `enable_op_determinism` |
| `ambiente_colab.py` | Colab-only glue: Drive mount, dataset extraction, file download |
| `conjunto_dados.py` | CSV load, filename validation, binary label, `patient_id`, patient-grouped split, split CSVs |
| `pre_processamento.py` | `recortar_otsu()` plus the visual-inspection and offline batch passes built on it |
| `pipeline_dados.py` | `ImageDataGenerator` train/val generators + scale sanity check |
| `modelo.py` | `criar_metricas()`, `construir_modelo()` |
| `treinamento.py` | Class weights, callback factory, Phase 1 training, Phase 2 fine-tuning |
| `avaliacao.py` | `avaliar_modelo()` — one function used by both phases |

## Domain logic worth knowing before editing

- **The target is binary, not the dataset's 5 classes.** `level` 0–4 is preserved as `level_original` and collapsed into `target`: 0 = no retinopathy, 1 = any grade 1–4.
- **The split is grouped by patient.** `StratifiedGroupKFold` on `patient_id` (parsed from `{id}_{left|right}.jpeg`) keeps both eyes of a patient on the same side; an assert enforces it. Stratification uses the original 5 grades, not the binary target.
- **Otsu crop is offline on purpose.** It finds the retina's bounding box, crops, resizes to 224×224, and caches to disk — it is not part of the Keras pipeline, because doing it per epoch cost ~40 min/epoch. Per-image status strings (`processada`, `existente`, `erro_leitura`, `mascara_vazia`, `recorte_invalido`, `erro_escrita`) are tallied and the run hard-fails if any image failed or is missing.
- **No rescale anywhere.** EfficientNet expects raw 0–255 input; `verificar_escala()` asserts `x.max() > 1.0` to catch an accidental normalization.
- **Validation generator must stay `shuffle=False`** so its order matches `val_data.classes` used in evaluation.
- **BatchNormalization stays frozen during fine-tuning** even though the backbone is unfrozen — easy to miss and easy to break. Phase 2 also requires the recompile (`lr=1e-5`) after touching `trainable`.
- **Decision threshold is not 0.5.** It is picked from the ROC curve as the point reaching at least 90% sensitivity with the lowest false positive rate — a medical screening tradeoff. Falls back to 0.5 only if no threshold qualifies.
- Class imbalance is handled with `compute_class_weight("balanced")` passed to `fit(class_weight=...)`.

## Conventions and gotchas

- **Hyperparameters live only in `configuracao.py`.** Don't reintroduce inline literals in modules or notebook cells; add a named constant instead.
- **No IPython magics in `.py` files.** `!unzip`/`!cat` were converted to `subprocess.run(..., shell=True, check=True)` in `ambiente_colab.py`; shell escapes only work in notebook cells.
- **Both phases share one evaluation function and one callback factory** (`avaliar_modelo`, `criar_callbacks` + the `criar_callbacks_fase1/2` wrappers), and the visual inspection shares `recortar_otsu` with the batch pass. These were duplicated in the original notebook — keep them unified, parameterizing instead of copying.
- In the notebook, `modelo` is the Keras model variable, so the module of the same name is never imported there; `construir_modelo` is imported directly to avoid the shadowing.
- Any change to preprocessing, the split, or the label mapping invalidates `/content/dataset_otimizado_otsu_v1` and the saved split CSVs — bump the version in the path or regenerate them rather than silently reusing stale caches.
- The original monolithic notebook (pre-refactor, with its run outputs) is preserved in git history at commit `b6d5b32`.
