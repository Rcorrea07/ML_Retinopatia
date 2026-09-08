# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repository contains a single Jupyter notebook, `Retinopatia_0_1.ipynb`, implementing a binary image classifier for diabetic retinopathy detection (fundus/retina photos) using transfer learning with EfficientNetB0 (Keras/TensorFlow). All code, comments, and print output are in Portuguese — preserve that convention when editing.

There is currently no other source code, package structure, tests, or build tooling in the repo — the notebook is the entire project.

## Execution environment

The project is designed to run **entirely inside Google Colab**, not locally:

- It mounts Google Drive (`google.colab.drive`) and reads a dataset zip from `/content/drive/MyDrive/Datasets/diabetic-retinopathy-detection.zip`.
- It relies on a Colab GPU runtime for training EfficientNetB0.
- Dataset extraction, offline image preprocessing, and split CSVs are cached under `/content/...` and `/content/drive/MyDrive/ProjetoRetinopatia/splits`, keyed by a fixed `SEED`, so re-running a cell after the first successful run skips redundant work (extraction and per-image preprocessing both check for existing output before redoing it).
- There is no `requirements.txt`, virtualenv, or local run/lint/test command — do not assume one exists or try to install/run TensorFlow locally unless the user asks for that explicitly.

To "run" this project in practice means: open the notebook in Colab and execute cells top to bottom.

## Architecture / notebook flow

The notebook is organized as a linear pipeline where each markdown section feeds state (DataFrames, generators, the compiled model) into the next section. Understanding it requires reading the cells in order:

1. **Setup & determinism** — sets `SEED = 42`, `PYTHONHASHSEED`, `tf.keras.utils.set_random_seed`, and best-effort `tf.config.experimental.enable_op_determinism` (wrapped in try/except since full determinism isn't always available).
2. **Drive mount & dataset extraction** — idempotent: skips unzip steps if the target image folder already exists.
3. **CSV loading & label engineering** — reads `trainLabels.csv`, validates the `{id}_{left|right}.jpeg` filename pattern (raises if violated), and **collapses the original 5-class severity (`level` 0–4) into a binary `target`**: 0 = no retinopathy, 1 = any retinopathy grade (1–4). This binary collapse is the actual modeling target — don't assume the model is multi-class.
4. **Patient-grouped stratified split** — uses `StratifiedGroupKFold` grouped by `patient_id` (derived from the image filename) so that both eyes of the same patient never end up split across train and validation. This is asserted explicitly (`pacientes_treino.isdisjoint(pacientes_val)`). Splits are saved as seed-versioned CSVs to Drive for reproducibility across sessions.
5. **Offline preprocessing (Otsu crop)** — a separate, one-time pass (not part of the Keras data pipeline) that Otsu-thresholds each image to find the retina's bounding box, crops, resizes to 224×224, and writes the result to a cache folder. This exists specifically to avoid paying the crop cost every epoch (the notebook comment notes this used to cost ~40 min/epoch). Per-image status is tracked (`processada`, `existente`, `erro_leitura`, `mascara_vazia`, `recorte_invalido`, `erro_escrita`) and the run hard-fails (`raise RuntimeError`) if any image failed or is missing afterward.
6. **Keras data generators** — `ImageDataGenerator.flow_from_dataframe` for train (with augmentation: rotation/flip/zoom) and validation (no augmentation, `shuffle=False` so validation order matches `val_data.classes` for later evaluation). Deliberately **no rescale** — EfficientNet expects raw 0–255 input, and the notebook asserts `x.max() > 1.0` to catch accidental normalization.
7. **Class weights** — `sklearn.utils.class_weight.compute_class_weight(balanced)` on the binary target, passed into `model.fit(class_weight=...)` to handle class imbalance (only classes `{0, 1}` are expected; anything else raises).
8. **Model architecture** — EfficientNetB0 (`include_top=False`, ImageNet weights) as a frozen feature extractor + `GlobalAveragePooling2D` → `Dense(256, relu)` → `Dropout(0.5)` → `Dense(1, sigmoid)`. Metrics are built via a `criar_metricas()` factory (accuracy, precision, sensitivity/recall, ROC-AUC, PR-AUC) so a fresh set of metric objects is created on each `compile()`.
9. **Two-phase training**:
   - **Phase 1**: backbone frozen, `Adam(lr=1e-3)`, callbacks monitor `val_pr_auc` (checkpoint + early stopping) and `val_loss` (`ReduceLROnPlateau`).
   - **Fine-tuning (Phase 2)**: backbone unfrozen, but **BatchNormalization layers are explicitly kept frozen** (`layer.trainable = False` for every `BatchNormalization` instance) — a standard but easy-to-miss requirement when fine-tuning EfficientNet. Recompiled with a much lower `Adam(lr=1e-5)` and separate checkpoint/early-stopping/reduce-LR callbacks with longer patience.
   - Each phase reloads its own best checkpoint weights (`*.weights.h5`) after training before evaluation.
10. **Evaluation (run once per phase, nearly identical code both times)** — plots train/val accuracy and loss curves; then, instead of a fixed 0.5 decision threshold, picks a threshold from the ROC curve that achieves at least **90% sensitivity/recall** (`sensibilidade_desejada = 0.90`) while minimizing false positive rate — appropriate for a medical screening use case where missed positives are costlier than false alarms. Prints a confusion matrix and `classification_report` at that threshold.
11. **Final model export** — saves the fine-tuned model (`modelo_retinopatia_final.keras`) and downloads it via `google.colab.files.download`.

## Working with this codebase

- Keep all identifiers, comments, and printed messages in Portuguese, matching the existing style.
- The two evaluation blocks (post-Phase-1 and post-Phase-2) and the Otsu-crop logic (visual inspection cell vs. offline preprocessing cell) are near-duplicates of each other in the current notebook — be aware of this when making changes so a fix or tweak doesn't need to be applied twice by hand.
- Hyperparameters (seed, image size, batch size, learning rates, epochs, callback patiences, the 0.90 sensitivity target) are currently hardcoded inline per cell rather than centralized — check every occurrence when changing one of these values.
- Any change to preprocessing, the train/val split, or the binary label mapping invalidates the cached files under `/content/dataset_otimizado_otsu_v1` and the saved split CSVs — those would need to be regenerated (or the seed/version in the path bumped) rather than silently reused.
