# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Binary image classifier for diabetic retinopathy detection (fundus/retina photos) using transfer learning with EfficientNet (Keras/TensorFlow; default EfficientNetB4 at 448×448, sized for Colab L4/A100). The v2 design choices come from the top Kaggle solutions of the DR 2015 and APTOS 2019 competitions (see README "Origem das escolhas").

The project is split in two halves that must stay in their roles:

- `src/retinopatia/` — all the real logic, as importable Python modules. This is what gets edited.
- `notebooks/Retinopatia_0_1.ipynb` — a **thin orchestrator** for Google Colab. Each code cell is one or two calls into the package. Do not move logic back into the notebook; when a cell would grow beyond a call plus its arguments, the logic belongs in a module.

All code, comments, docstrings, and print output are in Portuguese — preserve that convention when editing, including new module and function names.

## Execution environment

Editing happens locally (VS Code); **execution happens entirely inside Google Colab** because of the GPU.

- There is no local virtualenv, TensorFlow, or CUDA install, and none is wanted. Do not try to `pip install` the dependencies or run the pipeline locally, and do not assume any of the imports (`tensorflow`, `cv2`, `sklearn`, …) resolve on this machine.
- `requirements.txt` is documentation of what Colab provides — not a local install target.
- The code reaches Colab through git: the notebook's first cell clones `https://github.com/Rcorrea07/ML_Retinopatia.git` into `/content/ML_Retinopatia`, fetches and checks out `REVISAO_CODIGO` detached (default `origin/main`; a tag or commit reproduces an old run), and puts its `src/` on `sys.path`. `%autoreload 2` is enabled so re-running that cell picks up pushed edits without restarting the runtime. **Changes only reach Colab once pushed** — mention that when finishing work the user will run there.
- **Every training is traceable to its code.** `COMMIT_CODIGO` (short hash read from git at import) is part of `NOME_EXPERIMENTO`, so new code never overwrites an old run's folder. `utilitarios.salvar_configuracao()` writes `configuracao.json` (all constants + commit, uncommitted-changes flag, GPU, TF version) and `codigo_src.zip` (the package as it ran). `avaliacao.registrar_experimento()` appends one row per finished run to `resultados/registro_experimentos.csv`. Don't pull mid-experiment: the commit, and with autoreload the results folder, would change. Tag runs worth keeping (`treino-v2` = `de5cc37`, the code the first v2 run used).
- Each epoch's CSV history also gets the learning rate, peak GPU memory and system RAM (`MonitorRecursos`) — use it to size batch/resolution/backbone instead of guessing.
- Dataset comes from Drive: `/content/drive/MyDrive/Datasets/diabetic-retinopathy-detection.zip`. Extraction, the preprocessed image cache (`/content/dataset_retina_v3_448`), and the split CSVs (`/content/drive/MyDrive/ProjetoRetinopatia/dados_internos/splits/{treino,validacao,teste}_v2_seed_{SEED}.csv`) are all cached, and each step skips itself or reloads if its output already exists.
- Everything that must survive a Colab disconnect lives under `MyDrive/ProjetoRetinopatia/`: `modelos/` (one final `.keras` per run, named `NOME_MODELO` = backbone_size_preprocessing_commit, e.g. `B4_448_v3_1ccd2cb.keras`; the v2 model is `B4_448_v2_de5cc37.keras`), `resultados/<NOME_EXPERIMENTO>/` (checkpoints, `CSVLogger` histories, `predicoes_*.csv`, `configuracao.json`, `codigo_src.zip`) plus `registro_experimentos.csv`, `relatorios/` and `notebooks/` (the user's PDFs and executed notebook copies, not written by code), and `dados_internos/` with `splits/` and `cache/` (a zip of the preprocessed images per preprocessing version — when it exists, the notebook restores it and extracts only `trainLabels.csv` instead of the raw dataset).

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
| `utilitarios.py` | `configurar_sementes()` — seeds + best-effort `enable_op_determinism`; `configurar_precisao_mista()` |
| `ambiente_colab.py` | Colab-only glue: Drive mount, dataset extraction, file download |
| `conjunto_dados.py` | CSV load, filename validation, binary label, `patient_id`, patient-grouped train/val/test split, split CSVs (saved once, reloaded afterwards) |
| `pre_processamento.py` | `processar_retina()` (retina mask → square crop centered on the retina → Ben Graham color normalization inside the retina → circular mask) plus the visual-inspection and offline batch passes built on it |
| `pipeline_dados.py` | `tf.data` train/val/test pipelines, the augmentation block (`criar_aumento_dados`, used inside the model), scale sanity check, `medir_velocidade()` |
| `modelo.py` | `GeM` layer, `criar_metricas()`, `compilar_modelo()`, `construir_modelo()` (augmentation block after the input; two outputs: `doente`, `grau`) |
| `treinamento.py` | Class weights → per-sample weights, callback factory, Phase 1 training, Phase 2 fine-tuning with cosine LR schedule |
| `avaliacao.py` | TTA predictions of both heads, eye- and patient-level reports, prediction CSVs; `avaliar_modelo()` (validation, picks thresholds, used by both phases) and `avaliar_no_teste()` |
| `diagnostico.py` | Works on the prediction tables: sensitivity per grade, referable DR (grade ≥ 2), grade head vs binary head, patient aggregation options, patient-level bootstrap CIs, false negatives; `preparar_dados_v2()` rebuilds v2 images for val/test to evaluate the old v2 model |

## Domain logic worth knowing before editing

- **The decision is binary, but the model also learns the 5 grades.** `level` 0–4 is preserved as `level_original` and collapsed into `target`: 0 = no retinopathy, 1 = any grade 1–4. The model has two outputs: `doente` (sigmoid on `target`, BCE — this is the decision) and `grau` (linear regression on `level_original`, Huber loss, weight `PESO_PERDA_GRAU`) as an auxiliary head. Evaluation uses only `doente`.
- **Extra training data (`USAR_DADOS_EXTRA`).** The Kaggle zip has 88,702 images: 35,126 in `train` (the only ones the split uses) and 53,576 in `test`, whose labels are not in the zip — `carregar_dados_extra` downloads `retinopathy_solution.csv` (the organizers' file, same URL TensorFlow Datasets uses) to `MyDrive/Datasets/`. Those images only ever go into training (`ampliar_treino`); validation and test stay the original split so results remain comparable. They are processed into a separate folder/cache (`*_extra`), and the per-row `pasta_processada` column tells `criar_dataset` where each image lives. Experiment and model names get the `_extra` suffix. Raw extraction uses `7z` straight from the `*.zip.00x` parts and deletes them afterwards (the old `cat` join filled the Colab disk).
- **The split is grouped by patient and has three parts.** `StratifiedGroupKFold` (10 folds) on `patient_id` (parsed from `{id}_{left|right}.jpeg`): fold 0 = test, fold 1 = validation, rest = train. Asserts enforce that no patient crosses sets. Stratification uses the original 5 grades.
- **Validation picks, test measures.** Checkpoints and the decision thresholds are chosen on validation; the test set is only touched by `avaliar_no_teste()`. Never tune anything on test.
- **Preprocessing is offline on purpose.** `processar_retina` runs once and caches to disk — it is not part of the Keras pipeline, because doing it per epoch cost ~40 min/epoch. Per-image status strings (`processada`, `existente`, `mascara_suspeita`, `erro_leitura`, `mascara_vazia`, `recorte_invalido`, `erro_escrita`) are tallied and the run hard-fails if any image failed or is missing; `mascara_suspeita` (mask covers < `FRACAO_MINIMA_MASCARA` of the expected retina) is saved and listed, not a failure. `pipeline_dados` uses `tf.ensure_shape` so a cache built at another size fails loudly.
- **v2 vs v3 preprocessing.** v3 builds the mask with a fixed low threshold + largest component + convex hull (Otsu failed on dark photos) and computes Ben's local mean only inside the retina (`blur(img·m)/blur(m)`), which removes the bright/dark bands v2 produced where the camera crops the top and bottom of the circle. The v2 behavior stays reachable through the `metodo_mascara="otsu"` / `ben_respeita_mascara=False` options (`OPCOES_PRE_PROCESSAMENTO_V2`), which the diagnostic of the old model needs.
- **Background color is coupled to preprocessing.** With Ben normalization the neutral background is gray 128 (`COR_FUNDO`); the outside-retina fill, the circular mask and the rotation/zoom fill of the augmentation must all use it.
- **Augmentation runs inside the model, on the GPU.** It used to run in `tf.data` on the CPU and was the training bottleneck (≈1.2 s/batch in `fit` vs 0.1 s/batch in `predict`). The random layers only act with `training=True`, so `predict`/TTA see clean images. Don't move it back into `tf.data`.
- **No rescale anywhere.** EfficientNet expects raw 0–255 input; `verificar_escala()` asserts `x.max() > 1.0` to catch an accidental normalization.
- **Validation and test pipelines must never shuffle** — predictions are matched to `df_val`/`df_teste` rows by order (labels and `patient_id` come from the DataFrame).
- **BatchNormalization stays frozen during fine-tuning** even though the backbone is unfrozen — the backbone is called with `training=False` and the BN layers get `trainable=False`. Phase 2 also requires the recompile (`compilar_modelo`) after touching `trainable`.
- **Mixed precision** is enabled by `configurar_precisao_mista()` before the model is built. Output layers and `GeM` are forced to float32; augmentation layers are created with `dtype="float32"` so the backbone input stays full-precision 0–255.
- **Learning rate.** Phase 1 is a short head warm-up at a constant LR. Phase 2 uses `CosineDecay` with linear warm-up (`criar_taxa_cosseno`); there is no `ReduceLROnPlateau` (it monitored `val_loss`, which disagreed with the PR-AUC checkpoint monitor, and Keras rejects it with a schedule anyway).
- **Compare experiments on validation only.** The test set is evaluated once, on the chosen final model; the diagnostic comparisons (grade head, patient aggregation) are run on validation.
- **Decision threshold is not 0.5.** It is picked from the ROC curve as the point reaching at least 90% sensitivity with the lowest false positive rate — a medical screening tradeoff. Falls back to 0.5 only if no threshold qualifies. There are two thresholds: per eye and per patient (patient probability = max of the two eyes).
- Class imbalance is handled with `compute_class_weight("balanced")` turned into per-sample weights by `adicionar_pesos_amostra` — Keras `class_weight` does not work with multiple outputs.
- `GeM` is deliberately not registered with `register_keras_serializable` (autoreload would re-register and Keras rejects duplicates); load the saved model with `custom_objects={"GeM": GeM}`.

## Conventions and gotchas

- **Hyperparameters live only in `configuracao.py`.** Don't reintroduce inline literals in modules or notebook cells; add a named constant instead. The one exception is `BACKBONE`, which `configuracao.py` reads from the `RETINOPATIA_BACKBONE` environment variable set in the notebook's first cell, so one commit can train several backbones (the backbone is part of `NOME_EXPERIMENTO`).
- **No IPython magics in `.py` files.** `!unzip`/`!cat` were converted to `subprocess.run(..., shell=True, check=True)` in `ambiente_colab.py`; shell escapes only work in notebook cells.
- **Both phases share one evaluation function and one callback factory** (`avaliar_modelo`, `criar_callbacks` + the `criar_callbacks_fase1/2` wrappers) and one compile function (`compilar_modelo`), and the visual inspection shares `processar_retina` with the batch pass. These were duplicated in the original notebook — keep them unified, parameterizing instead of copying.
- In the notebook, `modelo` is the Keras model variable, so the module of the same name is never imported there; `construir_modelo` is imported directly to avoid the shadowing.
- Any change to preprocessing, the split, or the label mapping invalidates the image cache (local and the zip in `cache/`) and the saved split CSVs — bump `VERSAO_PRE_PROCESSAMENTO` / `VERSAO_SPLIT` in `configuracao.py` rather than silently reusing stale caches. The version is part of the cache paths and of `NOME_EXPERIMENTO`.
- `OPERACOES_DETERMINISTICAS` can be turned off if `enable_op_determinism` slows training or raises for a GPU op without a deterministic kernel.
- The original monolithic notebook (pre-refactor, with its run outputs) is preserved in git history at commit `b6d5b32`.
