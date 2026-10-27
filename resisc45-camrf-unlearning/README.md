# CAM-RF and Sim-LP for Class-Wise Machine Unlearning

Minimal research code for class-wise machine unlearning on NWPU-RESISC45 with ResNet-18. The repository retains the end-to-end experiment, CAM-RF channel scoring, Sim-LP mask construction, privacy evaluation, and baseline comparisons. Checkpoints, datasets, generated results, and one-off search scripts are intentionally excluded.

## Environment

- Python 3.9 or later
- PyTorch 2.1 or later with a matching torchvision build
- CUDA is recommended for full experiments; the smoke test runs on CPU

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Data

NWPU-RESISC45 contains 45 aerial-scene classes with 700 images per class. The preparation script downloads a pinned archive, verifies its SHA-256 digest, and checks the class folders:

```bash
python prepare_resisc45.py --data-root data/NWPU-RESISC45
```

The default `--data-protocol all_shared` uses ALL images for training, validation, attack calibration, and final testing: these four sample index lists are identical. Final metrics describe resubstitution on the full dataset, not held-out generalization. `split.json` records the protocol and exact indices.

The optional `--data-protocol disjoint` retains the former stratified 50/10/10/30 protocol. Use a NEW output directory when changing protocols. MIA is unavailable in all_shared because no nonmember samples remain. CMIA reuses the same calibration/test data and is labeled accordingly.

The dataset is not included in this repository. Please follow the dataset's CC BY-NC 4.0 terms and cite the original NWPU-RESISC45 publication.

## Main experiment

The following command trains the original models for seeds 42–44, computes channel importance, performs unlearning, evaluates privacy and utility, and writes summaries:

```bash
python run_resisc45.py \
  --data-root data/NWPU-RESISC45 \
  --output results_resisc45 \
  --forget-counts 1 3 5 \
  --device cuda
```

The implemented defaults are random ResNet-18 initialization, 100 epochs, SGD with momentum 0.9, learning rate 0.1, batch size 32, weight decay `5e-4`, and cosine annealing. Command-line options override these values.

Run individual stages with `--action train`, `importance`, `unlearn`, `evaluate`, or `summarize`. Use `python run_resisc45.py --help` for all options.

## Additional baselines

After the original checkpoints and split have been created:

```bash
python run_resisc45_baselines.py \
  --results-root results_resisc45 \
  --data-root data/NWPU-RESISC45 \
  --device cuda
```

`run_resisc45_baselines.py` implements disclosed adaptations of Negative Gradient, Fisher, DELETE, and SalUn. These adaptations should not be described as bit-for-bit reproductions of unavailable reference implementations.

FedOrtho is not included because its implementation was stored in a separate project. Do not claim that this repository reproduces FedOrtho unless that implementation and its exact configuration are added.

## Smoke test and unit tests

```bash
python make_smoke_data.py
python run_resisc45.py --data-root smoke_data --output results_smoke --device cpu --smoke
python -m unittest -v test_protocol.py test_download.py
```

Smoke outputs verify plumbing only and must not be reported as scientific results.

## Repository contents

- `run_resisc45.py`: end-to-end training, unlearning, and evaluation
- `camrf_core.py`: CAM-RF importance computation
- `mask_core.py`: similarity-aware layer-wise pruning
- `methods.py`: training, baselines, importance, and mask search
- `privacy.py`: MIA and calibrated MIA metrics
- `data.py`: deterministic dataset split and loaders
- `random_requests.py`: reproducible class-wise forget requests
- `run_resisc45_baselines.py`: additional baseline adaptations
- `prepare_resisc45.py`: verified dataset preparation
- `test_*.py`: protocol and download tests

## Reproducibility notes

- Reuse the same `split.json`, original checkpoint, request plan, and seeds for every method.
- Use a new output directory after changing any scientific setting; the code rejects incompatible caches.
- The reported online mask-application time excludes offline importance estimation, similarity computation, and mask construction. Their computation is accounted for separately.
- Large data files and model checkpoints are ignored by Git. Publish checkpoints through a release or archival service only if redistribution is permitted.
- Add the final paper citation and choose a repository license before publication.

## Paper method alignment

The complete method uses the supplied reference's LAST variant, `TFIDF_x_PaperCAMRF`: TF-IDF multiplied by the sample mean of input-space Grad-CAM within the receptive field of each channel's **ReLU activation peak**.

There is no fifth power, activation-magnitude weighting or spatial averaging over all feature positions. Grad-CAM is normalized on its native grid before upsampling. Both factors use the same deterministic training subset.

Sim-LP computes cosine(target importance, mean retained importance), with similarity zero when the product of norms is <= 1e-12, then clips to [0,1].

The fractions are min(p*(1-s),1) and min(q*s,1). Top-r uses floor(r*K) channels with stable index-order ties (including all-zero profiles). Each target's protected channels are excluded before taking the union.

All 17 selected convolutions, including conv1, participate. There is no 0.8 cap, forced minimum of one channel, stem exemption or depth multiplier.

Selected convolution filters/biases and paired BN affine coordinates are zeroed once; BN running statistics for those coordinates are reset. This zeros the selected residual branch channels; shortcut contributions still follow ResNet addition. No fine-tuning is performed for the proposed method.

`--cam-aggregation spatial_mean` and `--depth-scaling` are rejected for the paper method.

After this update, use a NEW output directory; old importance, mask searches and unlearned models used different definitions.

Legacy untagged CAM caches must be recomputed.

Run the following command for formula, regression and protocol checks:

```bash
python -m unittest -v test_paper_method.py test_protocol.py test_download.py
```

These are synthetic checks; publication accuracy and forgetting metrics require rerunning the experiment.

## Full dataset evaluation (default in this package)

All 31,500 RESISC45 images belong to each of train, validation, attack_cal, and test.

DataLoader training may apply random augmentation; evaluation uses deterministic resizing/normalization on the SAME underlying images.

Original training uses all classes; retained-only retraining/fine-tuning still excludes the requested forgotten classes.

Final F and R use all images of forgotten and retained classes respectively.

Default importance/search sample limits are zero, meaning all images; smoke mode deliberately caps work.

Result files record `data_protocol=all_shared` and `evaluation_scope=all_dataset_resubstitution`.

MIA values are null with a reason, and cannot be interpreted as zero. CMIA records that calibration and test share data.

These full-data results are not an independent test estimate.

No trained experiment results or dataset are included in the source package.