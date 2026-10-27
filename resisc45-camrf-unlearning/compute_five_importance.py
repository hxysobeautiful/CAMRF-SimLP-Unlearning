"""Compute five class-wise channel-importance variants for NWPU-RESISC45 / ResNet-18.

Requires camrf_core.py alongside this script. No training or pruning.
By default ImageFolder includes ALL images under --data-root; use a train-only
ImageFolder root to avoid validation/test leakage.

Variants:
  TF_IDF: original FUDP TF-IDF.
  TFIDF_x_PeakRFCAM: RF-CAM at argmax of raw convolution output.
  TFIDF_x_PeakRFCAM5: (RF-CAM at argmax of ReLU output) ** 5.
  TFIDF_x_PaperCAMRF: RF-CAM at argmax of ReLU output, no exponent (paper Eq. 8-11).
  TFIDF_x_MeanRFCAM: spatial mean of RF-CAM over ALL feature positions
                        (no activation-magnitude weighting).
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import camrf_core as core

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / 'data/NWPU-RESISC45'
METHODS = ('TF_IDF', 'TFIDF_x_PeakRFCAM', 'TFIDF_x_PeakRFCAM5', 'TFIDF_x_MeanRFCAM', 'TFIDF_x_PaperCAMRF')


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--checkpoint', type=Path, default=HERE / 'results_all_shared/seed_42/Original/best.pth')
    p.add_argument('--output', type=Path, default=HERE / 'importance_five_methods')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--mean', type=float, nargs=3, default=[0.485, 0.456, 0.406])
    p.add_argument('--std', type=float, nargs=3, default=[0.229, 0.224, 0.225])
    p.add_argument('--max-samples-per-class', type=int, default=0,
                   help='0=all images in each class; otherwise first N images per class')
    return p.parse_args()


def load_model(path, dataset_classes, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError('Expected best.pth to contain a checkpoint/state_dict dictionary.')
    ckpt_classes = ckpt.get('classes')
    if ckpt_classes is not None and list(ckpt_classes) != list(dataset_classes):
        raise ValueError('Checkpoint class ordering differs from ImageFolder ordering.')
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    if not isinstance(state, dict):
        raise ValueError('Cannot find a model state_dict in best.pth.')
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model = models.resnet18(weights=None, num_classes=len(dataset_classes))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), ckpt


def main():
    a = get_args()
    if not a.data_root.is_dir():
        raise FileNotFoundError(f'Dataset not found: {a.data_root}')
    if not a.checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {a.checkpoint}')
    if a.max_samples_per_class < 0 or a.batch_size <= 0 or a.workers < 0:
        raise ValueError('Invalid sample limit, batch size, or worker count.')
    a.output.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((a.image_size, a.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=a.mean, std=a.std),
    ])
    ds = datasets.ImageFolder(str(a.data_root), transform=transform)
    if len(ds.classes) != 45:
        raise ValueError(f'Expected 45 ImageFolder class folders, found {len(ds.classes)}. '
                         'For train/test subfolders point --data-root to train.')
    model, _ = load_model(a.checkpoint, ds.classes, a.device)
    core.CIFAR_NUM_CLASSES = len(ds.classes)
    core.CIFAR_IMAGE_SIZE = a.image_size
    core.ALL_CLASSES = tuple(range(len(ds.classes)))
    classes = tuple(range(len(ds.classes)))
    indices = {c: [] for c in classes}
    for idx, (_, cls) in enumerate(ds.samples):
        if not a.max_samples_per_class or len(indices[cls]) < a.max_samples_per_class:
            indices[cls].append(idx)
    if any(not indices[c] for c in classes):
        raise ValueError('One or more classes have no images.')
    selected = [idx for c in classes for idx in indices[c]]
    loader_args = dict(batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                       pin_memory=a.device.startswith('cuda'))

    print('Computing FUDP TF-IDF on the SAME selected images as CAM-RF...', flush=True)
    all_loader = DataLoader(Subset(ds, selected), **loader_args)
    activations = core.compute_class_channel_activations(model, all_loader, a.device, classes)
    tfidf = {c: core.compute_fudp_tfidf_importance(activations, classes, c) for c in classes}

    layers = core.select_mask_compatible_conv_layers(model)
    rf_table = core.compute_resnet18_rf(model)
    cam_name, cam_layer, cam_shape = core.select_spatial_gradcam_layer(model, input_size=a.image_size)
    print(f'Grad-CAM target: {cam_name} {cam_shape}', flush=True)

    peak = {}
    peak5 = {}
    paper_camrf = {}
    mean_rf = {}
    rows = []
    for c in classes:
        print(f'[{c+1}/45] {ds.classes[c]}: {len(indices[c])} images', flush=True)
        collector = core.ActivationCollector(layers)
        gradcam = core.GradCAM(model, cam_layer)
        totals_peak = {n: torch.zeros(m.out_channels, dtype=torch.float64) for n, m in layers.items()}
        totals_peak5 = {n: torch.zeros(m.out_channels, dtype=torch.float64) for n, m in layers.items()}
        totals_paper = {n: torch.zeros(m.out_channels, dtype=torch.float64) for n, m in layers.items()}
        totals_mean = {n: torch.zeros(m.out_channels, dtype=torch.float64) for n, m in layers.items()}
        count = 0
        try:
            class_loader = DataLoader(Subset(ds, indices[c]), **loader_args)
            for images, _ in class_loader:
                images = images.to(a.device, non_blocking=True)
                collector.clear()
                cam = gradcam.generate(images, c, tuple(images.shape[-2:]))
                if set(collector.outputs) != set(layers):
                    raise RuntimeError('Missing activation hooks for one or more layers.')
                for name, feature in collector.outputs.items():
                    rf_cam = core.mean_cam_in_receptive_fields(
                        cam, feature.shape[-2], feature.shape[-1], rf_table[name]
                    ).clamp_min(0)
                    flat_rf = rf_cam.flatten(1)
                    # Variant 2: raw activation peak; NO exponent.
                    raw_peak_idx = feature.flatten(2).argmax(dim=2)
                    raw_peak_cam = flat_rf.gather(1, raw_peak_idx)
                    # Variant 3: ReLU activation peak; exponent BEFORE averaging images.
                    relu_peak_idx = feature.relu().flatten(2).argmax(dim=2)
                    relu_peak_cam = flat_rf.gather(1, relu_peak_idx)
                    relu_peak_cam5 = relu_peak_cam.double().pow(5)
                    # Variant 4: all spatial responses, unweighted mean of RF-CAM.
                    # Since RF-CAM is spatial (shared by channels), this factor is
                    # identical for all channels in a given layer/image.
                    mean_cam = flat_rf.mean(dim=1, keepdim=True).expand(-1, feature.shape[1])
                    totals_peak[name] += raw_peak_cam.double().sum(dim=0).cpu()
                    totals_peak5[name] += relu_peak_cam5.sum(dim=0).cpu()
                    totals_paper[name] += relu_peak_cam.double().sum(dim=0).cpu()
                    totals_mean[name] += mean_cam.double().sum(dim=0).cpu()
                count += images.size(0)
                collector.clear()
        finally:
            gradcam.close()
            collector.close()
        if count != len(indices[c]):
            raise RuntimeError(f'Class {c}: expected {len(indices[c])}, processed {count}.')
        peak[c] = core.combine_fudp_and_cam_rf(
            tfidf[c], {n: (v / count).float() for n, v in totals_peak.items()})
        peak5[c] = core.combine_fudp_and_cam_rf(
            tfidf[c], {n: (v / count).float() for n, v in totals_peak5.items()})
        paper_camrf[c] = core.combine_fudp_and_cam_rf(
            tfidf[c], {n: (v / count).float() for n, v in totals_paper.items()})
        mean_rf[c] = core.combine_fudp_and_cam_rf(
            tfidf[c], {n: (v / count).float() for n, v in totals_mean.items()})
        for name in layers:
            rows.append({'class_id': c, 'class_name': ds.classes[c], 'layer': name,
                         'channels': layers[name].out_channels, 'samples': count})
        print(f'  Completed: {ds.classes[c]}', flush=True)

    result = {
        'TF_IDF': tfidf,
        'TFIDF_x_PeakRFCAM': peak,
        'TFIDF_x_PeakRFCAM5': peak5,
        'TFIDF_x_MeanRFCAM': mean_rf,
        'TFIDF_x_PaperCAMRF': paper_camrf,
        'classes': ds.classes,
        'class_to_idx': ds.class_to_idx,
        'sample_counts': {ds.classes[c]: len(indices[c]) for c in classes},
        'checkpoint': str(a.checkpoint),
        'data_root': str(a.data_root),
        'definitions': {
            'TF_IDF': 'Original FUDP TF-IDF',
            'TFIDF_x_PeakRFCAM': 'TFIDF * mean_x[RF-CAM at argmax raw activation]',
            'TFIDF_x_PeakRFCAM5': 'TFIDF * mean_x[(RF-CAM at argmax ReLU activation)^5]',
            'TFIDF_x_MeanRFCAM': 'TFIDF * mean_x[mean_all_spatial_positions(RF-CAM)]',
            'TFIDF_x_PaperCAMRF': 'TFIDF * mean_x[RF-CAM at argmax ReLU activation]; paper Eq. (8)-(11)',
        },
        'image_size': a.image_size, 'mean': a.mean, 'std': a.std,
        'gradcam_layer': cam_name,
    }
    output_path = a.output / 'importance_all_45_classes.pth'
    torch.save(result, output_path)
    for c in classes:
        for method in METHODS:
            torch.save(result[method][c], a.output / f'{ds.classes[c]}_{method}.pth')
    with (a.output / 'index.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output / 'class_names.json').write_text(json.dumps(ds.classes, indent=2), encoding='utf-8')
    print('Saved:', output_path, flush=True)
    print('Total: 45 classes x 5 methods; no pruning, no training.', flush=True)


if __name__ == '__main__':
    main()
