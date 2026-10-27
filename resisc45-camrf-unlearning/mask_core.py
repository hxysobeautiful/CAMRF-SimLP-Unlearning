# Adapted from the user's existing importance implementation; see PROVENANCE.json.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
CIFAR_NUM_CLASSES = 45
CIFAR_IMAGE_SIZE = 224
ALL_CLASSES = tuple(range(45))
CLASS_NAMES = tuple(str(x) for x in ALL_CLASSES)


def paired_bn_name(model: nn.Module, conv_name: str) -> str:
    if conv_name == 'conv1':
        bn_name = 'bn1'
    else:
        parts = conv_name.split('.')
        if len(parts) != 3 or parts[2] not in ('conv1', 'conv2'):
            raise ValueError(f'Unsupported Conv name: {conv_name}')
        bn_name = f'{parts[0]}.{parts[1]}.bn{parts[2][-1]}'
    conv = model.get_submodule(conv_name)
    bn = model.get_submodule(bn_name)
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f'{conv_name} is not Conv2d.')
    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError(f'{bn_name} is not BatchNorm2d.')
    if int(conv.out_channels) != int(bn.num_features):
        raise ValueError(f'{conv_name}: Conv out_channels={conv.out_channels}, BN features={bn.num_features}.')
    return bn_name

@torch.no_grad()
def _apply_resnet18_bn_hard_masks(model: nn.Module, keep_masks: Mapping[str, torch.Tensor]) -> Dict[str, Dict[str, object]]:
    details: Dict[str, Dict[str, object]] = {}
    for conv_name, raw_keep in keep_masks.items():
        bn_name = paired_bn_name(model, conv_name)
        bn = model.get_submodule(bn_name)
        keep = torch.as_tensor(raw_keep).detach().to(device=bn.weight.device, dtype=torch.float32)
        removed = torch.nonzero(keep == 0, as_tuple=False).flatten()
        if removed.numel() == 0:
            continue
        if bn.affine:
            idx = removed.to(bn.weight.device)
            bn.weight.data[idx] = 0
            bn.bias.data[idx] = 0
        if bn.track_running_stats:
            idx = removed.to(bn.running_mean.device)
            bn.running_mean.data[idx] = 0
            bn.running_var.data[idx] = 1
        details[bn_name] = {'paired_conv': conv_name, 'zeroed_count': int(removed.numel()), 'zeroed_indices': [int(v) for v in removed.cpu().tolist()]}
    return details

@torch.no_grad()
def apply_similarity_adaptive_pruning(model: nn.Module, adaptive_plan: Mapping[str, Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    report: Dict[str, Dict[str, object]] = {}
    keep_masks: Dict[str, torch.Tensor] = {}
    for name, plan in adaptive_plan.items():
        conv = model.get_submodule(name)
        selected = torch.tensor(plan['final_pruned_indices'], dtype=torch.long)
        keep = torch.ones(conv.out_channels, dtype=torch.float32)
        keep[selected] = 0.0
        mask = keep.to(device=conv.weight.device, dtype=conv.weight.dtype)
        conv.weight.mul_(mask[:, None, None, None])
        if conv.bias is not None:
            conv.bias.mul_(mask)
        keep_masks[name] = keep
        report[name] = {**dict(plan), 'channels': int(conv.out_channels), 'final_pruned_count': int(selected.numel()), 'actual_pruned_ratio': float(selected.numel() / conv.out_channels)}
    bn_report = _apply_resnet18_bn_hard_masks(model, keep_masks)
    total_bn_zeroed = sum((int(row['zeroed_count']) for row in bn_report.values()))
    for row in report.values():
        row['bn_pruning_enabled'] = True
        row['bn_modules_edited_total'] = len(bn_report)
        row['bn_coordinates_zeroed_total'] = total_bn_zeroed
    return report


def prepare_similarity_policy(model, profiles, targets, simlp=True, epsilon=1e-12):
    """Paper Sim-LP: cosine to the mean retained profile, on Original only."""
    from camrf_core import select_mask_compatible_conv_layers
    targets = tuple(dict.fromkeys(int(c) for c in targets))
    if not targets or any(c not in profiles for c in targets):
        raise ValueError('Forget classes must be a nonempty subset of profile classes.')
    retained = sorted(set(profiles) - set(targets))
    if not retained:
        raise ValueError('Sim-LP requires at least one retained class.')
    policy = {}
    for name, conv in select_mask_compatible_conv_layers(model).items():
        vectors = {}
        for c in profiles:
            scores = torch.as_tensor(profiles[c][name]).detach().cpu().double()
            if scores.shape != (conv.out_channels,) or not torch.isfinite(scores).all() or (scores < 0).any():
                raise ValueError(f'Invalid importance profile: class {c}, {name}.')
            vectors[c] = scores
        mean_retained = torch.stack([vectors[c] for c in retained]).mean(0)
        items = []
        for c in targets:
            scores = vectors[c]
            denominator = float(scores.norm() * mean_retained.norm())
            similarity = float(torch.dot(scores, mean_retained)) / denominator if denominator > epsilon else 0.
            similarity = min(1., max(0., similarity))
            items.append({'class': c, 'similarity': similarity,
                          'order': torch.argsort(scores, descending=True, stable=True)})
        policy[name] = {'channels': conv.out_channels, 'simlp': simlp,
                        'retained_order': torch.argsort(mean_retained, descending=True, stable=True),
                        'targets': items}
    return policy


def build_similarity_masks(policy, p, q):
    """Top_r selects floor(r*K) channels; ties use ascending channel index.

    Coefficients may exceed one; the resulting fractions saturate at one.
    Each target's protection is removed BEFORE taking the multi-class union.
    """
    import math
    if not all(math.isfinite(v) and v >= 0 for v in (p, q)):
        raise ValueError('Pruning/protection coefficients must be finite and nonnegative.')
    masks, report = {}, {}
    for name, row in policy.items():
        width = row['channels']
        removed = torch.zeros(width, dtype=torch.bool)
        steps = []
        for target in row['targets']:
            s = target['similarity']
            pr = min(p * (1-s), 1.) if row['simlp'] else min(p, 1.)
            qr = min(q * s, 1.) if row['simlp'] else 0.
            # Tolerance only corrects floating-point error at integer boundaries.
            k = min(width, math.floor(pr * width + 1e-12))
            h = min(width, math.floor(qr * width + 1e-12))
            chosen = torch.zeros(width, dtype=torch.bool)
            chosen[target['order'][:k]] = True
            chosen[row['retained_order'][:h]] = False
            removed |= chosen
            steps.append({'class': target['class'], 'similarity': s,
                          'pruning_ratio': pr, 'protection_ratio': qr,
                          'candidate_count': k, 'protection_count': h})
        masks[name] = ~removed
        report[name] = {'channels': width, 'steps': steps, 'per_target': steps,
                        'final_pruned_indices': removed.nonzero().flatten().tolist(),
                        'final_pruned_count': int(removed.sum()),
                        'final_pruned_ratio': float(removed.double().mean())}
    return masks, report
