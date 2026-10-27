"""Shared seed-specific request selection; no unlearning or metric implementation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import write_json

POLICY = 'seed_random_train_similarity_v1'


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def groups_for_seed(plan, seed, classes=None):
    if plan.get('selection_policy') != POLICY or 'by_seed' not in plan:
        raise ValueError('Expected a seed-specific random request_plan.json; prepare a NEW random-results directory.')
    try:
        groups = plan['by_seed'][str(seed)]['groups']
    except KeyError as error:
        raise ValueError(f'No shared requests for seed {seed}.') from error
    for group in groups:
        ids = group['classes']
        if (not ids or any(type(c) is not int or c < 0 for c in ids)
                or len(ids) != len(set(ids)) or group['name'] != f'forget_{len(ids)}'):
            raise ValueError('Invalid shared request.')
        if classes is not None and (max(ids) >= len(classes) or len(ids) > len(classes)-2
                or group['class_names'] != [classes[c] for c in ids]):
            raise ValueError('Shared request class mapping differs from checkpoint.')
    return groups


def reference_checkpoint(output, request_seed, group, model_seed):
    output = Path(output)
    if request_seed == model_seed:
        return output / f'seed_{model_seed}/{group["name"]}/Retrain/best.pth'
    return output / f'cmia_references/request_seed_{request_seed}/{group["name"]}/seed_{model_seed}/Retrain/best.pth'


def choose_anchor(similarity, seed, counts, used_anchors, used_sets):
    """Seeded permutation; reject repeated anchors or unordered request sets.

    Seeds are processed in sorted order, independent of CLI ordering. Each
    candidate's neighbors are ALWAYS its most similar classes; no substitutions.
    """
    for candidate in np.random.default_rng(seed).permutation(len(similarity)):
        anchor = int(candidate)
        if anchor in used_anchors:
            continue
        order = [anchor] + sorted((c for c in range(len(similarity)) if c != anchor),
                                 key=lambda c: (-float(similarity[anchor, c]), c))
        keys = {n: tuple(sorted(order[:n])) for n in counts}
        if any(keys[n] in used_sets[n] for n in counts):
            continue
        used_anchors.add(anchor)
        for n in counts:
            used_sets[n].add(keys[n])
        return anchor, order
    raise ValueError('Cannot obtain distinct top-similarity request combinations for all seeds.')


@torch.inference_mode()
def training_similarity(model, data, device):
    model.eval()
    features = {}
    sums = counts = None
    hook = model.avgpool.register_forward_hook(lambda m, i, o: features.update(value=o.flatten(1)))
    try:
        for x, y in data.loader('train'):  # deterministic eval transform of TRAIN images only
            model(x.to(device))
            value = features['value']
            y = y.to(device, dtype=torch.long)
            if sums is None:
                sums = value.new_zeros((len(data.classes), value.shape[1]))
                counts = value.new_zeros(len(data.classes))
            sums.index_add_(0, y, value)
            counts.index_add_(0, y, torch.ones(len(y), device=device))
    finally:
        hook.remove()
    if sums is None or (counts <= 0).any():
        raise ValueError('Every class must have training samples for similarity selection.')
    centroids = F.normalize(sums / counts[:, None], dim=1)
    matrix = (centroids @ centroids.T).cpu().numpy()
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite training similarity.')
    return matrix, counts.cpu().tolist()


def build_plan(data, args, output, load_model):
    output = Path(output)
    path = output / 'request_plan.json'
    identities = {str(seed): file_sha256(output / f'seed_{seed}/Original/best.pth') for seed in sorted(args.seeds)}
    identity = {'selection_policy': POLICY, 'seeds': sorted(args.seeds),
                'forget_counts': list(args.forget_counts), 'classes': data.classes,
                'fingerprint': data.fingerprint, 'split_sha256': file_sha256(output / 'split.json'),
                'original_sha256_by_seed': identities, 'selection_code_sha256': file_sha256(__file__)}
    if path.exists():
        plan = json.loads(path.read_text(encoding='utf-8'))
        if any(plan.get(key) != value for key, value in identity.items()):
            raise ValueError('Saved requests differ from random protocol/source models. Use a NEW output directory.')
        for seed in args.seeds:
            groups_for_seed(plan, seed, data.classes)
        return plan
    used_anchors, used_sets, by_seed = set(), {n: set() for n in args.forget_counts}, {}
    for seed in sorted(args.seeds):
        model = load_model(seed)
        similarity, counts = training_similarity(model, data, args.device)
        del model
        anchor, order = choose_anchor(similarity, seed, args.forget_counts, used_anchors, used_sets)
        groups = [{'name': f'forget_{n}', 'classes': order[:n],
                   'class_names': [data.classes[c] for c in order[:n]]} for n in args.forget_counts]
        by_seed[str(seed)] = {'seed': seed, 'anchor_id': anchor, 'anchor_name': data.classes[anchor],
                             'reference_seed': seed, 'groups': groups,
                             'training_samples_per_class': counts, 'cosine': similarity.tolist()}
        print('RANDOM REQUESTS:', seed, groups, flush=True)
    plan = {**identity, 'selection_data': 'train only',
            'criterion': 'cosine of mean same-seed Original avgpool embeddings; ties by class ID',
            'collision_rule': 'seeded permutation; first unused anchor with distinct unordered sets',
            'by_seed': by_seed}
    write_json(path, plan)
    return plan
