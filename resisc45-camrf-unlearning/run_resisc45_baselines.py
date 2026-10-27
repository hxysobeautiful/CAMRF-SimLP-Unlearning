"""RESISC45 baselines from saved Original models, seeds 42/43/44.

Place beside data.py and privacy.py in the existing project.
Never trains Original/Retrain or computes/reads Ours channel importance.
Each method/request/seed starts from its corresponding saved Original.
DELETE: mask each sample's true-class teacher logit; KL distillation, T=1.
Fisher: retained-data diagonal expected-Fisher noise (disclosed batch estimator).
SalUn: signed gradient accumulation, global saliency mask, random labels,
optional retained-data repair, and restoration of masked-out parameters.
This is a disclosed RESISC45 adaptation, not a claim of paper table reproduction.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models

from data import Data, validate_manifest_indices, write_json
from random_requests import groups_for_seed
from privacy import unavailable_mia, log_true_probability, mia_metrics

PROJECT_DIR = Path(__file__).resolve().parent
METHODS = ('NegativeGradient', 'Fisher', 'DELETE', 'SalUn')
SOURCES = {
    'DELETE': 'https://github.com/shaaaaron/DELETE/blob/main/method/delete.py',
    'NegativeGradient': 'https://github.com/shaaaaron/DELETE/blob/main/method/gradient_ascent.py',
    'Fisher': 'https://github.com/AdityaGolatkar/SelectiveForgetting/blob/master/Forgetting.ipynb',
    'Fisher_batch_estimator': 'https://github.com/OPTML-Group/Unlearn-Saliency/blob/master/Classification/unlearn/fisher.py',
    'SalUn_mask': 'https://github.com/OPTML-Group/Unlearn-Saliency/blob/master/Classification/generate_mask.py',
    'SalUn_update': 'https://github.com/OPTML-Group/Unlearn-Saliency/blob/master/Classification/unlearn/RL.py',
}

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def load_original(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    required = {'architecture', 'classes', 'fingerprint', 'seed', 'state_dict', 'metadata'}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError('Expected this RESISC45 project Original checkpoint dictionary.')
    if checkpoint['architecture'] != 'torchvision_resnet18_standard_stem':
        raise ValueError('Expected standard-stem torchvision ResNet-18.')
    classes = checkpoint['classes']
    if not isinstance(classes, list) or len(classes) != 45 or len(set(classes)) != 45:
        raise ValueError('Expected 45 unique checkpoint class names.')
    if checkpoint['metadata'].get('forget_classes'):
        raise ValueError('Expected an Original checkpoint, not a previously unlearned model.')
    model = models.resnet18(weights=None, num_classes=45)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval(), checkpoint


def resolve_requests(args, checkpoint):
    names = checkpoint['classes']
    if args.forget_classes:
        unknown = [name for name in args.forget_classes if name not in names]
        if unknown:
            raise ValueError(f'Unknown class names: {unknown}. Valid names: {names}')
        classes = [names.index(name) for name in args.forget_classes]
        groups = [{'name': f'forget_{len(classes)}', 'classes': classes,
                   'class_names': list(args.forget_classes)}]
        selection = {'type': 'explicit_class_names'}
    else:
        path = args.request_plan or args.results_root / 'request_plan.json'
        if not path.is_file():
            raise FileNotFoundError(f'No request plan at {path}; pass --forget-classes airport (or your target classes).')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        groups = groups_for_seed(manifest, checkpoint['seed'], names)
        if manifest['fingerprint'] != checkpoint['fingerprint'] or manifest['classes'] != names:
            raise ValueError('Request plan dataset differs from Original.')
        if manifest['split_sha256'] != sha256_file(args.results_root / 'split.json'):
            raise ValueError('Request plan and experiment use different data splits.')
        source_path = getattr(args, 'checkpoint', None)
        if source_path is None:
            template = getattr(args, 'checkpoint_template', None)
            source_path = Path(template.format(seed=checkpoint['seed'])) if template else args.results_root / f"seed_{checkpoint['seed']}/Original/best.pth"
        if manifest['original_sha256_by_seed'][str(checkpoint['seed'])] != sha256_file(source_path):
            raise ValueError('Request plan was computed from a different Original.')
        selection = {'type': 'saved_request_plan', 'path': str(path.resolve()),
                     'sha256': sha256_file(path), 'reference_seed': manifest.get('reference_seed')}
    if not groups:
        raise ValueError('No forgetting requests.')
    seen = set()
    for group in groups:
        label = group['name']
        if not isinstance(label, str) or not label or not all(c.isalnum() or c in '_-' for c in label):
            raise ValueError('Request names must be simple directory names.')
        if label in seen:
            raise ValueError('Duplicate request names.')
        seen.add(label)
        targets = group['classes']
        if (not isinstance(targets, list) or not 1 <= len(targets) <= 43
                or any(type(c) is not int or not 0 <= c < 45 for c in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError('Each request must have distinct valid class IDs and leave at least two retained classes.')
        expected = [names[c] for c in targets]
        if group.get('class_names') != expected:
            raise ValueError('Request plan class names do not match the Original class mapping.')
    return groups, selection


def load_data(args, config, checkpoint):
    split_path = args.results_root / 'split.json'
    if not split_path.is_file():
        raise FileNotFoundError(f'Missing saved Original split: {split_path}')
    manifest = json.loads(split_path.read_text(encoding='utf-8'))
    if manifest.get('classes') != checkpoint['classes'] or manifest.get('fingerprint') != checkpoint['fingerprint']:
        raise ValueError('Original and split.json have different class mapping or dataset.')
    validate_manifest_indices(manifest, config.get('data_protocol', 'disjoint'))
    if config.get('smoke', False):
        raise ValueError('Expected a real RESISC45 experiment, not smoke data.')
    data_args = SimpleNamespace(
        data_root=str(args.data_root), output=str(args.results_root),
        image_size=config['image_size'], split_seed=config['split_seed'],
        data_protocol=config.get('data_protocol', 'disjoint'),
        batch_size=args.batch_size, workers=args.workers, device=args.device, smoke=False,
    )
    data = Data(data_args)  # Existing split must exist: Data only reads it.
    if checkpoint['metadata'].get('data_protocol', 'disjoint') != data.protocol:
        raise ValueError('Original checkpoint training protocol differs from saved data protocol.')
    if data.classes != checkpoint['classes'] or data.fingerprint != checkpoint['fingerprint']:
        raise ValueError('Dataset differs from the Original checkpoint.')
    return data


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def training_mode(model, bn_policy):
    model.train()
    if bn_policy == 'frozen':
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def delete_targets(teacher_logits, labels):
    """Paper Algorithm 1: mask sample label u only, including multi-class requests."""
    masked = teacher_logits.detach().clone()
    masked.scatter_(1, labels[:, None], -torch.inf)
    return masked.softmax(dim=1)


def saliency_mask(model, loader, fraction, device):
    """abs(sum_batch gradient of mean CE), not sum_batch(abs(gradient))."""
    model.eval()
    params = dict(model.named_parameters())
    accumulated = {name: torch.zeros_like(parameter) for name, parameter in params.items()}
    for images, labels in loader:
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images.to(device)), labels.to(device, dtype=torch.long))
        loss.backward()
        with torch.no_grad():
            for name, parameter in params.items():
                if parameter.grad is not None:
                    accumulated[name].add_(parameter.grad)
    values = torch.cat([value.abs().reshape(-1) for value in accumulated.values()])
    count = int(values.numel() * fraction)
    if count < 1:
        raise ValueError('SalUn fraction selects no parameters.')
    selected = torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
    selected[torch.argsort(values, descending=True, stable=True)[:count]] = True
    result, offset = {}, 0
    for name, parameter in params.items():
        size = parameter.numel()
        result[name] = selected[offset:offset + size].reshape_as(parameter).clone()
        offset += size
    model.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def restore_non_salient(model, mask, original, optimizer):
    for name, parameter in model.named_parameters():
        active = mask[name]
        parameter[~active] = original[name][~active]
        momentum = optimizer.state.get(parameter, {}).get('momentum_buffer')
        if momentum is not None:
            momentum.mul_(active)


def run_gradient_method(model, method, targets, data, args, seed, folder):
    num_classes = len(data.classes)
    retained = sorted(set(range(num_classes)) - set(targets))
    teacher, mask, original = None, None, None
    floader = data.loader('train', targets, augment=True, seed=seed)
    # Avoid creating/accessing retained training data for DELETE/NegativeGradient.
    rloader = data.loader('train', retained, augment=True, seed=seed) if method == 'SalUn' and args.salun_retain else None
    sync(args.device)
    start = time.perf_counter()
    if method == 'DELETE':
        teacher = copy.deepcopy(model).eval()
        teacher.requires_grad_(False)
    if method == 'SalUn':
        mask = saliency_mask(model, data.loader('train', targets), args.salun_fraction, args.device)
        original = {name: value.detach().clone() for name, value in model.named_parameters()}
        atomic_save(folder / 'saliency_mask.pth', {name: value.cpu() for name, value in mask.items()})
    lr = {'NegativeGradient': args.ng_lr, 'DELETE': args.delete_lr, 'SalUn': args.salun_lr}[method]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=0)
    history = []
    for epoch in range(args.epochs):
        passes = [('forget', floader)] + ([('retain', rloader)] if rloader is not None else [])
        losses = {}
        for phase, loader in passes:
            training_mode(model, args.bn_policy)
            total_loss, total = 0., 0
            for images, labels in loader:
                images, labels = images.to(args.device), labels.to(args.device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                if method == 'DELETE':
                    with torch.no_grad():
                        soft_targets = delete_targets(teacher(images), labels)
                    loss = F.kl_div(logits.log_softmax(dim=1), soft_targets, reduction='batchmean')
                elif method == 'NegativeGradient':
                    loss = -F.cross_entropy(logits, labels)
                else:
                    if phase == 'forget':
                        # Guaranteed different from the sample label; can be another forgotten class.
                        labels = (labels + torch.randint(1, num_classes, labels.shape, device=labels.device)) % num_classes
                    loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite {method} loss; adjust its learning rate.')
                loss.backward()
                if mask is not None:
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(mask[name])
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if mask is not None:
                    restore_non_salient(model, mask, original, optimizer)
                total_loss += float(loss.detach()) * len(labels)
                total += len(labels)
            losses[phase] = total_loss / total
        history.append({'epoch': epoch + 1, 'losses': losses})
        write_json(folder / 'training_history.json', history)
        print(f'{method} seed={seed} epoch={epoch + 1}/{args.epochs} losses={losses}', flush=True)
    sync(args.device)
    elapsed = time.perf_counter() - start
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f'Nonfinite {method} parameters.')
    return model.eval(), {
        'unlearning_seconds': elapsed, 'epochs': args.epochs, 'learning_rate': lr,
        'momentum': args.momentum, 'weight_decay': 0., 'bn_policy': args.bn_policy,
        'grad_clip': args.grad_clip, 'uses_retained_training_data': rloader is not None,
        'salun_fraction': args.salun_fraction if method == 'SalUn' else None,
        'salun_active_parameters': sum(int(v.sum()) for v in mask.values()) if mask is not None else None,
        'delete_mask': 'per_sample_true_label' if method == 'DELETE' else None,
        'implementation_variant': {
            'NegativeGradient': 'negative_CE_SGD',
            'DELETE': 'paper_Algorithm1_T1_KL_masked_frozen_original',
            'SalUn': 'signed_gradient_saliency_random_label_two_pass' if args.salun_retain else 'signed_gradient_saliency_random_label_forget_only',
        }[method],
    }


def diagonal_expected_fisher(model, loader, device):
    """Class-expectation of squared batch-mean CE gradients, as fisher_new.

    With batch_size=1 this is the exact per-example expected diagonal Fisher.
    For batch_size>1 it is a batch approximation, not empirical label Fisher.
    """
    model.eval()
    parameters = tuple(model.parameters())
    result = [torch.zeros_like(parameter) for parameter in parameters]
    batches = 0
    last_message = time.monotonic()
    for images, _ in loader:
        logits = model(images.to(device))
        probabilities = logits.detach().softmax(dim=1).mean(dim=0)
        classes = logits.shape[1]
        for cls in range(classes):
            labels = torch.full((len(images),), cls, dtype=torch.long, device=logits.device)
            gradient = torch.autograd.grad(F.cross_entropy(logits, labels), parameters,
                                           retain_graph=cls < classes - 1, allow_unused=True)
            with torch.no_grad():
                for accumulator, derivative in zip(result, gradient):
                    if derivative is not None:
                        accumulator.add_(probabilities[cls] * derivative.square())
            if time.monotonic() - last_message > 20:
                print(f'Fisher batch={batches + 1}/{len(loader)} class={cls + 1}/{classes}', flush=True)
                last_message = time.monotonic()
        batches += 1
    if not batches:
        raise ValueError('Empty Fisher retained loader.')
    for value in result:
        value.div_(batches)
        if not torch.isfinite(value).all():
            raise RuntimeError('Nonfinite Fisher estimate.')
    return result


def fisher_mean_variance(parameter, diagonal, name, targets, alpha):
    classifier = name in ('fc.weight', 'fc.bias')
    variance = (diagonal + 1e-8).reciprocal().clamp(max=100. if classifier else 1000.) * alpha
    if parameter.ndim > 1:
        variance = variance.mean(dim=1, keepdim=True).expand_as(parameter).clone()
    mean = parameter.detach().clone()
    if classifier:
        mean[targets] = 0.
        variance[targets] = 1e-4
    if classifier or parameter.ndim == 1:
        variance *= 10.
    return mean, variance


def run_fisher(model, targets, data, args, seed, folder):
    retained = sorted(set(range(len(data.classes))) - set(targets))
    # Saved training split, deterministic preprocessing, no augmentation.
    indices = data.indices('train', retained, limit=args.fisher_samples_per_class)
    loader = DataLoader(Subset(data.eval, list(map(int, indices))),
                        batch_size=args.fisher_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=torch.device(args.device).type == 'cuda')
    sync(args.device)
    start = time.perf_counter()
    diagonal = diagonal_expected_fisher(model, loader, args.device)
    with torch.no_grad():
        for (name, parameter), estimate in zip(model.named_parameters(), diagonal):
            mean, variance = fisher_mean_variance(parameter, estimate, name, targets, args.fisher_alpha)
            parameter.copy_(mean + variance.sqrt() * torch.randn_like(parameter))
    sync(args.device)
    elapsed = time.perf_counter() - start
    write_json(folder / 'fisher_metadata.json', {
        'samples': len(indices), 'batches': len(loader), 'batch_size': args.fisher_batch_size,
        'estimator': 'exact_per_example_expected_diagonal' if args.fisher_batch_size == 1 else 'batch_mean_gradient_expected_diagonal_approximation',
        'formula': 'mean_b sum_c mean_batch(p(c|x)) * (gradient mean_batch CE(c))^2',
        'samples_per_class_limit': args.fisher_samples_per_class,
    })
    return model.eval(), {
        'unlearning_seconds': elapsed, 'epochs': None, 'learning_rate': None,
        'fisher_alpha': args.fisher_alpha, 'fisher_batch_size': args.fisher_batch_size,
        'fisher_samples': len(indices), 'fisher_samples_per_class': args.fisher_samples_per_class,
        'uses_retained_training_data': True, 'bn_policy': 'frozen',
        'implementation_variant': 'Golatkar_Fisher_noise_with_class_head_reset_and_disclosed_batch_estimator',
    }


@torch.inference_mode()
def collect_logits(model, data, device, folder):
    path = folder / 'evaluation_logits.npz'
    if path.exists():
        with np.load(path) as bundle:
            return bundle['logits']
    model.eval()
    logits = np.full((len(data.labels), len(data.classes)), np.nan, dtype=np.float32)
    for split in (('test',) if data.shared_data else ('train', 'attack_cal', 'test')):
        indices = data.indices(split)
        offset = 0
        for images, _ in data.loader(indices=indices):
            values = model(images.to(device)).cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError('Nonfinite evaluation logits.')
            logits[indices[offset:offset + len(values)]] = values
            offset += len(values)
        if offset != len(indices):
            raise RuntimeError('Evaluation sample count differs from saved split.')
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, logits=logits)
    temporary.replace(path)
    return logits


def evaluate_logits(logits, data, targets, seed, original_f):
    retained = sorted(set(range(len(data.classes))) - set(targets))
    def accuracy(split, classes):
        indices = data.indices(split, classes)
        return float(np.mean(logits[indices].argmax(1) == data.labels[indices]) * 100)
    f, r = accuracy('test', targets), accuracy('test', retained)
    per_class = {data.classes[c]: accuracy('test', [c]) for c in targets}
    if data.shared_data:
        mia = unavailable_mia()
    else:
        mia_indices = data.mia_indices(targets, seed)
        scores = {key: log_true_probability(logits[indices], data.labels[indices])
                  for key, indices in mia_indices.items()}
        mia = mia_metrics(**scores)
    drop = original_f - f
    return {
        'F_test': f, 'R_test': r,
        'F_train': accuracy('train', targets), 'R_train': accuracy('train', retained),
        'forget_test_drop_pp': drop,
        'worst_F_test': max(per_class.values()),
        'forgotten_class_test_accuracy': per_class,
        **mia, 'data_protocol':data.protocol,
        'evaluation_scope':'all_dataset_resubstitution' if data.shared_data else 'held_out_test',
    }


def save_tables(output, rows, seeds):
    fields = ['forget_classes', 'forget_names', 'request', 'method', 'seed', 'status', 'F_test', 'R_test', 'F_train', 'R_train',
              'forget_test_drop_pp', 'worst_F_test', 'data_protocol', 'evaluation_scope', 'mia_status',
              'mia_auc', 'mia_auc_raw', 'mia_tpr_at_calibrated_1pct_fpr', 'mia_actual_test_fpr',
              'unlearning_seconds', 'evaluation_seconds', 'uses_retained_training_data',
              'checkpoint_bytes', 'implementation_variant', 'error']
    with (output / 'results.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    metrics = ['F_test', 'R_test', 'F_train', 'R_train', 'worst_F_test', 'mia_auc', 'unlearning_seconds']
    summary = []
    for request, method in sorted({(row['request'], row['method']) for row in rows}):
        selected = [row for row in rows if row['request'] == request and row['method'] == method and row['status'] == 'OK']
        item = {'request': request, 'method': method, 'n': len(selected),
                'expected_n': len(seeds), 'complete': sorted(row['seed'] for row in selected) == sorted(seeds)}
        for metric in metrics:
            values = [row[metric] for row in selected if row[metric] is not None]
            item[metric + '_mean'] = float(np.mean(values)) if values else None
            item[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(item)
    write_json(output / 'summary.json', summary)
    with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f'Updated: {output / "results.csv"} and {output / "summary.csv"}', flush=True)



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=PROJECT_DIR / 'results_resisc45')
    parser.add_argument('--checkpoint-template', help='Optional path template containing {seed}.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--data-root', type=Path, default=PROJECT_DIR / 'data/NWPU-RESISC45')
    parser.add_argument('--output', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--forget-classes', nargs='+')
    group.add_argument('--request-plan', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--epochs', type=int, help='Default: saved config unlearn_epochs, otherwise 5.')
    parser.add_argument('--ng-lr', type=float, default=1e-5)
    parser.add_argument('--delete-lr', type=float, default=1e-3)
    parser.add_argument('--salun-lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--bn-policy', choices=('frozen', 'update'), default='frozen')
    parser.add_argument('--grad-clip', type=float, default=0.)
    parser.add_argument('--salun-fraction', type=float, default=.5)
    parser.add_argument('--salun-retain', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fisher-alpha', type=float, default=1e-7)
    parser.add_argument('--fisher-batch-size', type=int, default=32,
                        help='1: exact per-example expected diagonal; >1: batch approximation.')
    parser.add_argument('--fisher-samples-per-class', type=int, default=0, help='0: all retained training images.')
    parser.add_argument('--mia-samples-per-class', type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.results_root = args.results_root.resolve()
    args.data_root = args.data_root.resolve()
    args.output = (args.output or args.results_root / 'baseline_results').resolve()
    config_path, split_path = args.results_root / 'config.json', args.results_root / 'split.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if args.mia_samples_per_class is None:
        args.mia_samples_per_class = config['mia_samples_per_class']
    elif args.mia_samples_per_class != config['mia_samples_per_class']:
        raise ValueError('MIA sample limit must match the shared reference config.')
    args.epochs = args.epochs if args.epochs is not None else int(config.get('unlearn_epochs', 5))
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        raise ValueError('Seeds must be unique integers in [0, 2**32).')
    if len(set(args.methods)) != len(args.methods):
        raise ValueError('Duplicate methods.')
    if min(args.batch_size, args.threads, args.epochs, args.fisher_batch_size, args.mia_samples_per_class) <= 0:
        raise ValueError('Batch sizes, epochs, threads and MIA sample limit must be positive.')
    if min(args.workers, args.fisher_samples_per_class, args.grad_clip) < 0:
        raise ValueError('Workers, sample limit and clipping cannot be negative.')
    if not 0 < args.salun_fraction <= 1 or not 0 <= args.momentum < 1:
        raise ValueError('Invalid saliency fraction or momentum.')
    for value in (args.ng_lr, args.delete_lr, args.salun_lr, args.fisher_alpha):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Learning rates and Fisher alpha must be finite and positive.')
    if not math.isfinite(args.grad_clip):
        raise ValueError('Clipping must be finite.')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; launch the GPU container with --nv or use --device cpu.')
    torch.set_num_threads(args.threads)

    # Preflight every seed before any unlearning; missing 43/44 cannot silently become a single-seed run.
    originals, checkpoints, identities = {}, {}, {}
    for seed in args.seeds:
        path = (Path(args.checkpoint_template.format(seed=seed)) if args.checkpoint_template
                else args.results_root / f'seed_{seed}' / 'Original/best.pth').resolve()
        if args.checkpoint_template and '{seed}' not in args.checkpoint_template:
            raise ValueError('--checkpoint-template must contain {seed}.')
        if args.output == path.parent or args.output in path.parents:
            raise ValueError('Output directory must be separate from saved Original checkpoints.')
        if not path.is_file():
            raise FileNotFoundError(f'Missing Original for seed={seed}: {path}')
        model, checkpoint = load_original(path, 'cpu')
        if checkpoint['seed'] != seed:
            raise ValueError(f'Checkpoint seed does not match requested seed {seed}.')
        if checkpoints:
            first = next(iter(checkpoints.values()))
            if any(checkpoint[key] != first[key] for key in ('classes', 'fingerprint', 'architecture')):
                raise ValueError('All seeds must use the same architecture, dataset and class mapping.')
        originals[seed], checkpoints[seed] = model, checkpoint
        identities[str(seed)] = {'path': str(path), 'sha256': sha256_file(path), 'bytes': path.stat().st_size}
        print(f'Validated Original seed={seed}: {path}', flush=True)
    first = checkpoints[args.seeds[0]]
    groups_by_seed, selection = {}, {}
    for seed in args.seeds:
        groups_by_seed[seed], selection[str(seed)] = resolve_requests(args, checkpoints[seed])
    data = load_data(args, config, first)
    data.args.mia_samples_per_class = args.mia_samples_per_class
    for seed, groups in groups_by_seed.items():
        for group in groups_by_seed[seed]:
            print(f'Request seed={seed} {group["name"]}: {group["class_names"]}', flush=True)

    settings = {key: value for key, value in vars(args).items() if key not in ('output', 'checkpoint_template', 'request_plan')}
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}
    manifest = {'schema': 1, 'originals': identities, 'groups_by_seed': {str(k): v for k, v in groups_by_seed.items()}, 'selection': selection,
                'settings': settings, 'config_sha256': sha256_file(config_path),
                'split_sha256': sha256_file(split_path), 'runner_sha256': sha256_file(Path(__file__)),
                'project_source_sha256': {name: sha256_file(PROJECT_DIR / name) for name in ('data.py', 'privacy.py')},
                'sources': SOURCES}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest['signature'] = signature
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'run_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Output belongs to different inputs/settings/code. Choose a new --output directory.')
    elif any(args.output.iterdir()):
        raise ValueError('Output is nonempty without a matching manifest. Choose a new --output directory.')
    else:
        write_json(manifest_path, manifest)

    rows, failures = [], []
    for seed in args.seeds:
        seed_all(seed)
        original_folder = args.output / f'seed_{seed}' / 'Original'
        original_folder.mkdir(parents=True, exist_ok=True)
        original = originals[seed]
        start = time.perf_counter()
        original_logits = collect_logits(original.to(args.device), data, args.device, original_folder)
        sync(args.device)
        original_evaluation_seconds = time.perf_counter() - start
        original.cpu()
        for group in groups_by_seed[seed]:
            targets = group['classes']
            test_indices = data.indices('test', targets)
            original_f = float(np.mean(original_logits[test_indices].argmax(1) == data.labels[test_indices]) * 100)
            original_row = {'forget_classes': targets, 'forget_names': group['class_names'], 'request': group['name'], 'method': 'Original', 'seed': seed, 'status': 'OK',
                            'unlearning_seconds': 0., 'evaluation_seconds': original_evaluation_seconds,
                            'uses_retained_training_data': False, 'checkpoint_bytes': identities[str(seed)]['bytes'],
                            'implementation_variant': 'saved_Original',
                            **evaluate_logits(original_logits, data, targets, seed, original_f)}
            rows.append(original_row)
            for method in args.methods:
                folder = args.output / f'seed_{seed}' / group['name'] / method
                folder.mkdir(parents=True, exist_ok=True)
                path, result_path = folder / 'best.pth', folder / 'result.json'
                print(f'\nRunning {method} seed={seed} request={group["name"]}', flush=True)
                model = None
                try:
                    if result_path.exists():
                        result = json.loads(result_path.read_text(encoding='utf-8'))
                        if result['signature'] != signature or result['checkpoint_sha256'] != sha256_file(path):
                            raise ValueError('Completed result/checkpoint identity mismatch.')
                        row = result['row']
                        print('Reusing completed result.', flush=True)
                    else:
                        model = copy.deepcopy(original).to(args.device)
                        seed_all(seed)  # Independent methods start from identical Original, never sequentially unlearned.
                        if path.exists():
                            saved = torch.load(path, map_location='cpu', weights_only=True)
                            if saved['metadata']['run_signature'] != signature:
                                raise ValueError('Partial checkpoint signature mismatch.')
                            model.load_state_dict(saved['state_dict'], strict=True)
                            info = saved['metadata']['unlearning']
                        else:
                            function = run_fisher if method == 'Fisher' else run_gradient_method
                            if method == 'Fisher':
                                model, info = function(model, targets, data, args, seed, folder)
                            else:
                                model, info = function(model, method, targets, data, args, seed, folder)
                            atomic_save(path, {
                                'architecture': first['architecture'], 'classes': first['classes'],
                                'fingerprint': first['fingerprint'], 'seed': seed,
                                'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                                'metadata': {'method': method, 'forget_classes': targets, 'run_signature': signature,
                                             'original': identities[str(seed)], 'unlearning': info},
                            })
                        start = time.perf_counter()
                        logits = collect_logits(model, data, args.device, folder)
                        metrics = evaluate_logits(logits, data, targets, seed, original_f)
                        sync(args.device)
                        row = {'forget_classes': targets, 'forget_names': group['class_names'], 'request': group['name'], 'method': method, 'seed': seed, 'status': 'OK',
                               'evaluation_seconds': time.perf_counter() - start,
                               'checkpoint_bytes': path.stat().st_size, **info, **metrics}
                        write_json(result_path, {'signature': signature, 'checkpoint_sha256': sha256_file(path), 'row': row})
                    rows.append(row)
                    (folder / 'failure.json').unlink(missing_ok=True)
                    print(f'F_test={row["F_test"]:.3f}% R_test={row["R_test"]:.3f}% MIA_AUC={row["mia_auc"]}', flush=True)
                except Exception as error:
                    traceback.print_exc()
                    row = {'forget_classes': targets, 'forget_names': group['class_names'], 'request': group['name'], 'method': method, 'seed': seed, 'status': 'FAILED',
                           'error': f'{type(error).__name__}: {error}'}
                    rows.append(row)
                    failures.append(row)
                    write_json(folder / 'failure.json', row)
                finally:
                    del model
                    if torch.device(args.device).type == 'cuda':
                        torch.cuda.empty_cache()
                save_tables(args.output, rows, args.seeds)
    # Verify source files again, without writing to Original checkpoints.
    for identity in identities.values():
        if sha256_file(identity['path']) != identity['sha256']:
            raise RuntimeError('Original checkpoint changed during execution.')
    save_tables(args.output, rows, args.seeds)
    for item in json.loads((args.output / 'summary.json').read_text(encoding='utf-8')):
        if item['complete']:
            suffix = f' +/- {item["R_test_std"]:.3f}' if item['R_test_std'] is not None else ''
            print(f'{item["request"]} {item["method"]}: n={item["n"]}, F={item["F_test_mean"]:.3f}%, R={item["R_test_mean"]:.3f}{suffix}%', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} failed runs; inspect results.csv and method failure.json.')
    return rows


if __name__ == '__main__':
    main()
