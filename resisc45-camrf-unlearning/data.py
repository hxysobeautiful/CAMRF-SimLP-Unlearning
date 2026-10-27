import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_manifest_indices(manifest, protocol=None):
    """Validate either deliberately identical full sets or disjoint partitions."""
    protocol = protocol or manifest.get('data_protocol', 'disjoint')
    if protocol not in ('all_shared', 'disjoint'):
        raise ValueError('Unknown data protocol: '+str(protocol))
    if manifest.get('data_protocol', 'disjoint') != protocol:
        raise ValueError('Data protocol changed; use a NEW output directory.')
    groups = manifest['indices']
    if set(groups) != {'train', 'validation', 'attack_cal', 'test'}:
        raise ValueError('Expected train/validation/attack_cal/test sample groups.')
    expected = list(range(len(manifest['files'])))
    if any(type(i) is not int for values in groups.values() for i in values):
        raise ValueError('Sample indices must be integers.')
    if protocol == 'all_shared':
        if any(values != expected for values in groups.values()):
            raise ValueError('all_shared requires every group to contain ALL images in the same order.')
    elif sorted(i for values in groups.values() for i in values) != expected:
        raise ValueError('Disjoint partitions must cover each image exactly once.')
    return protocol


class Data:
    def __init__(self, args):
        root = Path(args.data_root).expanduser().resolve()
        if (root / 'NWPU-RESISC45').is_dir():
            root = root / 'NWPU-RESISC45'
        size = args.image_size
        eval_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                      transforms.Normalize(MEAN, STD)])
        train_tf = transforms.Compose([transforms.RandomResizedCrop(size, scale=(.7, 1.)),
                                       transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                       transforms.Normalize(MEAN, STD)])
        self.eval = datasets.ImageFolder(root, transform=eval_tf)
        self.train = datasets.ImageFolder(root, transform=train_tf)
        self.classes = self.eval.classes
        if len(self.classes) != 45 and not args.smoke:
            raise ValueError('Expected 45 class subfolders under --data-root; no label remapping is guessed')
        self.labels = np.asarray(self.eval.targets)
        if not args.smoke and any(np.sum(self.labels == c) != 700 for c in range(45)):
            raise ValueError('Expected 700 images/class in complete NWPU-RESISC45; verify download/extraction')
        files = [str(Path(p).relative_to(root)).replace('\\', '/') for p, _ in self.eval.samples]
        # Filename + byte size detects missing/replaced datasets cheaply. Split contains all relative names.
        fingerprint = digest([(p, Path(root / p).stat().st_size) for p in files])
        self.protocol = getattr(args, 'data_protocol', 'all_shared')
        if self.protocol not in ('all_shared', 'disjoint'):
            raise ValueError('Unknown data protocol.')
        self.shared_data = self.protocol == 'all_shared'
        path = Path(args.output) / 'split.json'
        if path.exists():
            manifest = json.loads(path.read_text(encoding='utf-8'))
            if manifest['fingerprint'] != fingerprint or manifest['split_seed'] != args.split_seed:
                raise ValueError('Dataset or split seed changed; use a new output directory')
            validate_manifest_indices(manifest, self.protocol)
            self.split = {k: np.asarray(v, dtype=int) for k, v in manifest['indices'].items()}
        else:
            self.split = {k: [] for k in ('train', 'validation', 'attack_cal', 'test')}
            if self.shared_data:
                self.split = {k: list(range(len(self.labels))) for k in self.split}
            else:
                rng = np.random.default_rng(args.split_seed)
                for c in range(len(self.classes)):
                    idx = rng.permutation(np.flatnonzero(self.labels == c))
                    n = len(idx)
                    a, b, d = int(n*.5), int(n*.1), int(n*.1)
                    if min(a, b, d, n-a-b-d) < 2:
                        raise ValueError('Need at least 20 images/class for disjoint smoke splits')
                    for key, values in zip(self.split, (idx[:a], idx[a:a+b], idx[a+b:a+b+d], idx[a+b+d:])):
                        self.split[key].extend(map(int, values))
            manifest = {'dataset': 'SYNTHETIC_SMOKE' if args.smoke else 'NWPU-RESISC45',
                        'class_to_idx': self.eval.class_to_idx, 'classes': self.classes,
                        'files': files, 'fingerprint': fingerprint, 'split_seed': args.split_seed,
                        'data_protocol': self.protocol, 'indices': self.split,
                        'evaluation_scope': 'all_dataset_resubstitution' if self.shared_data else 'held_out_test',
                        'note': ('Training, validation, attack calibration and test reuse ALL images; not held-out evaluation'
                                 if self.shared_data else 'Custom stratified 50/10/10/30 disjoint protocol')}
            validate_manifest_indices(manifest, self.protocol)
            write_json(path, manifest)
            self.split = {k: np.asarray(v, dtype=int) for k, v in self.split.items()}
        self.fingerprint = fingerprint
        self.args = args

    def indices(self, split, classes=None, limit=0):
        indices = self.split[split]
        if classes is not None:
            indices = indices[np.isin(self.labels[indices], classes)]
        if limit:
            indices = np.concatenate([indices[self.labels[indices] == c][:limit]
                                      for c in sorted(set(self.labels[indices]))])
        if not len(indices):
            raise ValueError('Empty dataset subset')
        return indices

    def loader(self, split='train', classes=None, augment=False, seed=0, limit=0, indices=None):
        if indices is None:
            indices = self.indices(split, classes, limit)
        return DataLoader(Subset(self.train if augment else self.eval, list(map(int, indices))),
                          batch_size=self.args.batch_size, shuffle=augment,
                          num_workers=self.args.workers, pin_memory=self.args.device.startswith('cuda'),
                          generator=torch.Generator().manual_seed(seed))

    def mia_indices(self, forget, seed):
        if self.shared_data:
            raise ValueError('MIA unavailable: all images are training members; there are no held-out nonmembers.')
        rng = np.random.default_rng(seed + 7341)
        groups = {k: [] for k in ('member_cal', 'nonmember_cal', 'member_test', 'nonmember_test')}
        for c in forget:
            members = rng.permutation(self.indices('train', [c]))
            noncal = rng.permutation(self.indices('attack_cal', [c]))
            nontest = rng.permutation(self.indices('test', [c]))
            ncal = min(len(members)//2, len(noncal))
            ntest = min(len(members)-ncal, len(nontest), self.args.mia_samples_per_class)
            chunks = (members[:ncal], noncal[:ncal], members[ncal:ncal+ntest], nontest[:ntest])
            for k, v in zip(groups, chunks):
                groups[k].extend(map(int, v))
        return groups
