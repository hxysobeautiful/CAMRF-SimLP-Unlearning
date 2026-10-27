import copy
import csv
import hashlib
import json
import time
from decimal import Decimal
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import camrf_core as core
from mask_core import apply_similarity_adaptive_pruning, prepare_similarity_policy, build_similarity_masks


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def grid(low, high, step):
    a, b, d = map(lambda x: Decimal(str(x)), (low, high, step))
    if d <= 0 or b < a or a < 0:
        raise ValueError('Invalid grid range')
    if (b-a) % d != 0:
        raise ValueError('Grid maximum must lie on the requested step; no silent linspace resampling')
    return [float(a+i*d) for i in range(int((b-a)/d)+1)]


def importance(model, data, device, args):
    """TFIDF_x_PaperCAMRF, matching the supplied five-variant reference."""
    if args.cam_aggregation != 'peak':
        raise ValueError('The paper requires ReLU-peak CAM-RF; spatial_mean is not the paper method.')
    model.eval()
    core.CIFAR_NUM_CLASSES = len(data.classes)
    core.ALL_CLASSES = tuple(range(len(data.classes)))
    acts = core.compute_class_channel_activations(
        model, data.loader(limit=args.importance_samples_per_class), device, core.ALL_CLASSES)
    result = {}
    for c in core.ALL_CLASSES:
        tfidf = core.compute_fudp_tfidf_importance(acts, core.ALL_CLASSES, c)
        # The same deterministic training subset is used for TF-IDF and CAM-RF.
        result[c] = core.compute_paper_camrf_importance(
            model, data.loader(classes=[c], limit=args.importance_samples_per_class),
            device, c, tfidf, image_size=args.image_size)
        print('Importance:', c, data.classes[c], core.PAPER_IMPORTANCE_METHOD, flush=True)
    return result


def plan(model, imp, forget, p, q, depth_scaling=False):
    """Paper Sim-LP; no depth factor, stem exemption or extra pruning cap."""
    if depth_scaling:
        raise ValueError('Depth scaling is absent from the paper Sim-LP equations.')
    _, report = build_similarity_masks(prepare_similarity_policy(model, imp, forget), p, q)
    return report


@torch.inference_mode()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        z = model(x.to(device))
        if not torch.isfinite(z).all():
            raise RuntimeError('Nonfinite model output; check learning rate')
        correct += int((z.argmax(1).cpu() == y).sum())
        total += len(y)
    return 100 * correct/total


def search(model, imp, forget, data, args, folder):
    device = args.device
    f_loader = data.loader('validation', forget, limit=args.search_samples_per_class)
    retain = sorted(set(range(len(data.classes)))-set(forget))
    r_loader = data.loader('validation', retain, limit=args.search_samples_per_class)
    ps, qs = grid(args.p_min, args.p_max, args.p_step), grid(args.q_min, args.q_max, args.q_step)
    print('Exact grid:', len(ps), 'x', len(qs), 'step=', args.p_step, args.q_step, flush=True)
    original = {k: v.detach().clone() for k,v in model.state_dict().items()}
    candidate = copy.deepcopy(model)
    cache = {}
    prior_seconds = 0.
    cache_path = folder / 'search_cache.jsonl'
    if cache_path.exists():
        for line in cache_path.read_text(encoding='utf-8').splitlines():
            try:
                old = json.loads(line)
            except json.JSONDecodeError:
                continue  # A process interruption can leave only its last line incomplete.
            prior_seconds = max(prior_seconds, old['elapsed_seconds'])
            if 'mask_hash' in old:
                cache[old['mask_hash']] = tuple(old['metrics'])
        print('Resume search:', len(cache), 'cached masks', flush=True)
    policy = prepare_similarity_policy(model, imp, forget)
    if args.depth_scaling:
        raise ValueError('Depth scaling is absent from the paper Sim-LP equations.')
    best = None
    sync(device)
    start = time.perf_counter()
    with open(folder / 'search.csv', 'w', newline='', encoding='utf-8') as file, cache_path.open('a', encoding='utf-8') as cache_file:
        writer = csv.DictWriter(file, fieldnames=['p','q','F_val','R_val','H_val','unique_mask','mask_hash'])
        writer.writeheader()
        for p in ps:
            for q in qs:
                _, mp = build_similarity_masks(policy, p, q)
                key = hashlib.sha256(repr([(n, v['final_pruned_indices']) for n,v in mp.items()]).encode()).hexdigest()
                unique = key not in cache
                if unique:
                    candidate.load_state_dict(original)
                    apply_similarity_adaptive_pruning(candidate, mp)
                    fa, ra = accuracy(candidate, f_loader, device), accuracy(candidate, r_loader, device)
                    h = 2*(100-fa)*ra/max(100-fa+ra, 1e-12)
                    cache[key] = (fa, ra, h)
                    cache_file.write('\n'+json.dumps({'mask_hash':key,'metrics':[fa,ra,h],
                        'elapsed_seconds':prior_seconds+time.perf_counter()-start})+'\n')
                    cache_file.flush()
                fa, ra, h = cache[key]
                row = dict(p=p, q=q, F_val=fa, R_val=ra, H_val=h, unique_mask=unique, mask_hash=key)
                writer.writerow(row)
                rank = (h, ra, -fa, -p, -q)
                if best is None or rank > best[0]:
                    best = (rank, mp, row)
            file.flush()
            cache_file.write('\n'+json.dumps({'elapsed_seconds':prior_seconds+time.perf_counter()-start})+'\n')
            cache_file.flush()
            print('Ours grid P={:.2f}: best H={:.3f}, unique masks={}'.format(p,best[0][0],len(cache)), flush=True)
    sync(device)
    search_seconds = prior_seconds+time.perf_counter()-start
    model.load_state_dict(original)
    sync(device)
    start = time.perf_counter()
    report = apply_similarity_adaptive_pruning(model, best[1])
    sync(device)
    apply_seconds = time.perf_counter()-start
    return model, {'search_seconds': search_seconds, 'mask_apply_seconds': apply_seconds,
                   'unlearning_seconds': search_seconds+apply_seconds,
                   'selected': best[2], 'unique_masks': len(cache), 'plan': report}


def baseline(model, method, forget, data, args, seed):
    """Explicit local baselines; SalUn is a retained-label, two-phase adaptation.

    SalUn uses zero-momentum/zero-weight-decay SGD so non-salient parameters
    cannot drift through momentum/decay. BN statistics frozen during unlearning.
    """
    device = args.device
    retained = sorted(set(range(len(data.classes)))-set(forget))
    floader = data.loader(classes=forget, augment=True, seed=seed)
    rloader = data.loader(classes=retained, augment=True, seed=seed)
    mask = None
    sync(device)
    start = time.perf_counter()
    if method == 'SalUn':
        model.eval()
        saliency = {n: torch.zeros_like(p) for n,p in model.named_parameters()}
        for x, y in data.loader(classes=forget):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x.to(device)), y.to(device)).backward()
            for n,p in model.named_parameters():
                if p.grad is not None:
                    saliency[n] += p.grad.abs()*len(y)
        flat = torch.cat([v.flatten() for v in saliency.values()])
        threshold = flat.kthvalue(max(1, int(flat.numel()*(1-args.salun_fraction)))).values
        mask = {n: v >= threshold for n,v in saliency.items()}
        del saliency, flat
    lr = args.ng_lr if method == 'NegativeGradient' else args.baseline_lr
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)
    choices = torch.tensor(retained, device=device)
    for epoch in range(args.unlearn_epochs):
        # Freeze BN statistics; this policy is logged and identical for these baselines.
        model.eval()
        passes = [(floader, True)] if method == 'NegativeGradient' else [(rloader, False)]
        if method == 'SalUn':
            passes = [(floader, True), (rloader, False)]
        for loader, forgetting in passes:
            for x, y in loader:
                x,y = x.to(device),y.to(device)
                if method == 'SalUn' and forgetting:
                    y = choices[torch.randint(len(choices), y.shape, device=device)]
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                if method == 'NegativeGradient':
                    loss = -loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite baseline loss')
                loss.backward()
                if mask is not None:
                    for n,p in model.named_parameters():
                        if p.grad is not None:
                            p.grad.mul_(mask[n])
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                opt.step()
        print(method, 'epoch', epoch+1, '/', args.unlearn_epochs, flush=True)
    sync(device)
    return model, {'unlearning_seconds': time.perf_counter()-start}
