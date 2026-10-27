"""Reproducible centralized NWPU-RESISC45 / ResNet18 unlearning benchmark."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import models
from data import Data, digest, write_json
from random_requests import build_plan, groups_for_seed, reference_checkpoint, file_sha256
from methods import accuracy, baseline, grid, importance, search, sync
from camrf_core import PAPER_IMPORTANCE_METHOD, PAPER_IMPORTANCE_VERSION
from privacy import unavailable_mia, cmia_evaluate, log_true_probability, mia_metrics


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def model_for(data, device, seed):
    seed_all(seed)
    return models.resnet18(weights=None, num_classes=len(data.classes)).to(device)


def from_checkpoint(path, data, args):
    ck = load(path)
    if ck['classes'] != data.classes or ck['fingerprint'] != data.fingerprint:
        raise ValueError('Checkpoint class mapping or dataset differs')
    if ck['metadata'].get('data_protocol', 'disjoint') != data.protocol:
        raise ValueError('Checkpoint training protocol differs; retrain in a NEW output directory.')
    model = model_for(data, args.device, ck['seed'])
    model.load_state_dict(ck['state_dict'], strict=True)
    return model, ck


def checkpoint(model, data, seed, metadata):
    metadata = {**metadata, 'data_protocol':data.protocol,
                'evaluation_scope':'all_dataset_resubstitution' if data.shared_data else 'held_out_test'}
    return {'architecture': 'torchvision_resnet18_standard_stem', 'classes': data.classes,
            'fingerprint': data.fingerprint, 'seed': seed,
            'state_dict': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'metadata': metadata}


def train_model(data, args, seed, folder, forget=()):
    path = folder / 'best.pth'
    if path.exists():
        saved = load(path)
        if (saved['seed'] != seed or saved['classes'] != data.classes
                or saved['fingerprint'] != data.fingerprint
                or saved['metadata'].get('data_protocol', 'disjoint') != data.protocol
                or sorted(saved['metadata'].get('forget_classes', [])) != sorted(forget)):
            raise ValueError('Cached training checkpoint has a different request: ' + str(path))
        print('Reuse completed:', path, flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    model = model_for(data, args.device, seed)
    retained = sorted(set(range(len(data.classes))) - set(forget))
    batches = data.loader(classes=retained, augment=True, seed=seed)
    val = data.loader('validation', retained)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_acc, best_state, best_epoch = -1., None, -1
    history = []
    sync(args.device)
    start = time.perf_counter()
    # Original and Retrain both start from random initialization; no ImageNet weights.
    for epoch in range(args.epochs):
        model.train()
        total_loss = total = 0
        for x,y in batches:
            x,y = x.to(args.device),y.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())*len(y)
            total += len(y)
        va = accuracy(model, val, args.device)
        history.append({'epoch': epoch+1, 'loss':total_loss/total, 'validation_accuracy':va})
        if va > best_acc:
            best_acc, best_epoch = va, epoch+1
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        scheduler.step()
        print(folder.name, 'seed', seed, 'epoch', epoch+1, '/', args.epochs,
              'val={:.2f}'.format(va), flush=True)
        write_json(folder / 'training_history.json', history)
    sync(args.device)
    elapsed = time.perf_counter()-start
    model.load_state_dict(best_state)
    save(path, checkpoint(model, data, seed, {'training_seconds':elapsed,
        'unlearning_seconds':elapsed if forget else 0., 'forget_classes':list(forget),
        'best_epoch':best_epoch, 'best_validation_accuracy':best_acc,
        'initialization':'random', 'classifier_outputs':len(data.classes)}))


def make_plan(data, args, output):
    def load_original(seed):
        model, _ = from_checkpoint(output / f'seed_{seed}/Original/best.pth', data, args)
        return model
    return build_plan(data, args, output, load_original)


def compute_importance(data, args, output, seed):
    path = output / f'seed_{seed}/importance.pth'
    if path.exists():
        cached = load(path)
        if cached.get('method') != PAPER_IMPORTANCE_METHOD or cached.get('definition_version') != PAPER_IMPORTANCE_VERSION:
            raise ValueError('Importance cache predates paper alignment. Use a NEW --output directory.')
        return cached
    model,_ = from_checkpoint(output / f'seed_{seed}/Original/best.pth', data, args)
    model.eval()
    sync(args.device)
    start = time.perf_counter()
    values = importance(model, data, args.device, args)
    sync(args.device)
    record = {'importance':values, 'importance_seconds':time.perf_counter()-start,
              'aggregation':args.cam_aggregation, 'source':'all dataset images' if data.shared_data else 'training samples only',
              'method':PAPER_IMPORTANCE_METHOD, 'definition_version':PAPER_IMPORTANCE_VERSION,
              'class_names':data.classes, 'fingerprint':data.fingerprint, 'seed':seed}
    save(path, record)
    return record


def unlearn(data, args, output, plan_data):
    for request_seed in args.seeds:
        for group in groups_for_seed(plan_data, request_seed, data.classes):
            # CMIA references must erase THIS request, even when initialized with other seeds.
            for model_seed in args.seeds:
                path = reference_checkpoint(output, request_seed, group, model_seed)
                train_model(data, args, model_seed, path.parent, group['classes'])
    for seed in args.seeds:
        imp = compute_importance(data, args, output, seed) if 'Ours' in args.methods else None
        for group in groups_for_seed(plan_data, seed, data.classes):
            for method in args.methods:
                if method in ('Retrain','Original'):
                    continue
                folder = output / f"seed_{seed}/{group['name']}/{method}"
                path = folder / 'best.pth'
                if path.exists():
                    saved = load(path)
                    if sorted(saved['metadata'].get('forget_classes', [])) != sorted(group['classes']):
                        raise ValueError('Cached unlearning model has a different request: ' + str(path))
                    print('Reuse completed:', path, flush=True)
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                model,_ = from_checkpoint(output / f'seed_{seed}/Original/best.pth', data, args)
                seed_all(seed)
                if method == 'Ours':
                    model, meta = search(model, imp['importance'], group['classes'], data, args, folder)
                    meta['search_and_apply_seconds'] = meta['unlearning_seconds']
                    meta['unlearning_seconds'] = meta['mask_apply_seconds']
                    meta['timing_scope'] = 'final_mask_application_only'
                    meta['importance_seconds'] = imp['importance_seconds']
                    meta['total_with_importance_seconds'] = meta['search_and_apply_seconds'] + imp['importance_seconds']
                    # Plan is json, not checkpoint metadata, to keep artifacts inspectable.
                    write_json(folder / 'mask_plan.json', meta.pop('plan'))
                else:
                    model,meta = baseline(model, method, group['classes'], data, args, seed)
                meta['forget_classes'] = group['classes']
                save(path, checkpoint(model, data, seed, meta))
                write_json(folder / 'method_metadata.json', meta)


@torch.inference_mode()
def logits_for(path, data, args):
    file = path.parent / 'evaluation_logits.npz'
    if file.exists():
        with np.load(file) as bundle:
            return bundle['logits']
    model,_ = from_checkpoint(path, data, args)
    model.eval()
    logits = np.full((len(data.labels),len(data.classes)), np.nan, dtype=np.float32)
    for split in (('test',) if data.shared_data else ('train','attack_cal','test')):
        indices = data.indices(split)
        values = []
        for x,y in data.loader(indices=indices):
            z = model(x.to(args.device))
            if not torch.isfinite(z).all():
                raise RuntimeError('Nonfinite predictions: '+str(path))
            values.append(z.cpu().numpy())
        logits[indices] = np.concatenate(values)
    np.savez_compressed(file, logits=logits)
    return logits


@torch.inference_mode()
def deployment(path, data, args):
    model,_ = from_checkpoint(path, data, args)
    model.eval()
    x,_ = next(iter(data.loader('test')))
    x = x[:1].to(args.device)
    for _ in range(args.latency_warmup):
        model(x)
    samples = []
    for _ in range(args.latency_repeats):
        sync(args.device)
        start = time.perf_counter()
        model(x)
        sync(args.device)
        samples.append((time.perf_counter()-start)*1000)
    nbytes = path.stat().st_size
    return {'checkpoint_bytes':nbytes, 'parameter_count':sum(p.numel() for p in model.parameters()),
            'inference_batch1_ms_median':float(np.median(samples)),
            'inference_batch1_ms_p95':float(np.percentile(samples,95)),
            'device':args.device,
            'hardware':torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else platform.processor(),
            'ideal_transfer_seconds':8*nbytes/(args.link_mbps*1e6),
            'link_mbps':args.link_mbps,
            'transfer_note':'Analytical payload/rate lower bound; no wireless link experiment',
            'latency_note':'Measured on this host; excludes decoding, preprocessing and host/device transfer'}


def evaluate(data, args, output, requests):
    for seed in args.seeds:
        for group in groups_for_seed(requests, seed, data.classes):
            forget = group['classes']
            ci, ti = data.indices('attack_cal'), data.indices('test')
            ref_z = {}
            for reference_seed in args.seeds:
                path = reference_checkpoint(output, seed, group, reference_seed)
                saved = load(path)
                if sorted(saved['metadata'].get('forget_classes', [])) != sorted(forget):
                    raise ValueError('CMIA reference erases different classes: ' + str(path))
                ref_z[reference_seed] = logits_for(path,data,args)
            mids = None if data.shared_data else data.mia_indices(forget, seed)
            write_json(output / f"seed_{seed}/{group['name']}/mia_indices.json", mids)
            for method in dict.fromkeys(['Original','Retrain'] + args.methods):
                path = (output / f'seed_{seed}/Original/best.pth' if method == 'Original' else
                        output / f"seed_{seed}/{group['name']}/{method}/best.pth")
                if not path.exists():
                    raise FileNotFoundError('Run --action all or unlearn first: '+str(path))
                target = output / f"seed_{seed}/{group['name']}/{method}/result.json"
                if target.exists():
                    saved_result = json.loads(target.read_text(encoding='utf-8'))
                    if saved_result['forget_classes'] != forget:
                        raise ValueError('Cached evaluation has a different request.')
                    continue
                z = logits_for(path,data,args)
                if data.shared_data:
                    mia = unavailable_mia()
                else:
                    scores = {key:log_true_probability(z[ids], data.labels[ids]) for key,ids in mids.items()}
                    mia = mia_metrics(**scores)
                # Retrain evaluation uses other seeds only, avoiding self-reference leakage.
                refs = [(v[ci],v[ti]) for s,v in ref_z.items() if method != 'Retrain' or s != seed]
                selection_refs = [(v[ci],v[ti]) for v in ref_z.values()]
                cmia = cmia_evaluate(z[ci],z[ti],refs,data.labels[ci],data.labels[ti],forget,selection_refs)
                per_class = cmia.pop('per_class')
                fi = data.indices('test', forget)
                ri = data.indices('test', sorted(set(range(len(data.classes)))-set(forget)))
                meta = load(path)['metadata']
                record = {'seed':seed, 'request':group['name'], 'forget_classes':forget,
                          'forget_names':group['class_names'], 'method':method,
                          'F':float(np.mean(z[fi].argmax(1)==data.labels[fi])*100),
                          'R':float(np.mean(z[ri].argmax(1)==data.labels[ri])*100),
                          **mia, **cmia, **deployment(path,data,args),
                          'unlearning_seconds':meta.get('unlearning_seconds',0.),
                          'importance_seconds':meta.get('importance_seconds',0.),
                          'search_seconds':meta.get('search_seconds',0.),
                          'mask_apply_seconds':meta.get('mask_apply_seconds',0.),
                          'total_with_importance_seconds':meta.get('total_with_importance_seconds',meta.get('unlearning_seconds',0.)),
                          'smoke_only':args.smoke,
                          'data_protocol':data.protocol,
                          'evaluation_scope':'all_dataset_resubstitution' if data.shared_data else 'held_out_test',
                          'cmia_calibration_test_shared':data.shared_data}
                write_json(target, record)
                write_json(target.with_name('cmia_per_class.json'), per_class)
                print('RESULT',seed,group['name'],method,'F={:.2f} R={:.2f} MIA-AUC={} CMIA-gap={:.2f}pp'.format(
                    record['F'],record['R'],record['mia_auc'],record['cmia_gap_pp']), flush=True)


def summarize(output):
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in output.glob('seed_*/forget_*/*/result.json')]
    if not rows:
        raise ValueError('No completed evaluation results')
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (output / 'results.csv').open('w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    metrics = ['F','R','mia_auc','cmia_response_pct','cmia_gap_pp','cmia_model_auc_extension',
               'unlearning_seconds','importance_seconds','search_seconds','mask_apply_seconds',
               'total_with_importance_seconds','checkpoint_bytes','inference_batch1_ms_median','ideal_transfer_seconds']
    summary = []
    expected = json.loads((output / 'config.json').read_text(encoding='utf-8'))['seeds']
    for req,method in sorted(set((r['request'],r['method']) for r in rows)):
        selected = [r for r in rows if r['request']==req and r['method']==method]
        item = {'request':req,'method':method,'n':len(selected),
                'data_protocol':selected[0].get('data_protocol','disjoint'),
                'evaluation_scope':selected[0].get('evaluation_scope','held_out_test'),
                'complete':sorted(r['seed'] for r in selected)==sorted(expected)}
        for metric in metrics:
            v = [r[metric] for r in selected if r[metric] is not None]
            item[metric+'_mean'] = float(np.mean(v)) if v else None
            item[metric+'_std'] = float(np.std(v,ddof=1)) if len(v)>1 else None
        summary.append(item)
    write_json(output / 'summary.json', summary)
    with (output / 'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    print('Saved:',output/'summary.csv',flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='./data/NWPU-RESISC45')
    p.add_argument('--output',default='./results_resisc45')
    p.add_argument('--action',choices=['all','train','importance','unlearn','evaluate','summarize'],default='all')
    p.add_argument('--device',default='cuda')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--data-protocol', choices=['all_shared','disjoint'], default='all_shared',
                   help='Default: train/validation/test all use the entire dataset; disjoint is optional.')
    p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--methods',nargs='+',choices=['Ours','Retrain','NegativeGradient','SalUn','FineTune'],
                   default=['Ours','Retrain','NegativeGradient','SalUn'])
    p.add_argument('--anchor',default='airport',help='Legacy config field; random requests do not use it.')
    p.add_argument('--forget-counts',type=int,nargs='+',default=[1,3])
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--lr',type=float,default=.1)
    p.add_argument('--weight-decay',type=float,default=5e-4)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--image-size',type=int,default=224)
    for name in ('p','q'):
        p.add_argument('--'+name+'-min',type=float,default=.01)
        p.add_argument('--'+name+'-max',type=float,default=2.)
        p.add_argument('--'+name+'-step',type=float,default=.01)
    p.add_argument('--cam-aggregation',choices=['peak','spatial_mean'],default='peak')
    p.add_argument('--depth-scaling',action='store_true')
    p.add_argument('--importance-samples-per-class',type=int,default=0,help='0 = all training images')
    p.add_argument('--search-samples-per-class',type=int,default=0,help='0 = all validation images')
    p.add_argument('--mia-samples-per-class',type=int,default=150)
    p.add_argument('--unlearn-epochs',type=int,default=5)
    p.add_argument('--ng-lr',type=float,default=1e-4)
    p.add_argument('--baseline-lr',type=float,default=.001)
    p.add_argument('--salun-fraction',type=float,default=.5)
    p.add_argument('--latency-warmup',type=int,default=10)
    p.add_argument('--latency-repeats',type=int,default=50)
    p.add_argument('--link-mbps',type=float,default=10.)
    p.add_argument('--smoke',action='store_true',help='Explicit tiny synthetic plumbing test, NOT experiment results')
    a=p.parse_args()
    if a.cam_aggregation != 'peak' or a.depth_scaling:
        p.error('Paper method requires --cam-aggregation peak and no --depth-scaling')
    if a.smoke:
        a.image_size=32; a.epochs=1; a.unlearn_epochs=1; a.workers=0; a.batch_size=8
        a.p_min=.01; a.p_max=.02; a.p_step=.01
        a.q_min=.01; a.q_max=.02; a.q_step=.01
        a.forget_counts=[1,2]; a.importance_samples_per_class=2
        a.latency_warmup=1; a.latency_repeats=3
        a.mia_samples_per_class=3
        print('SMOKE MODE: small grid and synthetic data; NOT publication results',flush=True)
    if len(set(a.seeds)) < 3 or len(a.seeds)!=len(set(a.seeds)):
        p.error('Use at least three distinct seeds for repeated experiments and CMIA references')
    for name in ('p','q'):
        grid(getattr(a,name+'_min'),getattr(a,name+'_max'),getattr(a,name+'_step'))
    if min(a.epochs,a.batch_size,a.image_size,a.latency_repeats,a.link_mbps,a.mia_samples_per_class,a.unlearn_epochs) <= 0:
        p.error('Epochs, sizes, counts and link rate must be positive')
    if not 0 < a.salun_fraction <= 1 or min(a.workers,a.importance_samples_per_class,a.search_samples_per_class) < 0:
        p.error('Invalid saliency fraction or negative data-loader limits')
    return a


def main():
    args=parse_args()
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.action=='summarize':
        summarize(output); return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run inside apptainer --nv or explicitly pass --device cpu')
    torch.set_num_threads(args.threads)
    config = {k:v for k,v in vars(args).items() if k not in ('action','output','data_root','methods','device','workers','threads')}
    source_dir = Path(__file__).resolve().parent
    config['implementation_sha256'] = digest({n:(source_dir/n).read_text(encoding='utf-8')
        for n in ('run_resisc45.py','data.py','methods.py','privacy.py','camrf_core.py','mask_core.py')})
    config['request_protocol_sha256'] = file_sha256(source_dir / 'random_requests.py')
    cpath=output/'config.json'
    if cpath.exists() and json.loads(cpath.read_text(encoding='utf-8')) != config:
        raise ValueError('Scientific configuration changed. Use a NEW --output directory; cached results will not be silently reused.')
    write_json(cpath,config)
    write_json(output/'environment.json',{'torch':torch.__version__,'torchvision':torchvision.__version__,
        'python':platform.python_version(),'device':args.device,'threads':args.threads,'platform':platform.platform()})
    data=Data(args)
    if len(set(args.forget_counts))!=len(args.forget_counts) or any(n<1 or n>len(data.classes)-2 for n in args.forget_counts):
        raise ValueError('Distinct positive forget counts must leave at least two retained classes for CMIA')
    if args.action in ('all','train'):
        for seed in args.seeds:
            train_model(data,args,seed,output/f'seed_{seed}/Original')
    requests=make_plan(data,args,output)
    if args.action=='train':
        return
    if args.action=='importance':
        for seed in args.seeds:
            compute_importance(data,args,output,seed)
        return
    if args.action in ('all','unlearn'):
        unlearn(data,args,output,requests)
    if args.action in ('all','evaluate'):
        evaluate(data,args,output,requests)
        summarize(output)


if __name__=='__main__':
    main()
