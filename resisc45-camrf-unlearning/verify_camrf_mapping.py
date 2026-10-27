#!/usr/bin/env python3
"""Validate CAM-RF receptive-field geometry and CAM regional means on one airport image.
Run in the RESISC45 project folder beside camrf_core.py and data.py.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torchvision import models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import camrf_core as core
from data import Data

LAYERS = ('layer1.0.conv1','layer2.0.conv1','layer3.0.conv1','layer4.1.conv2')

def oracle_geometry(model):
    """Independent integer-coordinate receptive field bounds, including residual path unions.
    Coordinates are on the infinite padded input lattice; clamp only when plotting.
    Returns (left_edge, right_edge, stride) for a feature at index zero (inclusive).
    """
    def conv(g, m):
        lo, hi, jump = g
        k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
        s = m.stride[0] if isinstance(m.stride, tuple) else m.stride
        if s is None: s = k
        p = m.padding[0] if isinstance(m.padding, tuple) else m.padding
        d = m.dilation[0] if isinstance(m.dilation, tuple) else m.dilation
        return (lo-p*jump, hi+((k-1)*d-p)*jump, jump*s)
    state = conv((0,0,1), model.conv1)
    out = {'conv1':state}
    state = conv(state, model.maxpool)
    for stage_name in ('layer1','layer2','layer3','layer4'):
        for i, block in enumerate(getattr(model,stage_name)):
            old = state
            first = conv(old, block.conv1)
            out[f'{stage_name}.{i}.conv1'] = first
            second = conv(first, block.conv2)
            out[f'{stage_name}.{i}.conv2'] = second
            shortcut = old
            if block.downsample is not None:
                for m in block.downsample:
                    if isinstance(m,(torch.nn.Conv2d,torch.nn.MaxPool2d,torch.nn.AvgPool2d)):
                        shortcut = conv(shortcut,m)
            if second[2] != shortcut[2]:
                raise AssertionError('Residual stride mismatch')
            state = (min(second[0],shortcut[0]),max(second[1],shortcut[1]),second[2])
    return out

def core_bounds(info, u, length):
    center = info.start + u*info.jump
    low = max(0,min(length-1,int(np.floor(center-info.rf/2))))
    high = max(1,min(length,int(np.ceil(center+info.rf/2))))
    return low,max(low+1,high)

def oracle_bounds(g,u,length):
    lo,hi,jump=g
    return max(0,lo+u*jump),min(length,hi+u*jump+1)

def parse():
    p=argparse.ArgumentParser()
    root=Path(__file__).resolve().parent
    p.add_argument('--project-dir',type=Path,default=root)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--results-root',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=Path('results_camrf_rf_verification'))
    p.add_argument('--class-name',default='airport')
    p.add_argument('--image-index',type=int,default=0,help='Index among airport images in train loader')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--mean',nargs=3,type=float,default=[0.485,0.456,0.406],help='For display only; match data.py normalization')
    p.add_argument('--std',nargs=3,type=float,default=[0.229,0.224,0.225],help='For display only; match data.py normalization')
    return p.parse_args()

def main():
    args=parse()
    root=args.project_dir
    args.data_root=args.data_root or root/'data/NWPU-RESISC45'
    args.results_root=args.results_root or root/'results_resisc45_random_complete'
    args.checkpoint=args.checkpoint or args.results_root/'seed_42/Original/best.pth'
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if not isinstance(checkpoint,dict) or 'classes' not in checkpoint or 'state_dict' not in checkpoint:
        raise ValueError('Expected original RESISC45 checkpoint with classes/state_dict')
    classes=checkpoint['classes']
    if args.class_name not in classes: raise ValueError(f'{args.class_name} not found in checkpoint classes')
    target=classes.index(args.class_name)
    config=json.loads((args.results_root/'config.json').read_text())
    data_args=SimpleNamespace(data_root=str(args.data_root),output=str(args.results_root),image_size=config['image_size'],split_seed=config['split_seed'],batch_size=1,workers=args.workers,device=args.device,smoke=False,mia_samples_per_class=1)
    data=Data(data_args)
    if data.classes != classes or data.fingerprint != checkpoint['fingerprint']:
        raise ValueError('Dataset fingerprint or class order differs from checkpoint')
    model=models.resnet18(weights=None,num_classes=len(classes)).to(args.device)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval()
    loader=data.loader('train',classes=[target])
    image=None
    for i,(x,y) in enumerate(loader):
        if i==args.image_index:
            if int(y[0])!=target: raise AssertionError('Image label does not match CAM target')
            image=x.to(args.device)
            break
    if image is None: raise ValueError('Image index exceeds available class images')
    if image.shape[-2:] != (224,224): raise ValueError(f'Expected 224x224 input; got {image.shape[-2:]}')
    conv_layers=core.select_mask_compatible_conv_layers(model)
    rf_table=core.compute_resnet18_rf(model)
    geometry=oracle_geometry(model)
    cam_name,cam_layer,_=core.select_spatial_gradcam_layer(model,input_size=224)
    collector=core.ActivationCollector(conv_layers)
    gradcam=core.GradCAM(model,cam_layer)
    try:
        collector.clear()
        cam=gradcam.generate(image,target_class=target,output_size=(224,224))
        features={k:v.detach().clone() for k,v in collector.outputs.items() if k in LAYERS}
    finally:
        gradcam.close();collector.close()
    cam_np=cam[0].cpu().numpy()
    img=image[0].detach().cpu().permute(1,2,0).numpy()
    img=np.clip(img*np.asarray(args.std)+np.asarray(args.mean),0,1)
    plt.imsave(args.output/'airport_input.png',img)
    plt.imsave(args.output/'airport_gradcam.png',cam_np,cmap='jet',vmin=0,vmax=1)
    rows=[]
    for layer in LAYERS:
        feature=features[layer]
        H,W=feature.shape[-2:]
        coords=sorted(set([(0,0),(H//2,W//2),(H-1,W-1),(H//4,W//4),(3*H//4,3*W//4)]))
        # Independently check *all* positions in feature map for RF boundary agreement.
        mismatches=0
        for u in range(H):
            a=core_bounds(rf_table[layer],u,224)
            b=oracle_bounds(geometry[layer],u,224)
            if a!=b: mismatches+=1
        # compare integral implementation to direct averaging for all positions
        rf_cam=core.mean_cam_in_receptive_fields(cam,H,W,rf_table[layer])[0].cpu().numpy()
        max_error=0.0
        for u in range(H):
            y0,y1=core_bounds(rf_table[layer],u,224)
            for v in range(W):
                x0,x1=core_bounds(rf_table[layer],v,224)
                direct=float(cam_np[y0:y1,x0:x1].mean())
                max_error=max(max_error,abs(direct-float(rf_cam[u,v])))
        fig,axes=plt.subplots(1,2,figsize=(12,5.6))
        for ax in axes:
            ax.imshow(img)
            ax.set_xlim(0,224);ax.set_ylim(224,0)
            ax.set_xticks([]);ax.set_yticks([])
        axes[1].imshow(cam_np,cmap='jet',alpha=.50,vmin=0,vmax=1)
        for j,(u,v) in enumerate(coords):
            y0,y1=core_bounds(rf_table[layer],u,224)
            x0,x1=core_bounds(rf_table[layer],v,224)
            oy0,oy1=oracle_bounds(geometry[layer],u,224)
            ox0,ox1=oracle_bounds(geometry[layer],v,224)
            value=float(rf_cam[u,v]);direct=float(cam_np[y0:y1,x0:x1].mean())
            rows.append(dict(layer=layer,feature_y=u,feature_x=v,core_box=[x0,y0,x1,y1],oracle_box=[ox0,oy0,ox1,oy1],box_match=(x0,y0,x1,y1)==(ox0,oy0,ox1,oy1),integral_cam_mean=value,direct_cam_mean=direct,abs_error=abs(value-direct)))
            for ax in axes:
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=f'C{j}',linewidth=2))
                ax.text(x0+2,y0+10,str(j),color='white',fontsize=9,bbox={'facecolor':f'C{j}','alpha':.8,'pad':1})
        axes[0].set_title(f'{layer}: RF boxes on airport image')
        axes[1].set_title(f'True-label Grad-CAM ({cam_name}) + RF boxes')
        fig.suptitle(f'feature={H}x{W} | RF={rf_table[layer].rf:g} | jump={rf_table[layer].jump:g}\nall-row RF boundary mismatches={mismatches} | max CAM mean error={max_error:.2e}')
        fig.tight_layout()
        fig.savefig(args.output/(layer.replace('.','_')+'_rf_check.png'),dpi=180)
        plt.close(fig)
        print(f'{layer}: feature={H}x{W}, RF={rf_table[layer].rf}, geometry row mismatches={mismatches}/{H}, max direct-vs-integral error={max_error:.3e}',flush=True)
    (args.output/'rf_check_details.json').write_text(json.dumps({'class_name':args.class_name,'class_index':target,'gradcam_layer':cam_name,'image_index':args.image_index,'sampled_locations':rows},indent=2),encoding='utf-8')
    print('Saved:',args.output.resolve())
    print('NOTE: Geometry oracle verifies theoretical support only, not effective gradient-based receptive fields.')

if __name__=='__main__':main()
