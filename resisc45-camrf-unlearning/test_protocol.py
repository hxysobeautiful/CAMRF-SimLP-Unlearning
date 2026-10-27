"""Run with python -m unittest -v test_protocol.py; no dataset/download required."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torchvision import models
from data import Data
from methods import grid, plan
from mask_core import apply_similarity_adaptive_pruning
from privacy import binary_auc, cmia_evaluate, mia_metrics
import camrf_core as core


def synthetic(root, classes=4, count=30):
    rng=np.random.default_rng(57)
    for c in range(classes):
        folder=Path(root)/f'class_{c}'
        folder.mkdir(parents=True,exist_ok=True)
        for i in range(count):
            x=rng.integers(0,60,(32,32,3),dtype=np.uint8)
            x[:,:,c%3] += 140
            x[c*4:c*4+5,:] = 255
            Image.fromarray(x).save(folder/f'{i:03d}.png')


class ProtocolTests(unittest.TestCase):
    def test_exact_grid(self):
        values=grid(.01,2,.01)
        self.assertEqual(len(values),200)
        self.assertEqual(values[:3],[.01,.02,.03])
        self.assertEqual(values[-1],2.)
        with self.assertRaises(ValueError):
            grid(.01, .025, .01)

    def test_mia_direction_selected_only_on_calibration(self):
        # Calibration says larger = member; test is deliberately reversed.
        r=mia_metrics([2,3],[0,1],[0,1],[2,3])
        self.assertEqual(r['mia_auc'],0.)
        self.assertEqual(r['mia_direction'],1)
        self.assertEqual(binary_auc([1,1],[1,1]),.5)

    def test_splits_and_membership_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data'
            synthetic(root)
            a=SimpleNamespace(data_root=str(root),output=str(Path(tmp)/'out'),image_size=32,
                smoke=True,data_protocol='disjoint',split_seed=2026,batch_size=4,workers=0,device='cpu',mia_samples_per_class=3)
            d=Data(a)
            sets=[set(v) for v in d.split.values()]
            self.assertEqual(len(set.union(*sets)),120)
            for i,x in enumerate(sets):
                for y in sets[i+1:]: self.assertFalse(x&y)
            mi=d.mia_indices([0,1],42)
            self.assertFalse(set(mi['member_cal'])&set(mi['member_test']))
            self.assertFalse(set(mi['nonmember_cal'])&set(mi['nonmember_test']))
            self.assertEqual(len(mi['member_test']),len(mi['nonmember_test']))
            d2=Data(a)
            np.testing.assert_array_equal(d.split['train'],d2.split['train'])

    def test_standard_stem_receptive_field(self):
        m=models.resnet18(weights=None,num_classes=4)
        rf=core.compute_resnet18_rf(m)
        # Assert via dataclass fields instead of relying on field spelling.
        values=vars(rf['layer1.0.conv1'])
        self.assertIn(19.,values.values())  # 7x7 stem + maxpool + 3x3 conv -> RF=19
        self.assertIn(4.,values.values())

    def test_bn_mask_enforces_zero_residual_branch(self):
        m=models.resnet18(weights=None,num_classes=4).eval()
        with torch.no_grad(): m.layer1[0].bn2.bias.fill_(2.)
        apply_similarity_adaptive_pruning(m,{'layer1.0.conv2':{'final_pruned_indices':[0,1]}})
        self.assertTrue(torch.equal(m.layer1[0].bn2.bias[:2],torch.zeros(2)))
        self.assertTrue(torch.equal(m.layer1[0].conv2.weight[:2],torch.zeros_like(m.layer1[0].conv2.weight[:2])))

    def test_cmia_identity_reference(self):
        labels=np.repeat(np.arange(4),12)
        rng=np.random.default_rng(42)
        z=rng.normal(size=(48,4))
        z[np.arange(48),labels]+=3
        r=cmia_evaluate(z,z,[(z,z),(z,z)],labels,labels,[0])
        self.assertEqual(r['cmia_gap_pp'],0.)
        self.assertAlmostEqual(r['cmia_model_auc_extension'],.5)

    def test_zero_importance_follows_top_fraction_with_stable_ties(self):
        m=models.resnet18(weights=None,num_classes=4)
        layers=core.select_mask_compatible_conv_layers(m)
        imp={c:{n:torch.zeros(v.out_channels) for n,v in layers.items()} for c in range(4)}
        mp=plan(m,imp,[0],2.,.01)
        # Paper sets s=0 for zero denominator; Top_1 selects every channel.
        self.assertTrue(all(len(v['final_pruned_indices']) == v['channels'] for v in mp.values()))


if __name__=='__main__': unittest.main()
