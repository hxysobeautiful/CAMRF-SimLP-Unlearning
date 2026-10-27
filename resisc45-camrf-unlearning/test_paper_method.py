"""Independent numerical checks for the supplied PaperCAMRF and Sim-LP equations."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

import camrf_core as core
from methods import importance, plan
from mask_core import prepare_similarity_policy, build_similarity_masks
from run_resisc45_ablation import prepare_policy, make_masks, load_importance


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.)

    def forward(self, x):
        return self.conv(x).sum((2, 3))


class PaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gradcam_native_normalization_precedes_resize(self):
        model = TinyModel().eval()
        cam = core.GradCAM(model, model.conv)
        try:
            value = cam.generate(torch.tensor([[[[1., 3.], [2., 4.]]]]), 0, (1, 1))
            torch.testing.assert_close(value, torch.tensor([[[.625]]]))
        finally:
            cam.close()

    def test_regional_cam_matches_direct_pixel_average(self):
        cam = torch.arange(2*9*11, dtype=torch.float32).reshape(2, 9, 11) / 198
        info = core.RFInfo(rf=7., jump=2., start=.5)
        actual = core.mean_cam_in_receptive_fields(cam, 5, 6, info)
        expected = torch.empty_like(actual)
        for y in range(5):
            for x in range(6):
                y0, y1 = max(0, 2*y-3), min(9, 2*y+4)
                x0, x1 = max(0, 2*x-3), min(11, 2*x+4)
                expected[:, y, x] = cam[:, y0:y1, x0:x1].mean((1, 2))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_paper_relu_peak_no_exponent_and_sample_weighted_mean(self):
        model = TinyModel().eval()
        # On all-negative responses, ReLU argmax uses the first position,
        # whereas raw argmax uses the last. Uneven batches check sample means.
        x = torch.tensor([[[[-4., -3.], [-2., -1.]]],
                          [[[1., 2.], [3., 4.]]],
                          [[[1., 5.], [2., 3.]]]])
        loader = DataLoader(TensorDataset(x, torch.zeros(3, dtype=torch.long)), batch_size=2)
        maps = torch.tensor([[.2, .4], [.6, .8]])
        class FixedCAM:
            def __init__(self, *args): pass
            def generate(self, images, *args):
                model(images)
                return maps.expand(len(images), -1, -1)
            def close(self): pass
        with patch.object(core, 'select_mask_compatible_conv_layers', return_value={'conv': model.conv}), \
             patch.object(core, 'compute_resnet18_rf', return_value={'conv': core.RFInfo()}), \
             patch.object(core, 'select_spatial_gradcam_layer', return_value=('conv', model.conv, (2, 2))), \
             patch.object(core, 'GradCAM', FixedCAM):
            result = core.compute_paper_camrf_importance(model, loader, 'cpu', 0, {'conv': torch.tensor([2.])}, 2)
        torch.testing.assert_close(result['conv'], torch.tensor([2*(.2+.8+.4)/3]))
        self.assertEqual(len(model.conv._forward_hooks), 0)

    def test_main_importance_dispatches_paper_variant(self):
        model = TinyModel().eval()
        loader = object()
        data = SimpleNamespace(classes=['a', 'b'], loader=lambda **kw: loader)
        args = SimpleNamespace(cam_aggregation='peak', importance_samples_per_class=3, image_size=32)
        with patch.object(core, 'compute_class_channel_activations', return_value={}), \
             patch.object(core, 'compute_fudp_tfidf_importance', return_value={'conv': torch.ones(1)}), \
             patch.object(core, 'compute_paper_camrf_importance', return_value={'conv': torch.ones(1)}) as call:
            result = importance(model, data, 'cpu', args)
            self.assertEqual(set(result), {0, 1})
            self.assertEqual(call.call_count, 2)
            self.assertEqual(call.call_args.kwargs['image_size'], 32)
        args.cam_aggregation = 'spatial_mean'
        with self.assertRaises(ValueError):
            importance(model, data, 'cpu', args)

    def test_simlp_against_direct_equations_and_ablation(self):
        model = resnet18(weights=None, num_classes=4)
        layers = core.select_mask_compatible_conv_layers(model)
        rng = torch.Generator().manual_seed(13)
        profiles = {c: {n: torch.rand(v.out_channels, generator=rng) * (c+1)**3
                        for n, v in layers.items()} for c in range(4)}
        policy = prepare_similarity_policy(model, profiles, [0, 1])
        masks, report = build_similarity_masks(policy, 1.7, .31)
        main = plan(model, profiles, [0, 1], 1.7, .31)
        ab_masks, ab_report = make_masks(prepare_policy(model, profiles, [0, 1], True, False), 1.7, .31)
        for name, layer in layers.items():
            mean = (profiles[2][name].double()+profiles[3][name].double())/2
            expected = set()
            for c in (0, 1):
                target = profiles[c][name].double()
                sim = float(target.dot(mean)/(target.norm()*mean.norm()))
                k = int(min(1.7*(1-sim), 1)*layer.out_channels)
                h = int(min(.31*sim, 1)*layer.out_channels)
                expected |= set(target.argsort(descending=True, stable=True)[:k].tolist()) - set(mean.argsort(descending=True, stable=True)[:h].tolist())
                self.assertAlmostEqual(policy[name]['targets'][c]['similarity'], sim)
            self.assertEqual(set(report[name]['final_pruned_indices']), expected)
            self.assertEqual(main[name]['final_pruned_indices'], ab_report[name]['final_pruned_indices'])
            self.assertTrue(torch.equal(masks[name], ab_masks[name]))
        self.assertIn('conv1', report)

    def test_caps_zero_denominator_floor_and_no_stem_exemption(self):
        model = resnet18(weights=None, num_classes=3)
        layers = core.select_mask_compatible_conv_layers(model)
        imp = {c: {n: torch.zeros(v.out_channels) for n, v in layers.items()} for c in range(3)}
        for scores in imp[0].values(): scores[0] = 1
        full = plan(model, imp, [0], 2., 10.)
        tiny = plan(model, imp, [0], .001, 0.)
        for name, layer in layers.items():
            self.assertEqual(full[name]['final_pruned_count'], layer.out_channels)
            self.assertEqual(full[name]['per_target'][0]['similarity'], 0.)
            self.assertEqual(tiny[name]['final_pruned_count'], 0)
        with self.assertRaises(ValueError): plan(model, imp, [0], 1., 1., True)
        with self.assertRaises(ValueError): plan(model, imp, [0, 1, 2], 1., 1.)

    def test_five_variant_export_selects_last_method(self):
        model = resnet18(weights=None, num_classes=45)
        layers = core.select_mask_compatible_conv_layers(model)
        profiles = {c: {n: torch.ones(v.out_channels) for n, v in layers.items()} for c in range(45)}
        checkpoint = {'classes': [str(c) for c in range(45)], 'fingerprint': 'test', 'seed': 42}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'importance.pth'
            torch.save({core.PAPER_IMPORTANCE_METHOD: profiles,
                        'TFIDF_x_MeanRFCAM': {}, 'classes': checkpoint['classes']}, path)
            result, _ = load_importance(path, model, checkpoint, 'test')
            self.assertEqual(set(result), set(range(45)))
            torch.save({'importance': profiles, 'aggregation': 'peak'}, path)
            with self.assertRaises(ValueError): load_importance(path, model, checkpoint, 'test')

    def test_real_resnet_paper_importance_smoke(self):
        model = resnet18(weights=None, num_classes=3).eval()
        images = torch.randn(3, 3, 32, 32)
        labels = torch.zeros(3, dtype=torch.long)
        layers = core.select_mask_compatible_conv_layers(model)
        base = {n: torch.ones(v.out_channels) for n, v in layers.items()}
        values = core.compute_paper_camrf_importance(model, DataLoader(TensorDataset(images, labels), batch_size=2), 'cpu', 0, base, 32)
        for name, score in values.items():
            self.assertEqual(score.shape, base[name].shape)
            self.assertTrue(torch.isfinite(score).all())
            self.assertTrue(((score >= 0) & (score <= 1+1e-5)).all())
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))


if __name__ == '__main__':
    unittest.main()
