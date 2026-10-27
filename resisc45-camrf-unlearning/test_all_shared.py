"""Checks for intentionally identical full-dataset train/validation/test groups."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from data import Data, validate_manifest_indices
from privacy import unavailable_mia
from test_protocol import synthetic
from run_resisc45_baselines import evaluate_logits, save_tables
import run_resisc45 as runner


class AllSharedTests(unittest.TestCase):
    def make_data(self, tmp, protocol=None):
        root = Path(tmp)/'data'
        synthetic(root, classes=4, count=20)
        args = SimpleNamespace(data_root=str(root), output=str(Path(tmp)/'results'),
            image_size=32, smoke=True, split_seed=2026, batch_size=4,
            workers=0, device='cpu', mia_samples_per_class=3)
        if protocol is not None:
            args.data_protocol = protocol
        return Data(args), args

    def test_default_all_sets_are_identical_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, args = self.make_data(tmp)
            self.assertEqual(data.protocol, 'all_shared')
            for name in ('train', 'validation', 'attack_cal', 'test'):
                np.testing.assert_array_equal(data.indices(name), np.arange(80))
                self.assertEqual(len(data.loader(name).dataset), 80)
                self.assertEqual(len(data.indices(name, [1, 2])), 40)
            np.testing.assert_array_equal(Data(args).indices('test'), np.arange(80))
            manifest = json.loads((Path(args.output)/'split.json').read_text())
            self.assertEqual(validate_manifest_indices(manifest), 'all_shared')
            with self.assertRaises(ValueError):
                data.mia_indices([0], 42)
            args.data_protocol = 'disjoint'
            with self.assertRaises(ValueError):
                Data(args)

    def test_manifest_rejects_missing_duplicate_or_reordered_full_sets(self):
        manifest = {'files':['a','b','c'], 'data_protocol':'all_shared',
                    'indices':{k:[0,1,2] for k in ('train','validation','attack_cal','test')}}
        validate_manifest_indices(manifest)
        for invalid in ([0,1], [0,1,1], [2,1,0]):
            manifest['indices']['test'] = invalid
            with self.assertRaises(ValueError): validate_manifest_indices(manifest)

    def test_baselines_all_data_metrics_and_null_mia_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, args = self.make_data(tmp)
            z = np.eye(4)[data.labels]*3
            result = evaluate_logits(z, data, [0], 42, 100.)
            self.assertEqual(result['F_test'], 100.)
            self.assertEqual(result['R_test'], result['R_train'])
            self.assertIsNone(result['mia_auc'])
            self.assertEqual(result['mia_status'], 'unavailable_no_held_out_nonmembers')
            result.update(request='forget_1', method='Original', seed=42,
                          status='OK', unlearning_seconds=0.)
            save_tables(Path(args.output), [result], [42])
            summary = json.loads((Path(args.output)/'summary.json').read_text())
            self.assertIsNone(summary[0]['mia_auc_mean'])

    def test_main_evaluation_and_summary_complete_with_shared_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, args = self.make_data(tmp)
            args.seeds = [42,43,44]
            args.methods = []
            args.latency_repeats = 1
            output = Path(args.output)
            (output/'config.json').write_text(json.dumps({'seeds':args.seeds}))
            group = {'name':'forget_1','classes':[0],'class_names':[data.classes[0]]}
            for seed in args.seeds:
                for name in ('Original', 'forget_1/Retrain'):
                    folder = output/f'seed_{seed}'/name
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder/'best.pth').touch()
            rng = np.random.default_rng(2)
            z = rng.normal(size=(len(data.labels),4)) + np.eye(4)[data.labels]*2
            deploy = dict(checkpoint_bytes=0,inference_batch1_ms_median=0.,ideal_transfer_seconds=0.)
            with patch.object(runner,'groups_for_seed',return_value=[group]), \
                 patch.object(runner,'load',return_value={'metadata':{'forget_classes':[0]}}), \
                 patch.object(runner,'logits_for',return_value=z), \
                 patch.object(runner,'deployment',return_value=deploy):
                runner.evaluate(data,args,output,{})
            runner.summarize(output)
            result = json.loads((output/'seed_42/forget_1/Original/result.json').read_text())
            self.assertIsNone(result['mia_auc'])
            self.assertTrue(result['cmia_calibration_test_shared'])
            self.assertEqual(result['evaluation_scope'],'all_dataset_resubstitution')
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(len(summary),2)
            self.assertTrue(all(r['complete'] and r['mia_auc_mean'] is None for r in summary))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
