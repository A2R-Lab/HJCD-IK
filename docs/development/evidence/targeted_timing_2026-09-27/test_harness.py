"""CPU-only harness contracts. No real solver calls or CUDA initialization."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location('timing_harness', Path(__file__).with_name('run.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def empty_output():
    return dict(count=0, joint_config=np.empty((0, 7)), pose=np.empty((0, 7)),
                pos_errors=np.empty(0), ori_errors=np.empty(0))


class HarnessTests(unittest.TestCase):
    def test_case_matrix(self):
        self.assertEqual(len(h.cases()), 10)
        self.assertEqual(len({c['name'] for c in h.cases()}), 10)
        self.assertEqual(sum(c['mode'] == 'open' for c in h.cases()), 2)

    def test_schedules_match_counts_and_distinguish_cache_hits(self):
        repeated = h.schedule(32, 4, 'repeat', 0)
        switching = h.schedule(32, 4, 'switch', 0)
        self.assertEqual(Counter(repeated), Counter(switching))
        self.assertEqual(len(repeated), 128)
        self.assertEqual(sum(a == b for a, b in zip(repeated, repeated[1:])), 96)
        self.assertEqual(sum(a == b for a, b in zip(switching, switching[1:])), 0)
        self.assertEqual(h.schedule(32, 4, 'repeat', 1), repeated[::-1])

    def test_compact_input_equivalence_and_index_mapping(self):
        for mode in ('hard', 'both'):
            cs = [c for c in h.cases() if c['mode'] == mode and c['order'] == 'switch']
            full, ft, fs = h.workload(cs[0], 32)
            compact, ct, ss = h.workload(cs[1], 32)
            self.assertEqual(ft, ct)
            self.assertEqual(fs, ss)
            self.assertLess(len(compact), len(full))
            decoded = json.loads(compact)['problems']['box_panda']
            self.assertEqual(decoded, json.loads(full)['problems']['box_panda'][:32])
            for index, target in enumerate(ct):
                goal = decoded[index]['goal_pose']
                self.assertEqual(target, goal['position_xyz'] + goal['quaternion_wxyz'])

    def test_open_targets_match_frozen_source(self):
        text, targets, scenes = h.workload(h.cases()[0], 32)
        self.assertEqual(text, '')
        self.assertEqual(len(targets), 32)
        self.assertEqual(scenes, [None] * 32)

    def test_host_preflight_rejects_hash_mismatch(self):
        with patch.object(h, 'sha', return_value='tampered'):
            with self.assertRaisesRegex(RuntimeError, 'Changed input'):
                h.preflight()

    def test_clean_environment(self):
        with patch.dict(h.os.environ, {'HJCD_UNEXPECTED': '1', 'CUDA_LAUNCH_BLOCKING': '1',
                                      'PYTHONPATH': 'bad', 'LD_PRELOAD': 'bad'}):
            env = h.child_env('latest')
        self.assertNotIn('HJCD_UNEXPECTED', env)
        self.assertNotIn('CUDA_LAUNCH_BLOCKING', env)
        self.assertNotIn('LD_PRELOAD', env)
        self.assertEqual(env['PYTHONPATH'], h.ENDPOINTS['latest']['site'])
        self.assertEqual(env['HJCD_LM_WARPS'], '1')

    def test_contention_guards(self):
        h.guard({'foreign': []}, .1, .2)
        with self.assertRaisesRegex(RuntimeError, 'Foreign GPU'):
            h.guard({'foreign': ['123, competing-job']}, .1, .2)
        with self.assertRaisesRegex(RuntimeError, 'CPU busy'):
            h.guard({'foreign': []}, .3, .2)
        self.assertAlmostEqual(h.cpu_busy((100, 50), (200, 125)), .25)

    def test_zero_solution_validation_is_failure_to_solve_not_crash(self):
        _, targets, scenes = h.workload(h.cases()[0], 2)
        rows, model = h.validate([(0, 1.0, empty_output(), False)], targets, scenes)
        self.assertFalse(rows[0]['solved'])
        self.assertEqual(rows[0]['count'], 0)
        self.assertEqual(model['model'], 'hjcd')

    def test_numerical_validation_on_cpu_fk(self):
        fk, chain, *_ = h.oracle()
        lo, hi = fk._actuated_limits(chain)
        q = (lo + hi) / 2
        transform = fk._fk(chain, q)
        pose = np.r_[transform[:3, 3], fk._quat_from_R(transform[:3, :3])]
        out = dict(count=1, joint_config=q[None, :], pose=pose[None, :],
                   pos_errors=np.zeros(1), ori_errors=np.zeros(1))
        rows, _ = h.validate([(0, 1.0, out, False)], [pose.tolist()], [None])
        self.assertTrue(rows[0]['solved'])
        self.assertTrue(rows[0]['within_limits'])

    def test_worker_uses_keyword_contract_and_never_times_validation(self):
        calls = []
        def fake_solver(*args, **kwargs):
            self.assertEqual(len(args), 1)
            calls.append(kwargs)
            return empty_output()
        fake = types.SimpleNamespace(generate_solutions=fake_solver)
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / 'worker.json'
            args = argparse.Namespace(label='latest', case=2, targets=2, repeats=4, round=0, output=str(output))
            with patch.dict(h.sys.modules, {'hjcdik': fake}), \
                 patch.object(h, 'inspect_endpoint', return_value={}), \
                 patch.object(h.time, 'monotonic', side_effect=[0, 1]):
                h.worker(args)
            result = json.loads(output.read_text())
        self.assertEqual(len(calls), 1 + 4 + 8)  # cold, warm, measured
        self.assertEqual([row['index'] for row in result['rows']], [0]*4 + [1]*4)
        self.assertTrue(all(c['collision_mode'] == 'hard' and c['refine_fp64'] == 0
                            and c['write_stats'] is False for c in calls))
        self.assertEqual(len({c['problems_json_text'] for c in calls}), 1)

    def test_summary_only_accepts_completed_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            h.save(dest / 'run.json', {'completed': ['r00-open-baseline', 'r00-open-latest']})
            for label, ms in [('baseline', 2), ('latest', 1)]:
                row = dict(ms=ms, count=0, solved=False, invalid_environment_rows=0,
                           within_limits=True, pose_agreement_m=[])
                data = dict(label=label, case={'name': 'open'}, round=0, rows=[row])
                h.save(dest / f'r00-open-{label}.json', data)
            # A contaminated/unaccepted output must not enter paired statistics.
            h.save(dest / 'r01-open-latest.json', {'broken': True})
            self.assertEqual(h.summarize(dest, 'stopped'), [])
            report = json.loads((dest / 'summary.json').read_text())
            self.assertEqual(report['comparisons'][0]['delta_percent_by_round'], [-50])
            self.assertEqual(report['status'], 'stopped')

    def test_parent_foreign_gpu_stops_without_worker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # These are synthetic provenance artifacts, never executed.
            (root / 'run.sh').write_text('# test fixture\n')
            args = argparse.Namespace(minutes=30, rounds=3, targets=32, repeats=4, max_cpu_busy=.2)
            with patch.object(h, 'HERE', root), patch.object(h, 'preflight', return_value={}), \
                 patch.object(h, 'smi', return_value='GPU-0, RTX 5090, test-driver'), \
                 patch.object(h, 'telemetry', return_value={'foreign': ['123, peer'], 'gpu': 'GPU-0, 0'}), \
                 patch.object(h, 'cpu_sample', return_value=(100, 100)), \
                 patch.object(h.time, 'sleep'), patch.object(h.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'Foreign GPU'):
                    h.parent(args)
                popen.assert_not_called()
            report = json.loads(next(root.glob('results-*/run.json')).read_text())
            self.assertEqual(report['status'], 'stopped')
            self.assertEqual(report['completed'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
