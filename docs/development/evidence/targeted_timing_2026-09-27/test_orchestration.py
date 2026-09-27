"""CPU-only full parent-loop tests with synthetic subprocesses and GPU telemetry."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('timing_harness', Path(__file__).with_name('run.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class Process:
    def __init__(self, command, **kwargs):
        self.pid = 999999
        self.returncode = 0
        self.terminated = False
        args = dict(zip(command[3::2], command[4::2]))
        label = args['--label']
        row = dict(ms=2 if label == 'baseline' else 1, count=0, solved=False,
                   invalid_environment_rows=0, within_limits=True, pose_agreement_m=[])
        h.save(args['--output'], dict(label=label, case=h.cases()[int(args['--case'])],
                                     round=int(args['--round']), rows=[row]))

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15


class OrchestrationTests(unittest.TestCase):
    def context(self, root, process_factory):
        stack = ExitStack()
        (root / 'run.sh').write_text('# synthetic fixture\n')
        for target, value in [('HERE', root), ('preflight', lambda: {}),
                              ('smi', lambda *a, **k: 'GPU-0, RTX5090, driver'),
                              ('cpu_sample', lambda: (100, 100)),
                              ('telemetry', lambda *a: {'foreign': [], 'gpu': 'GPU-0, 0'})]:
            stack.enter_context(patch.object(h, target, value))
        stack.enter_context(patch.object(h.time, 'sleep'))
        stack.enter_context(patch.object(h.subprocess, 'Popen', side_effect=process_factory))
        stack.enter_context(patch('builtins.print'))
        return stack

    def test_complete_three_round_loop_and_reports(self):
        args = argparse.Namespace(minutes=30, rounds=3, targets=32, repeats=4, max_cpu_busy=.2)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            order = []
            def factory(command, **kwargs):
                process = Process(command, **kwargs)
                order.append(command[command.index('--label') + 1])
                return process
            with self.context(root, factory):
                h.parent(args)
            dest = next(root.glob('results-*'))
            run = json.loads((dest / 'run.json').read_text())
            summary = json.loads((dest / 'summary.json').read_text())
            self.assertEqual(run['status'], 'complete')
            self.assertEqual(len(run['completed']), 60)
            self.assertEqual(order[:4], ['baseline', 'latest', 'latest', 'baseline'])
            self.assertEqual(order[20:22], ['latest', 'baseline'])
            self.assertEqual(len(summary['comparisons']), 10)
            self.assertTrue(all(c['paired_rounds'] == [0, 1, 2] for c in summary['comparisons']))

    def test_deadline_terminates_only_own_worker_and_excludes_output(self):
        args = argparse.Namespace(minutes=30, rounds=3, targets=32, repeats=4, max_cpu_busy=.2)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            children = []
            def factory(command, **kwargs):
                process = Process(command, **kwargs)
                process.returncode = None
                children.append(process)
                return process
            with self.context(root, factory), patch.object(h.time, 'monotonic', side_effect=[0, 1, 2000]):
                with self.assertRaisesRegex(RuntimeError, 'wall-time budget'):
                    h.parent(args)
            self.assertEqual(len(children), 1)
            self.assertTrue(children[0].terminated)
            dest = next(root.glob('results-*'))
            run = json.loads((dest / 'run.json').read_text())
            summary = json.loads((dest / 'summary.json').read_text())
            self.assertEqual(run['status'], 'stopped')
            self.assertEqual(run['completed'], [])
            self.assertEqual(summary['comparisons'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
