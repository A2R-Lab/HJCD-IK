"""Frozen, local two-endpoint campaign. --check is CPU-only; --run needs a quiet slot."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
TASKS = HERE.parent
OLD = TASKS / 'timing-prebuild-2026-09-26'
NEW = TASKS / 'premerge-2026-09-27'
PYTHON = OLD / 'runtime/bin/python'
REFERENCE = NEW / 'receipt-verification'
MODULE = 'hjcdik/_hjcdik.cpython-312-x86_64-linux-gnu.so'
ENDPOINTS = {
    'baseline': dict(site=str(OLD / 'current-panda/site'), commit='209be95',
                     module_hash='764540a93c3c812b4a6365113e418074fb6f09f93195eb35f60a1b2269702cff',
                     header_hash='2e4a7433cc72f001bfdb452ca9b31f5ce3b019adf9e0a9f4894ea14fb6a800ef'),
    'latest': dict(site=str(NEW / 'timing-site'), commit='516b988',
                   module_hash='aa5a9ff7d673dcd4ac82d4e6ca97ad7297d484a55b6c1679c5dc21735672d430',
                   header_hash='3f6c65fd9e8e91c7ae50ee072215f45d4717edc8fe11a399ebc84e07922a802d'),
}
INPUT_HASHES = {
    'panda.json': 'f122146a74d01322a5fc6bb7a4175ef762438c1b8ceda1687691094c73b4e095',
    'mb_problems.json': '69a3eae85e68d8cf696280e12407ac870b887b59a2913b52c4214524bc55dc94',
}
REFERENCE_HASHES = {
    'benchmark/gen_targets.py': '093bedcee45b240d500188747878ca77c0eec9db6f2698c3ba6ea04a30fc16f7',
    'benchmark/panda_collision.py': '55a5dc7e64f3cb16252d7fc9d5cc2f88459b97d45bac02244269a30c54d62d10',
    'benchmark/panda_model.py': '33704dcef13bc53bb15fb84c42d5ac2c32bada24abbde76b61e17d780a1b649f',
    'benchmark/collision_check.py': 'e268a78fa75304e7d5e2af92b5360c1b111ab1ee14db1740d8958f2b11514685',
    'benchmark/reference/panda_collision_model.cuh': 'c1cc0653acf8d24fed163c623c599dec2dc8bb8ee12ed372ed09bfeba745fe7b',
    'csrc/urdf/panda.urdf': '90bb0a412eb12a3af0abb9d5d0f06be3dbbe1a72806b0416dd9769992cb35079',
    'external/foam/assets/panda/smaller_panda_spherized.urdf': 'f90e26dcce184014335ce013f62a563425d8b82ba5fecf8a71480a5e0722853e',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def child_env(label):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('HJCD', 'CUDA_', 'PYTHON'))
           and k not in ('LD_PRELOAD', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
    env.update(PYTHONPATH=ENDPOINTS[label]['site'], PYTHONSAFEPATH='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', HJCD_LM_WARPS='1',
               HJCD_LM_EPS_POS='1e-8', HJCD_LM_EPS_ORI='1e-8')
    return env


def inspect_endpoint(label):
    import numpy as np
    import hjcdik
    import hjcdik._hjcdik as native
    endpoint = ENDPOINTS[label]
    require(np.__version__ == '2.5.1', 'Expected matched NumPy 2.5.1')
    path = Path(endpoint['site']) / MODULE
    require(Path(native.__file__).resolve() == path.resolve(), 'Wrong native import path')
    require(sha(path) == endpoint['module_hash'], 'Native module hash mismatch')
    require(sha(Path(endpoint['site']) / 'hjcdik/__init__.py') ==
            '4b792e511cdbad8d10ae9b80529bedf85dee5cd66c7c3ef14781faccf6e6246f', 'Wrapper hash mismatch')
    info = hjcdik.build_info()  # Explicitly host-only: does not initialize CUDA.
    require(info == dict(num_joints=7, collision_enabled=True,
                        grid_header_sha256=endpoint['header_hash'],
                        cuda_compiler_version='13.2.86'), 'Build metadata mismatch')
    return dict(module=str(path), numpy=np.__version__, build=info,
                python=sys.version, wrapper_hash=sha(Path(endpoint['site']) / 'hjcdik/__init__.py'))


def cases():
    result = [dict(name='open-s1-fp64', mode='open', document='none', order='switch', solutions=1, precision=1),
              dict(name='open-s4-fp32', mode='open', document='none', order='switch', solutions=4, precision=0)]
    for mode in ('hard', 'both'):
        for document in ('full', 'compact'):
            for order in ('repeat', 'switch'):
                result.append(dict(name=f'{mode}-{document}-{order}', mode=mode,
                                   document=document, order=order, solutions=4, precision=0))
    return result


def workload(case, target_count):
    full = (OLD / 'inputs/mb_problems.json').read_text()
    scenes = json.loads(full)['problems']['box_panda'][:target_count]
    if case['mode'] == 'open':
        entries = json.loads((OLD / 'inputs/panda.json').read_text())['targets'][:target_count]
        targets = [entry['target'] for entry in entries]
        return '', targets, [None] * len(targets)
    require(len(scenes) == target_count, 'Insufficient frozen scenes')
    for scene in scenes:
        require(not (set(scene.get('obstacles', {})) - {'cuboid', 'cylinder'}),
                'Oracle does not handle this obstacle type')
    targets = [s['goal_pose']['position_xyz'] + s['goal_pose']['quaternion_wxyz'] for s in scenes]
    text = full if case['document'] == 'full' else json.dumps({'problems': {'box_panda': scenes}}, separators=(',', ':'))
    return text, targets, scenes


def schedule(count, repeats, order, round_number):
    indices = list(range(count))
    if round_number % 2:
        indices.reverse()
    if order == 'repeat':
        return [i for i in indices for _ in range(repeats)]
    return indices * repeats


def oracle():
    sys.path.insert(0, str(REFERENCE / 'benchmark'))
    import gen_targets as fk
    from panda_collision import mb_instance_to_world_dict, panda_config_collision_free
    from panda_model import collision_model_metadata
    chain = fk._chain_to_target(fk._parse_joints(REFERENCE / 'csrc/urdf/panda.urdf'), 'panda_grasptarget_hand')
    return fk, chain, mb_instance_to_world_dict, panda_config_collision_free, collision_model_metadata('hjcd')


def preflight():
    for name, expected in INPUT_HASHES.items():
        require(sha(OLD / 'inputs' / name) == expected, f'Changed input {name}')
    for name, expected in REFERENCE_HASHES.items():
        require(sha(REFERENCE / name) == expected, f'Changed oracle source {name}')
    metadata = {}
    for label in ENDPOINTS:
        command = [str(PYTHON), str(Path(__file__).resolve()), '--inspect', '--label', label]
        out = subprocess.check_output(command, env=child_env(label), cwd='/tmp', text=True, timeout=30)
        metadata[label] = json.loads(out)
    oracle()  # Load all CPU validation dependencies, including sphere geometry.
    for case in cases():
        workload(case, 32)
    return metadata


def validate(raw, targets, scenes):
    import numpy as np
    fk, chain, to_world, collision_free, model = oracle()
    lo, hi = fk._actuated_limits(chain)
    worlds = [to_world(scene) if scene is not None else None for scene in scenes]
    rows = []
    for index, ms, out, cache_hit in raw:
        q, poses = np.asarray(out['joint_config']), np.asarray(out['pose'])
        count = int(out['count'])
        require(count >= 0 and q.shape == (count, 7) and poses.shape == (count, 7), 'Unexpected output shape')
        require(np.isfinite(q).all() and np.isfinite(poses).all(), 'Nonfinite solver output')
        within_limits = bool(((q >= lo - 1e-5) & (q <= hi + 1e-5)).all())
        target = np.asarray(targets[index])
        target_quat = target[3:] / np.linalg.norm(target[3:])
        pe, oe, agreement, valid = [], [], [], []
        for qi, pose in zip(q, poses):
            transform = fk._fk(chain, qi)
            quat = fk._quat_from_R(transform[:3, :3])
            pe.append(float(1000 * np.linalg.norm(transform[:3, 3] - target[:3])))
            oe.append(float(2 * np.arccos(np.clip(abs(np.dot(quat, target_quat)), 0, 1))))
            agreement.append(float(np.linalg.norm(transform[:3, 3] - pose[:3])))
            valid.append(bool(collision_free(qi, worlds[index], model='hjcd')) if worlds[index] is not None else True)
        rows.append(dict(index=index, ms=ms, selected_scene_cache_hit=cache_hit,
                         count=count, solved=any(p < 5 and o < 1e-3 and v for p, o, v in zip(pe, oe, valid)),
                         invalid_environment_rows=sum(not v for v in valid), within_limits=within_limits,
                         reference_pos_mm=pe, reference_ori_rad=oe, pose_agreement_m=agreement,
                         reported_pos_mm=np.asarray(out['pos_errors']).tolist(),
                         reported_ori_rad=np.asarray(out['ori_errors']).tolist(),
                         q=q.tolist(), pose=poses.tolist()))
    return rows, model


def worker(args):
    import hjcdik
    metadata = inspect_endpoint(args.label)
    case = cases()[args.case]
    text, targets, scenes = workload(case, args.targets)
    kwargs = dict(batch_size=2000, num_solutions=case['solutions'],
                  refine_fp64=case['precision'], write_stats=False,
                  collision_free=case['mode'] != 'open', collision_mode=case['mode'] if text else 'hard',
                  problems_json_text=text, problem_set_name='box_panda' if text else '')
    def solve(index):
        return hjcdik.generate_solutions(targets[index], problem_idx=index if text else 0, **kwargs)
    start = time.perf_counter_ns()
    solve(0)
    cold_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.monotonic()
    warmup = 0
    previous = 0
    while warmup < 4 or time.monotonic() - start < 0.3:
        previous = warmup % len(targets)
        solve(previous)
        warmup += 1
    raw = []
    for index in schedule(len(targets), args.repeats, case['order'], args.round):
        start = time.perf_counter_ns()
        out = solve(index)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        raw.append((index, elapsed, out, bool(text) and index == previous))
        previous = index
    # All FK, collision checks and serialization occur after all measured calls.
    rows, model = validate(raw, targets, scenes)
    save(args.output, dict(label=args.label, case=case, round=args.round, timestamp=stamp(),
                           metadata=metadata, document_sha256=hashlib.sha256(text.encode()).hexdigest(),
                           document_bytes=len(text.encode()), warmup_calls=warmup, cold_ms=cold_ms,
                           validation_model=model, rows=rows))


def smi(query, kind='gpu'):
    return subprocess.check_output(['nvidia-smi', f'--query-{kind}={query}',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10).strip()


def cpu_sample():
    values = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return sum(values), values[3] + values[4]


def cpu_busy(before, after):
    total = after[0] - before[0]
    return 0.0 if total <= 0 else 1 - (after[1] - before[1]) / total


def telemetry(allowed_pid=None):
    apps = smi('pid,process_name', 'compute-apps')
    foreign = [line for line in apps.splitlines() if line and int(line.split(',')[0]) != allowed_pid]
    return dict(time=stamp(), foreign=foreign, apps=apps, load=os.getloadavg(),
                gpu=smi('uuid,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used'))


def guard(sample, busy, limit):
    require(not sample['foreign'], 'Foreign GPU compute process: ' + str(sample['foreign']))
    require(busy <= limit, f'CPU busy fraction {busy:.1%} exceeds {limit:.1%}; slot contaminated')


def summarize(dest, status):
    import numpy as np
    accepted = set(json.loads((dest / 'run.json').read_text())['completed'])
    results = [json.loads(p.read_text()) for p in sorted(dest.glob('r*-*.json')) if p.stem in accepted]
    groups = {}
    failures = []
    for result in results:
        rows = result['rows']
        if any(row['invalid_environment_rows'] or not row['within_limits']
               or max(row['pose_agreement_m'], default=0) > 1e-5 for row in rows):
            failures.append(f"{result['case']['name']}/{result['label']}/r{result['round']}")
        groups.setdefault(result['case']['name'], {}).setdefault(result['label'], []).append(result)
    lines = ['# HJCD-IK targeted timing', '', f'Campaign status: **{status}**.', '',
             'Environment-only collision oracle; no independent self-collision claim.',
             'Positive delta means latest is slower. Review quality and variability, not just latency.', '',
             '| Case | Paired rounds | Median delta range | Baseline/latest median ms | Baseline/latest p95 ms | Solved calls |',
             '| --- | --- | --- | --- | --- | --- |']
    table = []
    for name, labels in groups.items():
        if set(labels) != set(ENDPOINTS):
            continue
        by_round = {label: {r['round']: r for r in values} for label, values in labels.items()}
        common = sorted(set(by_round['baseline']) & set(by_round['latest']))
        if not common:
            continue
        flat = {label: [row for rnd in common for row in by_round[label][rnd]['rows']] for label in ENDPOINTS}
        stats = {label: dict(median=float(np.median([r['ms'] for r in rows])),
                             p95=float(np.percentile([r['ms'] for r in rows], 95)),
                             solved=sum(r['solved'] for r in rows), calls=len(rows),
                             zero_count=sum(r['count'] == 0 for r in rows),
                             invalid_environment_rows=sum(r['invalid_environment_rows'] for r in rows))
                 for label, rows in flat.items()}
        ratios = [100 * (statistics.median(r['ms'] for r in by_round['latest'][rnd]['rows']) /
                         statistics.median(r['ms'] for r in by_round['baseline'][rnd]['rows']) - 1)
                  for rnd in common]
        a, b = stats['baseline'], stats['latest']
        lines.append(f"| {name} | {len(common)} | {min(ratios):+.2f}% … {max(ratios):+.2f}% | "
                     f"{a['median']:.4f} / {b['median']:.4f} | {a['p95']:.4f} / {b['p95']:.4f} | "
                     f"{a['solved']}/{a['calls']} / {b['solved']}/{b['calls']} |")
        table.append(dict(case=name, paired_rounds=common, delta_percent_by_round=ratios, stats=stats))
    lines.extend(['', f'Quality-check failures: {failures or "none in completed outputs"}.',
                  'A stopped/contaminated campaign is not acceptance evidence; see run.json and telemetry.jsonl.',
                  'Raw outputs retain scene-cache-hit tags, zero counts, FK errors and every returned configuration.'])
    (dest / 'summary.md').write_text('\n'.join(lines) + '\n')
    save(dest / 'summary.json', dict(status=status, quality_failures=failures, comparisons=table))
    return failures


def parent(args):
    # Lock only our campaign. Cross-project serialization is the coordinator's job.
    with (HERE / '.campaign.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dest = HERE / ('results-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        dest.mkdir()
        deadline = time.monotonic() + args.minutes * 60
        import shutil
        shutil.copy2(__file__, dest / 'harness.py')
        shutil.copy2(HERE / 'run.sh', dest / 'launcher.sh')
        run = dict(start=stamp(), status='preflight', settings=vars(args), endpoints=ENDPOINTS,
                   inputs=INPUT_HASHES, oracle=REFERENCE_HASHES, cases=cases(), completed=[],
                   harness_sha256=sha(__file__), launcher_sha256=sha(HERE / 'run.sh'),
                   solver_environment={k: v for k, v in child_env('latest').items()
                                       if k.startswith(('HJCD', 'OMP', 'OPENBLAS'))})
        save(dest / 'run.json', run)
        print(f'Artifacts: {dest}', flush=True)
        log = (dest / 'telemetry.jsonl').open('w')
        def record(sample):
            log.write(json.dumps(sample) + '\n')
            log.flush()
        try:
            run['imports'] = preflight()
            run['hardware'] = smi('uuid,name,driver_version')
            require(len(run['hardware'].splitlines()) == 1, 'This frozen campaign expects one visible GPU')
            # Startup utilization gating only: subsequent readings include our own work.
            last = cpu_sample()
            for _ in range(3):
                time.sleep(1)
                sample = telemetry()
                now = cpu_sample()
                sample['cpu_busy_fraction'] = cpu_busy(last, now)
                guard(sample, sample['cpu_busy_fraction'], args.max_cpu_busy)
                require(float(sample['gpu'].split(',')[1]) <= 10, 'GPU not idle at startup')
                record(sample)
                last = now
            run['status'] = 'running'
            save(dest / 'run.json', run)
            for rnd in range(args.rounds):
                for ci, case in enumerate(cases()):
                    labels = ['baseline', 'latest'] if (rnd + ci) % 2 == 0 else ['latest', 'baseline']
                    for label in labels:
                        require(time.monotonic() < deadline - 10, 'Campaign wall-time budget exhausted')
                        sample = telemetry()
                        guard(sample, 0, args.max_cpu_busy)
                        record(sample)
                        key = f'r{rnd:02d}-{case["name"]}-{label}'
                        output = dest / (key + '.json')
                        command = [str(PYTHON), str(Path(__file__).resolve()), '--worker', '--label', label,
                                   '--case', str(ci), '--round', str(rnd), '--targets', str(args.targets),
                                   '--repeats', str(args.repeats), '--output', str(output)]
                        with (dest / (key + '.log')).open('w') as worker_log:
                            proc = subprocess.Popen(command, env=child_env(label), cwd=dest,
                                                    stdout=worker_log, stderr=subprocess.STDOUT)
                            before = cpu_sample()
                            try:
                                while True:
                                    try:
                                        proc.wait(timeout=2)
                                    except subprocess.TimeoutExpired:
                                        pass
                                    sample = telemetry(proc.pid)
                                    after = cpu_sample()
                                    sample.update(case=key, worker_pid=proc.pid, cpu_busy_fraction=cpu_busy(before, after))
                                    record(sample)
                                    guard(sample, sample['cpu_busy_fraction'], args.max_cpu_busy)
                                    before = after
                                    require(time.monotonic() < deadline, 'Campaign wall-time budget exhausted')
                                    if proc.poll() is not None:
                                        break
                                require(proc.returncode == 0, f'Worker failed: {key}; inspect its log')
                            finally:
                                if proc.poll() is None:
                                    proc.terminate()
                                    try:
                                        proc.wait(timeout=5)
                                    except subprocess.TimeoutExpired:
                                        proc.kill()
                                        proc.wait()
                        run['completed'].append(key)
                        save(dest / 'run.json', run)
                        print(f'{len(run["completed"])}/{args.rounds * len(cases()) * 2}: {key}', flush=True)
            run['status'] = 'complete'
        except BaseException as exc:
            run['status'] = 'stopped'
            run['reason'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            log.close()
            run['finished'] = stamp()
            save(dest / 'run.json', run)
            failures = summarize(dest, run['status'])
            if failures and run['status'] == 'complete':
                run['status'] = 'quality_failed'
                save(dest / 'run.json', run)
                summarize(dest, run['status'])
            print(f'Summary: {dest / "summary.md"}', flush=True)
        require(not failures, 'Output quality checks failed; inspect summary before accepting timing')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group(required=True)
    for mode in ('check', 'run', 'worker', 'inspect'):
        modes.add_argument('--' + mode, action='store_true')
    ap.add_argument('--minutes', type=float, default=30)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--targets', type=int, default=32)
    ap.add_argument('--repeats', type=int, default=4)
    ap.add_argument('--max-cpu-busy', type=float, default=0.20,
                    help='Abort above this aggregate CPU busy fraction; heuristic, not proof of isolation')
    ap.add_argument('--label', choices=list(ENDPOINTS))
    ap.add_argument('--case', type=int, choices=range(len(cases())))
    ap.add_argument('--round', type=int, default=0)
    ap.add_argument('--output')
    args = ap.parse_args()
    require(args.minutes > 0 and args.rounds >= 1 and 2 <= args.targets <= 32 and args.repeats >= 2,
            'Invalid duration/round/target/repeat settings')
    require(0 < args.max_cpu_busy <= 1, 'Invalid CPU busy fraction')
    if args.inspect:
        require(args.label is not None, '--inspect requires --label')
        print(json.dumps(inspect_endpoint(args.label)))
    elif args.worker:
        require(args.label is not None and args.case is not None and args.output, 'Missing worker arguments')
        worker(args)
    elif args.check:
        print(json.dumps(dict(status='CPU-only preflight passed; no solves or timings', imports=preflight(),
                              cases=cases(), expected_workers=args.rounds * len(cases()) * 2), indent=2))
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        parent(args)


if __name__ == '__main__':
    main()
