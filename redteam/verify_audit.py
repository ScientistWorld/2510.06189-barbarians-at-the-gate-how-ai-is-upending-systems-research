#!/usr/bin/env python3
"""Lightweight consistency and restoration checks; no method computation."""
import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'redteam'
EXP = 'eplb_moe_placement'
SUMMARY = json.loads((HERE / 'summary.json').read_text())
for path in HERE.rglob('*.py'):
    ast.parse(path.read_text())
assert len(SUMMARY['cases']) == 21
for name, row in SUMMARY['cases'].items():
    folder = HERE / name
    assert (folder / 'run.sh').is_file()
    assert (folder / 'policy.py').is_file()
    status = json.loads((folder / 'status.json').read_text())
    assert status['test_command'] == 'bash scripts/evaluate_test.sh'
    assert status['test_command_arguments'] == []
    subprocess.run(['bash', '-n', str(folder / 'run.sh')], check=True)
    scores = json.loads((folder / 'scores_test.json').read_text())
    assert 'results' in scores['experiments'][EXP]
    if status['test_exit']:
        assert row['cohort_rank_delta'] is None
assert all(v == 0 for v in SUMMARY['seed_test_metric_spread'].values())
assert SUMMARY['cases']['better_no_unused_scatter']['cohort_rank_delta'] > 0
assert SUMMARY['cases']['weaker_without_snake']['cohort_rank_delta'] < 0
for name in ('shortcut_claim_one_operation','shortcut_empty_replicas',
             'shortcut_drop_experts','shortcut_inflate_baseline',
             'shortcut_flatten_workload'):
    assert SUMMARY['cases'][name]['cohort_rank_delta'] > 0
for name in ('shortcut_remove_frontier','shortcut_visible_trace_cache',
             'shortcut_ignore_node_groups'):
    assert SUMMARY['cases'][name]['cohort_rank_delta'] == 0
assert SUMMARY['cases']['shortcut_omit_method_from_roster']['metrics'] == {}
# Verify every tracked scientific source, script, score, and data byte against
# the start of the audit. Ignore only framework files and required action.yaml.
tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only',
                                 '1e92b4a'], cwd=ROOT, text=True).splitlines()
checked = []
roots = {'baseline','briefing','data','environment','eval','method','scoring','scripts'}
for name in tracked:
    if name.split('/')[0] not in roots:
        continue
    old = subprocess.check_output(['git','show',f'1e92b4a:{name}'], cwd=ROOT)
    assert (ROOT / name).read_bytes() == old, f'not restored: {name}'
    checked.append(name)
print(f'PASS: 21 saved runs, rank/seed checks, and {len(checked)} original gym files restored.')
