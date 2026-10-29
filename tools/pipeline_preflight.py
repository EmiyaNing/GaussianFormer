"""One-command preflight for the GPU queue.

The queue is a chain of nine scripts that runs unattended for hours, and the
individual safety properties are checked by separate tools scattered across this
repo.  Before leaving it alone, this runs all of them and prints a single verdict:

1. **Guard patterns** -- no guard may match a process that is not the thing it
   waits for, and no pattern may be able to match a shell that quotes it.
   (``tools/queue_guard_audit.py``; a false positive stalls the chain silently.)
2. **Checkpoints** -- every queued job's inputs exist, so a stage cannot fail
   after the GPU has been handed to it.
3. **Configs** -- every queued config loads and resolves to the adapter it claims.
4. **Kernel build** -- the in-place extension is on the default tile edge, so a
   P4-3/P4-4/P4-5 number is not silently quoted against a swept build.
5. **Data** -- the val split's labels and images exist.  Exhaustive by default is
   ~192k stat calls, so ``--data-sample`` limits it for a quick check and
   ``--full-data`` makes it exhaustive.

    python3.10 tools/pipeline_preflight.py
    python3.10 tools/pipeline_preflight.py --full-data
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import pathlib
import re
import subprocess
import sys
from pathlib import Path

from mmengine import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Jobs the queue will run, with the checkpoint(s) each needs.  A missing entry
# means the stage would abort or silently skip -- both worth knowing up front.
QUEUED_JOBS = {
    'tile sweep': [],
    'a2_knn_max': ['ckpts/sparseworld_strict_epoch_56.pth'],
    'P4.3 robustness (frozen A5)': ['out/p1_a5_gaussian/epoch_40.pth'],
    'accepted-degree sweep': [],
    'paired horizon probe': ['out/p1_a5_gaussian/epoch_40.pth'],
    'A4': ['ckpts/sparseworld_strict_epoch_56.pth'],
    'full-val anchors': ['ckpts/sparseworld_strict_epoch_56.pth',
                         'out/a2_knn_mean/epoch_40.pth',
                         'out/p1_a5_gaussian/epoch_40.pth'],
    "A2' (k=215)": ['ckpts/sparseworld_strict_epoch_56.pth'],
    "A2'' (k=59)": ['ckpts/sparseworld_strict_epoch_56.pth'],
    'precision/recall re-eval': ['out/p1_a5_gaussian/epoch_40.pth'],
    'cap/cell interaction probe': ['out/p1_a5_gaussian/epoch_40.pth'],
    'cap curve on other seeds': ['out/a5_s43/epoch_40.pth',
                                 'out/a5_s44/epoch_40.pth'],
    'tile-size holdout': [],
    'seed-pair dumps + bootstrap': ['out/a5_s43/epoch_40.pth',
                                    'out/a2_s43/epoch_40.pth',
                                    'out/a5_s44/epoch_40.pth',
                                    'out/a2_s44/epoch_40.pth'],
}

QUEUED_CONFIGS = [
    'config/opus/ablations/p1_a2_knn_max.py',
    'config/opus/ablations/p1_a2_knn_k215.py',
    'config/opus/ablations/p1_a2_knn_k59.py',
    'config/opus/ablations/p1_a4_fixed_gaussian.py',
    'config/opus/ablations/p1_a0_base.py',
    'config/opus/ablations/p1_a5_gaussian.py',
    'config/opus/ablations/p3_dense_g2v_block.py',
    'config/opus/ablations/p4_fullval_a0.py',
    'config/opus/ablations/p4_fullval_a2.py',
    'config/opus/ablations/p4_fullval_a4.py',
    'config/opus/ablations/p4_fullval_a5.py',
    'config/opus/ablations/p4_rob_cell0p9.py',
    'config/opus/ablations/p4_rob_cell3p6.py',
    'config/opus/ablations/p4_rob_support2.py',
    'config/opus/ablations/p4_rob_support4.py',
    'config/opus/ablations/p4_cap8_cell0p9.py',
    'config/opus/ablations/p4_cap8_cell3p6.py',
]


def check_guards():
    result = subprocess.run([sys.executable, 'tools/queue_guard_audit.py'],
                            cwd=str(ROOT), capture_output=True, text=True)
    return result.returncode == 0, result.stdout.strip().splitlines()[-1:]


def check_checkpoints():
    missing = []
    for job, paths in QUEUED_JOBS.items():
        for path in paths:
            if not (ROOT / path).exists():
                # A4's checkpoint is produced by the queue itself, so its absence
                # is expected; anything else is a real gap.
                missing.append((job, path))
    return missing


def check_configs():
    bad = []
    for name in QUEUED_CONFIGS:
        path = ROOT / name
        if not path.exists():
            bad.append((name, 'missing'))
            continue
        try:
            cfg = Config.fromfile(str(path))
            cfg.model.pts_bbox_head.num_query
        except Exception as error:                      # noqa: BLE001
            bad.append((name, str(error)[:80]))
    return bad


def check_tile_build():
    try:
        from model.ops.block_g2v import CUDA_BLOCK_G2V_AVAILABLE, tile_edge
    except Exception as error:                          # noqa: BLE001
        return None, str(error)[:80]
    if not CUDA_BLOCK_G2V_AVAILABLE:
        return None, 'extension not built'
    return tile_edge(), None


def cited_artifact_paths():
    """Durable artifact paths cited in backticks across the reports.

    Scope is ``benchmarks/results/*``: those are the artifacts a report's numbers
    are supposed to be reproducible from.  ``out/...`` run directories are
    deliberately excluded -- they are transient and several are named per
    variant, so they cannot be audited this way.

    This catches a class the KaTeX/link checker cannot: a typo inside an inline
    code span is not a markdown link, so nothing else validates it.
    """
    pattern = re.compile(
        r'benchmarks/results/[A-Za-z0-9_]+\.(?:jsonl|json|pt)\b')
    cited = set()
    for path in (ROOT / 'analysis/sparse_render/results').glob('*.md'):
        cited.update(pattern.findall(path.read_text(encoding='utf-8')))
    return sorted(cited)


def check_cited_artifacts(cited, promised):
    """Missing citations that the pending-report contract does not promise.

    A cited artifact that is absent is acceptable only when the contract says it
    is still coming (queued).  Anything else is either a typo in the report or an
    artifact that was never produced -- both would make a number unverifiable.
    """
    promised_basenames = {token.rsplit('/', 1)[-1] for token in promised}
    unpromised = []
    missing = []
    for path in cited:
        if (ROOT / path).exists():
            continue
        missing.append(path)
        if path.rsplit('/', 1)[-1] not in promised_basenames:
            unpromised.append(path)
    return missing, unpromised


SECTION_REF_RE = re.compile(r'\]\(([^)\s]+\.md)\)\s*§\s*(\d+(?:\.\d+)*)')
SECTION_HEADING_RE = re.compile(r'^#{2,5}\s+§?\s*(\d+(?:\.\d+)*)', re.M)


def check_section_references(reports_dir=None):
    """Dangling ``[report](file.md) §N`` references.

    The link checker validates that the *file* exists; nothing validated the
    *section*.  These reports are read as an argument, so pointing at "§4.3" of
    a document that only has §4 is a citation the reader cannot follow -- it was
    a real defect (P1-5 -> P4-7 §4.3) found when this check was first written.
    """
    directory = reports_dir or (ROOT / 'analysis/sparse_render/results')
    total = 0
    bad = []
    for path in sorted(directory.glob('*.md')):
        text = path.read_text(encoding='utf-8')
        for match in SECTION_REF_RE.finditer(text):
            target, section = match.group(1), match.group(2)
            total += 1
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                bad.append((path.name, target, section, 'file missing'))
                continue
            headings = set(SECTION_HEADING_RE.findall(
                resolved.read_text(encoding='utf-8')))
            if section not in headings:
                bad.append((path.name, target, section, 'no such section'))
    return total, bad


SCRIPT_REF_RE = re.compile(
    r'(?<![\w/.-])((?:benchmarks|tools|misc|model|tests)/'
    r'[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*\.(?:py|sh))')

# A line that says the cited path is absent, or was delivered under a different
# path, is *documenting a discrepancy* rather than telling the reader how to
# reproduce something -- those citations name files that are meant not to exist.
# Deliberately excludes a bare "或": an "or" between two reproduction commands
# must not be able to mask a renamed script.
ABSENT_MARKERS = ('不存在', '未实现', '未交付', '未产出', '路径不同')


def check_script_references(reports_dir=None):
    """Script paths cited in the reports must exist.

    The artifact checker covers ``benchmarks/results/*`` *data* files; nothing
    covered the *code* paths the reports cite in prose
    (``benchmarks/p1/aggregation_family.py``, ``tools/run_p1_a4.sh``).  Those are
    exactly the paths that get renamed while the prose stays put, leaving the
    reader with a reproduction step they cannot follow.
    """
    directory = reports_dir or (ROOT / 'analysis/sparse_render/results')
    total = 0
    missing = []
    for path in sorted(directory.glob('*.md')):
        for lineno, line in enumerate(
                path.read_text(encoding='utf-8').splitlines(), 1):
            tokens = SCRIPT_REF_RE.findall(line)
            if not tokens:
                continue
            total += len(tokens)
            if any(marker in line for marker in ABSENT_MARKERS):
                continue
            for token in tokens:
                if not (ROOT / token).exists():
                    missing.append((f'{path.name}:{lineno}', token))
    return total, missing


def check_data(sample):
    val = ROOT / 'data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl'
    if not val.exists():
        return None, 'val infos pickle missing'
    infos = pickle.load(val.open('rb'))['infos']
    occ = []
    cams = []
    for frames in infos.values():
        for frame in frames:
            if frame.get('occ_path'):
                occ.append(ROOT / 'data/occ3d' / frame['occ_path'])
            for info in (frame.get('data') or {}).values():
                if isinstance(info, dict) and info.get('filename'):
                    cams.append(ROOT / 'data/nuscenes' / info['filename'])
    random.seed(0)
    if sample and sample < len(cams):
        cams = random.sample(cams, sample)
    missing_occ = sum(1 for p in occ if not p.exists())
    missing_cam = sum(1 for p in cams if not p.exists())
    return (len(occ), missing_occ, len(cams), missing_cam), None


def report_contract_triggers():
    """Trigger artifacts promised by the README's pending-report contract."""
    readme = ROOT / 'analysis/sparse_render/results/README.md'
    if not readme.exists():
        return []
    triggers = []
    in_table = False
    for line in readme.read_text(encoding='utf-8').splitlines():
        if line.startswith('## 待出报告清单'):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith('|'):
            if triggers:
                break
            continue
        first = line.split('|')[1]
        for token in re.findall(r'`([^`]+)`', first):
            # ``p1_eval.json:a4_fixed_gaussian`` -> the file is what matters.
            token = token.split(':')[0].strip().rstrip('/').replace('…', '')
            token = token.rstrip('/')
            if not token or '.' not in token:
                continue
            # A contract cell may name the *producer* next to the artifact
            # (``tools/run_p4_cap_cell_probe.sh``).  Scripts are how artifacts
            # get written, so they are producers, never triggers -- an earlier
            # version special-cased ``benchmarks/p1/`` and then flagged the new
            # script as "unowned".
            if token.endswith(('.sh', '.py')):
                continue
            # Brace/shell expansion (``p1_bootstrap_ci_s{42,43,44}.json``) cannot
            # be matched against a corpus, because the script builds the name from
            # a variable.  Such tokens are skipped rather than reported as
            # unwritten; the contract row keeps a verifiable sibling token (the
            # directory or the combined artifact) so coverage is still asserted.
            if any(marker in token for marker in ('{', '}', '$')):
                continue
            triggers.append(token)
    return triggers


RESULT_RE = re.compile(
    r'benchmarks/results/[A-Za-z0-9_./-]+\.(?:json|pt|jsonl|png|md|txt|csv|log)')

# Artifacts a stage writes but that are *inputs to another stage* rather than
# results of their own: they are consumed by a diagnostic in the same queue and
# need no separate report.  Each entry must say which stage consumes it, so this
# cannot silently become a dumping ground for uncovered results.
INTERMEDIATE_ARTIFACTS = {
    'benchmarks/results/real_points.pt':
        'point dump consumed by the P4 diagnostics in the same stage',
    'benchmarks/results/real_points_grouped.pt':
        'grouped point dump consumed by subgaussian_redundancy / '
        'gaussian_attribute_distribution in the same stage',
}


def queue_stage_artifacts():
    """``{artifact: [stage scripts that name it]}`` for the queue stages.

    Only literal paths are found; a name built from a shell variable is invisible
    here, which is why this check is a *floor* on coverage and not a proof.
    """
    found = {}
    for path in sorted((ROOT / 'tools').glob('run_*.sh')):
        text = path.read_text(encoding='utf-8', errors='replace')
        for token in RESULT_RE.findall(text):
            found.setdefault(token, set()).add(path.name)
    return found


ANALYSIS_SCRIPT_GLOBS = ('benchmarks/**/*.py', 'misc/*.py', 'tools/*.py')


def analysis_artifacts():
    """``{artifact: [scripts]}`` for artifacts named by analysis/probe scripts.

    Complementary to :func:`queue_stage_artifacts`: a queue stage runs a script
    that writes results, but the report machinery also has its own outputs
    (``p1_receptive_field_controls.json`` and friends).  Both kinds must be
    traceable to a write-up.
    """
    found = {}
    for pattern in ANALYSIS_SCRIPT_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file():
                continue
            text = path.read_text(encoding='utf-8', errors='replace')
            for token in RESULT_RE.findall(text):
                found.setdefault(token, set()).add(path.name)
    return found


def check_artifact_coverage(readme=None, reports_dir=None):
    """Every artifact a script writes must be promised a report somewhere.

    ``check_report_contract`` verifies the forward direction (each promised
    artifact has a writer).  This is the reverse: a script that writes a result
    file which no report row and no report text ever mentions would leave an
    experiment with no write-up, which is exactly the objective's stated
    requirement ("每一组实验都需要有对应的 markdown 实验报告").  The corpus is the
    README (where the artifact plan lives) **plus every report** (which may cite
    an artifact directly, e.g. a machinery JSON quoted in a table).

    The first run found two classes of gap: a queue-produced JSON nobody had
    promised (``p4_real_workload_grouped.json``), and nine analysis artifacts --
    including a second output of the tile-holdout script whose *other* output was
    already promised.
    """
    directory = pathlib.Path(reports_dir) if reports_dir else (
        ROOT / 'analysis/sparse_render/results')
    text = (pathlib.Path(readme) if readme else directory / 'README.md'
            ).read_text(encoding='utf-8')
    for path in sorted(directory.glob('*.md')):
        text += '\n' + path.read_text(encoding='utf-8')
    tokens = dict(analysis_artifacts())
    for token, stages in queue_stage_artifacts().items():
        tokens.setdefault(token, set()).update(stages)
    uncovered = {}
    for token, stages in tokens.items():
        if token in INTERMEDIATE_ARTIFACTS:
            continue
        if token in text or token.rsplit('/', 1)[-1] in text:
            continue
        uncovered[token] = sorted(stages)
    return uncovered


def check_report_contract(triggers, corpus=None):
    """Every promised trigger must be *written by something* in the queue.

    A typo in the contract (or a renamed output) is invisible until the data
    lands hours later and the artifact turns up somewhere else.

    Matching is on the **basename** (``p4_robustness.json``), not the full path:
    a work-dir appears in the scripts as a directory, and a bare directory
    fallback would make the check vacuous because parent directories like
    ``benchmarks/results`` occur everywhere.  Glob and directory-shaped tokens
    (``out/p1_persample/*.json``, ``out/a5_s44``) keep their path form instead.
    """
    if corpus is None:
        parts = []
        for pattern in ('tools/*.sh', 'benchmarks/**/*.py', 'misc/*.py',
                        'tools/*.py', '*.py'):
            for path in ROOT.glob(pattern):
                if path.is_file():
                    parts.append(path.read_text(encoding='utf-8', errors='ignore'))
        corpus = '\n'.join(parts)
    unowned = []
    for token in triggers:
        # Brace/shell-expanded names are unverifiable by construction (the writer
        # builds them from a variable), so they are skipped here as well as in
        # ``report_contract_triggers`` -- the rule must not depend on which entry
        # point produced the token.
        if any(marker in token for marker in ('{', '}', '$')):
            continue
        basename = token.rsplit('/', 1)[-1]
        if '*' in basename:
            # A wildcard cannot be matched literally; the directory it lives in
            # is the meaningful part ("the per-sample dumps are written here").
            parent = token.rsplit('/', 1)[0] if '/' in token else token
            candidates = [parent, token]
        elif '.' not in basename:
            candidates = [token]                 # bare directory
        else:
            candidates = [basename, token]
        if not any(candidate in corpus for candidate in candidates):
            unowned.append(token)
    return unowned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-sample', type=int, default=500,
                        help='camera files to check; 0 with --full-data checks all')
    parser.add_argument('--full-data', action='store_true')
    args = parser.parse_args(argv)

    failures = []

    ok, tail = check_guards()
    print(f"[{'ok' if ok else 'FAIL'}] queue guards: {tail[0] if tail else ''}")
    if not ok:
        failures.append('guards')

    missing = check_checkpoints()
    expected = [(job, path) for job, path in missing
                if not path.startswith('out/a4_fixed_gaussian')]
    print(f"[{'ok' if not expected else 'FAIL'}] checkpoints: "
          f"{len(expected)} unexpected missing")
    for job, path in expected:
        print(f'      {job}: {path}')
    if expected:
        failures.append('checkpoints')

    bad = check_configs()
    print(f"[{'ok' if not bad else 'FAIL'}] configs: {len(QUEUED_CONFIGS)} checked, "
          f'{len(bad)} bad')
    for name, why in bad:
        print(f'      {name}: {why}')
    if bad:
        failures.append('configs')

    edge, error = check_tile_build()
    if error:
        print(f'[warn] block G2V: {error}')
    else:
        good = edge == 4
        print(f"[{'ok' if good else 'FAIL'}] block G2V tile edge = {edge} "
              f'(default 4 expected; the tile sweep restores it in a finally block)')
        if not good:
            failures.append('tile-build')

    sample = 0 if args.full_data else args.data_sample
    result, error = check_data(sample)
    if error:
        print(f'[FAIL] data: {error}')
        failures.append('data')
    else:
        occ_total, occ_missing, cam_total, cam_missing = result
        good = occ_missing == 0 and cam_missing == 0
        scope = 'all' if args.full_data else f'{cam_total} sampled'
        print(f"[{'ok' if good else 'FAIL'}] data: occ {occ_total - occ_missing}/"
              f"{occ_total}, camera {cam_total - cam_missing}/{cam_total} ({scope})")
        if not good:
            failures.append('data')

    triggers = report_contract_triggers()
    unowned = check_report_contract(triggers)
    print(f"[{'ok' if not unowned else 'FAIL'}] report contract: "
          f'{len(triggers)} triggers promised by the README, '
          f'{len(unowned)} with no writer')
    for token in unowned:
        print(f'      nobody writes {token}')
    if unowned:
        failures.append('report-contract')

    cited = cited_artifact_paths()
    missing, unpromised = check_cited_artifacts(cited, triggers)
    print(f"[{'ok' if not unpromised else 'FAIL'}] cited artifacts: "
          f'{len(cited)} cited, {len(missing)} not yet produced '
          f'(all must be contract triggers), {len(unpromised)} unexplained')
    for token in unpromised:
        print(f'      cited but absent and not promised: {token}')
    if unpromised:
        failures.append('cited-artifacts')

    refs, dangling = check_section_references()
    print(f"[{'ok' if not dangling else 'FAIL'}] section references: "
          f'{refs} checked, {len(dangling)} dangling')
    for source, target, section, why in dangling:
        print(f'      {source} -> {target} §{section}: {why}')
    if dangling:
        failures.append('section-references')

    uncovered = check_artifact_coverage()
    total_artifacts = len(set(analysis_artifacts()) | set(queue_stage_artifacts()))
    print(f"[{'ok' if not uncovered else 'FAIL'}] artifact coverage: "
          f'{total_artifacts} artifacts written by scripts, '
          f'{len(uncovered)} with no promised report')
    for token, stages in sorted(uncovered.items()):
        print(f'      {token} written by {stages} is in no report row')
    if uncovered:
        failures.append('queue-artifact-coverage')

    script_refs, script_missing = check_script_references()
    print(f"[{'ok' if not script_missing else 'FAIL'}] script references: "
          f'{script_refs} cited, {len(script_missing)} missing')
    for source, token in script_missing:
        print(f'      {source} cites {token}, which does not exist')
    if script_missing:
        failures.append('script-references')

    print()
    if failures:
        print('PREFLIGHT FAILED: ' + ', '.join(failures))
        return 1
    print('PREFLIGHT OK: the queue can be left unattended.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
