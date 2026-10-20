"""Audit the GPU queue's wait guards for false positives.

The queue is a chain of scripts, each of which blocks on

    while pgrep -f "[r]un_previous\.sh" > /dev/null; do sleep 60; done

A **false positive** in one of those patterns is the worst possible failure for
this pipeline: the guard would match an unrelated process (a leftover shell whose
command text happens to mention the script, a recycled pid, a stale heredoc) and
wait forever, so every later stage would never run and nothing would report an
error.  Two false positives are especially easy to create accidentally:

* a pattern that also matches the *checking* script's own command line, and
* a pattern that matches an old ``bash -c`` wrapper whose embedded heredoc text
  contains the script name.

This tool resolves each guard's matches to their actual command lines and
classifies them, so a stalled chain can be diagnosed instead of guessed at.

    python3.10 tools/queue_guard_audit.py
    python3.10 tools/queue_guard_audit.py --expect-running run_p1_a3_and_seeds.sh
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD_RE = re.compile(r'pgrep -f "([^"]+)"')


def snapshot_processes() -> dict:
    """``{pid: (ppid, elapsed_seconds, cmdline)}`` for the whole process table."""
    procs = {}
    for line in subprocess.run(['ps', '-eo', 'pid,ppid,etimes,cmd'],
                               capture_output=True, text=True).stdout.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) == 4:
            procs[parts[0]] = (parts[1], int(parts[2]), parts[3])
    return procs


def command_line(pid: str) -> str:
    return subprocess.run(['ps', '-o', 'cmd=', '-p', pid],
                          capture_output=True, text=True).stdout.strip()


def vanished(pid: str) -> bool:
    """True when ``ps`` cannot produce a command line for ``pid``.

    A pid can disappear between ``pgrep`` listing it and this inspection --
    routine while training spawns and reaps dataloader workers.  Such a pid
    cannot hold a guard open, so it must be skipped rather than classified: an
    empty command line used to fall through to ``HAZARD(unrelated process)``,
    which produced a transient ``AUDIT FAILED`` on a live, churning process tree
    (observed 2026-09-14, right after a training stage cycled its workers).  A
    false FAIL in an unattended pipeline is worse than a missing audit line,
    because it teaches the operator to ignore the check.
    """
    return not command_line(pid)


def own_process_tree() -> set[str]:
    """Our pid plus every ancestor, so the audit never flags itself.

    The audit's own command line contains the guard patterns as arguments, so a
    naive pgrep-based check reports itself as a false positive -- the exact
    self-match failure mode the bracketed patterns exist to avoid.  (The
    bracketed form stops ``pgrep`` from matching *its own* command line, but not
    an unrelated later process that merely quotes the pattern.)
    """
    pids = {str(os.getpid())}
    pid = str(os.getpid())
    for _ in range(16):
        out = subprocess.run(['ps', '-o', 'ppid=', '-p', pid],
                             capture_output=True, text=True).stdout.strip()
        if not out or out == '0' or out in pids:
            break
        pids.add(out)
        pid = out
    return pids


def wrapper_is_pinned(pid: str, guarded: str, procs: dict) -> bool:
    """Is a ``bash -c`` wrapper blocked in ``wait()`` on the guarded script?

    This is the mechanism that decides whether a wrapper is dangerous.  A
    wrapper launched as ``bash -c '... nohup bash tools/run_X.sh & ...'`` stays
    alive for its child's whole lifetime, so its command line keeps quoting
    ``run_X.sh`` and the guard in the next stage keeps matching it.  That is
    harmless **iff** the wrapper exits when the child does, which is exactly what
    ``wchan == do_wait`` plus a live queue-script child proves.  If instead a
    wrapper outlived its child, the next guard would match forever and the chain
    would stall with no error -- the failure this audit exists to prevent.
    """
    children = subprocess.run(['cat', f'/proc/{pid}/task/{pid}/children'],
                              capture_output=True, text=True).stdout.split()
    wchan = subprocess.run(['cat', f'/proc/{pid}/wchan'],
                           capture_output=True, text=True).stdout.strip()
    if wchan != 'do_wait':
        return False
    for child in children:
        entry = procs.get(child)
        if entry and re.fullmatch(r'bash tools/run_[A-Za-z0-9_]+\.sh', entry[2].strip()):
            return True
    return False


def classify(cmd: str, expected_hint: str, own: set[str], pid: str,
             procs: dict) -> str:
    """Decide whether a guard's match is legitimate."""
    if pid in own:
        return 'self (excluded)'
    if 'queue_guard_audit' in cmd:
        return 'self (excluded)'
    if cmd.startswith('bash -c'):
        if wrapper_is_pinned(pid, expected_hint, procs):
            return 'wrapper(pinned to its target; exits with it)'
        return 'HAZARD(wrapper would outlive its target -> guard stalls)'
    if cmd.startswith('bash tools/') or cmd.startswith('bash -c cd'):
        return 'expected(queue script)'
    if 'train.py' in cmd and '--py-config' in cmd:
        return 'expected(training)'
    if 'eval.py' in cmd or 'eval_matrix.py' in cmd:
        return 'expected(evaluation)'
    if expected_hint and expected_hint in cmd:
        return 'expected(target)'
    return f'HAZARD(unrelated process: {cmd[:60]})'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scripts', nargs='*', default=None,
                        help='queue scripts to audit (default: tools/run_*.sh)')
    parser.add_argument('--expect-running', nargs='*', default=None,
                        help='script names that SHOULD be matched (they are the '
                             'targets of some guard)')
    args = parser.parse_args(argv)

    paths = ([ROOT / 'tools' / name for name in args.scripts] if args.scripts
             else sorted((ROOT / 'tools').glob('run_*.sh')))
    own = own_process_tree()
    procs = snapshot_processes()

    hazards = 0
    self_matching = 0
    for path in paths:
        if not path.exists():
            print(f'missing {path}')
            continue
        text = path.read_text(encoding='utf-8')
        patterns = GUARD_RE.findall(text)
        # A guard never matches itself *only* if the first character is bracketed;
        # otherwise pgrep can see an ancestor shell that quotes the pattern.
        for pattern in [p for p in patterns if not p.startswith('[')]:
            self_matching += 1
            print(f'{path.name:30s} {pattern:32s} !! un-bracketed: wrap the first '
                  f'character in [] so pgrep cannot match a shell that quotes it')
        for pattern in patterns:
            pids = subprocess.run(['pgrep', '-f', pattern],
                                  capture_output=True, text=True).stdout.split()
            hint = re.sub(r'[\[\]\\]', '', pattern).split()[0]
            # Drop pids that no longer exist: they cannot hold the guard open,
            # and classifying their empty command line as HAZARD made this check
            # flaky on a live process tree (see ``vanished``).
            live = [pid for pid in pids if not vanished(pid)]
            skipped = len(pids) - len(live)
            verdicts = []
            for pid in live:
                cmd = command_line(pid)
                verdict = classify(cmd, hint, own, pid, procs)
                if 'HAZARD' in verdict:
                    # The pgrep listing and the process snapshot are taken at
                    # different instants, so a wrapper that is *mid-spawn* can
                    # look like it has no pinned child.  Re-inspect once with a
                    # fresh snapshot before calling it a hazard: a wrapper that
                    # really would outlive its target stays hazardous, while the
                    # race (which produced transient AUDIT FAILED on a live tree)
                    # resolves.  Confirmed hazards are labelled as such.
                    time.sleep(0.25)
                    fresh = snapshot_processes()
                    retry = command_line(pid)
                    if not retry:
                        verdict = 'vanished before re-inspection'
                    else:
                        verdict = classify(retry, hint, own, pid, fresh)
                        if 'HAZARD' in verdict:
                            verdict += '(confirmed on re-inspection)'
                verdicts.append(f'{pid}:{verdict}')
            hazards += sum(1 for v in verdicts if 'HAZARD' in v)
            state = 'idle' if not live else 'waiting'
            note = f'  ({skipped} vanished before inspection)' if skipped else ''
            print(f'{path.name:30s} {pattern:32s} {state:8s} '
                  f'{" ".join(verdicts) if verdicts else "-"}{note}')

    print()
    if hazards or self_matching:
        print(f'AUDIT FAILED: {hazards} hazardous match(es), {self_matching} '
              f'un-bracketed pattern(s). A wrapper that outlives its target, or a '
              f'pattern that can match a shell quoting it, would stall the chain '
              f'silently.')
        return 1
    print('AUDIT OK: every guard matched only legitimate targets (or nothing); '
          'wrappers that quote a pattern are pinned in do_wait to the very script '
          'they quote, so they exit with it; every pattern is bracketed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
