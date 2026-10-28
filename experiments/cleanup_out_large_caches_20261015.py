"""Remove only the requested tensor/array caches in three explicit out roots."""
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

from experiments.cleanup_out_20261015 import classify, disk_usage, files, metadata, sha, signature, write


REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'out'
AUDIT=REPO/'analysis/out_large_cache_cleanup_20261015'
TARGETS=('future_alignment_hypotheses/H4_joint_cache',
         'sparseworld_region_comparison/20261010_full_nuscenes',
         'future_distillation_response_based/2048x512')


def selected_root(path,out):
    relative=path.relative_to(out)
    for target in TARGETS:
        if relative.is_relative_to(Path(target)):
            return target
    return None


def make_plan(out):
    for target in TARGETS:
        path=out/target
        if not path.is_dir() or path.is_symlink() or not path.resolve().is_relative_to(out.resolve()):
            raise ValueError(f'expected real cache directory within out: {path}')
    candidates=[]
    logs=[]
    latest=[]
    for path in files(out):
        if path.is_symlink() and path.is_dir():
            continue
        target=selected_root(path,out)
        suffix=path.suffix.lower()
        if path.name.lower()=='latest.pth':
            latest.append(dict(path=str(path.relative_to(out)),metadata=metadata(path),
                sha256=sha(path) if target and path.is_file() else None))
        elif target and suffix in ('.pth','.npz'):
            if suffix=='.pth':
                kind,reason=classify(path,out)
                if kind!='tensor_cache':
                    raise ValueError(f'non-cache checkpoint encountered, preserve it: {path}, {kind}, {reason}')
            candidates.append(dict(path=str(path.relative_to(out)),root=target,kind=suffix,metadata=metadata(path)))
        name=path.name.lower()
        if ('.log' in name or path.name.startswith('events.out.tfevents') or
                suffix in ('.jsonl','.txt','.out','.err')) and path.is_file() and not path.is_symlink():
            logs.append(dict(path=str(path.relative_to(out)),sha256=sha(path)))
    excluded={r['path'] for r in candidates}
    if excluded&{r['path'] for r in logs}:
        raise ValueError('cache deletion must not overlap protected logs')
    return dict(status='planned',out=str(out.resolve()),targets=list(TARGETS),
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        deletions=candidates,counts=dict(Counter(r['kind'] for r in candidates)),
        by_directory={target:dict(files=sum(r['root']==target for r in candidates),
            file_allocated_bytes=sum((out/r['path']).lstat().st_blocks*512 for r in candidates if r['root']==target)) for target in TARGETS},
        logs=logs,latest=latest,protected_signature=signature(out,excluded),
        out_allocated_bytes_before=disk_usage(out),filesystem_free_before=shutil.disk_usage(out).free)


def apply_plan(out,audit,plan):
    if plan['out']!=str(out.resolve()) or plan['targets']!=list(TARGETS):
        raise ValueError('cleanup scope differs from the approved three directories')
    excluded={r['path'] for r in plan['deletions']}
    for row in plan['deletions']:
        path=out/row['path']
        target=selected_root(path,out)
        if (target!=row['root'] or not path.parent.resolve().is_relative_to((out/target).resolve()) or
                path.name.lower()=='latest.pth' or path.suffix.lower() not in ('.pth','.npz') or
                metadata(path)!=row['metadata']):
            raise ValueError(f'deletion scope or target changed: {path}')
    if signature(out,excluded)!=plan['protected_signature']:
        raise ValueError('a protected file changed since inventory')
    last=time.monotonic()
    with (audit/'deleted_files.jsonl').open('x',buffering=1) as journal:
        for index,row in enumerate(plan['deletions'],1):
            path=out/row['path']
            if metadata(path)!=row['metadata']:
                raise ValueError(f'cache file changed during cleanup: {path}')
            path.unlink()
            journal.write(json.dumps(row,ensure_ascii=False)+'\n')
            if index%10000==0 or time.monotonic()-last>=20:
                print(f'CACHE_CLEANUP deleted={index}/{len(plan["deletions"])}',flush=True)
                last=time.monotonic()
    if signature(out,excluded)!=plan['protected_signature']:
        raise ValueError('a non-target file changed during cleanup')
    for row in plan['logs']:
        if sha(out/row['path'])!=row['sha256']:
            raise ValueError(f'log content changed: {row["path"]}')
    for row in plan['latest']:
        path=out/row['path']
        if metadata(path)!=row['metadata'] or (row['sha256'] and sha(path)!=row['sha256']):
            raise ValueError(f'protected latest checkpoint changed: {path}')
    leftovers=[]
    for target in TARGETS:
        leftovers.extend(str(path.relative_to(out)) for path in files(out/target)
            if path.suffix.lower() in ('.pth','.npz') and path.name.lower()!='latest.pth')
    if leftovers:
        raise ValueError(f'requested caches remain: {leftovers[:5]}')
    after=disk_usage(out)
    result=dict(status='completed',finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        targets=list(TARGETS),deleted_pth_caches=plan['counts'].get('.pth',0),
        deleted_npz_caches=plan['counts'].get('.npz',0),deleted_files=len(plan['deletions']),
        out_allocated_bytes_before=plan['out_allocated_bytes_before'],out_allocated_bytes_after=after,
        released_bytes=plan['out_allocated_bytes_before']-after,
        filesystem_free_before=plan['filesystem_free_before'],filesystem_free_after=shutil.disk_usage(out).free,
        logs_content_verified=len(plan['logs']),latest_checkpoints_preserved=len(plan['latest']),
        non_target_files_verified=plan['protected_signature']['files'],remaining_requested_caches=0,
        by_directory=plan['by_directory'])
    write(audit/'summary.json',result)
    lines=['# 三个大目录缓存清理','',
        f"释放{result['released_bytes']/2**30:.3f} GiB；out占用从{plan['out_allocated_bytes_before']/2**30:.3f}降至{after/2**30:.3f} GiB。",'',
        f"删除.pth特征/目标缓存{result['deleted_pth_caches']:,}个、.npz数组缓存{result['deleted_npz_caches']:,}个。三个目录中对应缓存剩余0个。",'',
        '| 目录（相对out） | 删除文件数 | 删除文件分配空间 GiB |','|---|---:|---:|']
    for target,row in plan['by_directory'].items():
        lines.append(f"| {target} | {row['files']:,} | {row['file_allocated_bytes']/2**30:.3f} |")
    lines+=['',f"核验全部{len(plan['latest'])}个latest.pth保留，{len(plan['logs'])}个日志/事件/文本文件内容不变，全部非目标文件元数据不变。目标目录内latest.pth还验证了内容SHA256。",'',
        '日志、JSON、Parquet、代码快照以及其他目录中的缓存均保留。目录结构保留；被删除的缓存需要重新运行相应导出流程才能再使用。','',
        '- [完整清理清单及保护校验](plan.json)','- [逐文件删除记录](deleted_files.jsonl)','- [空间与核验汇总](summary.json)','']
    (audit/'report.md').write_text('\n'.join(lines))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    AUDIT.mkdir(parents=True,exist_ok=True)
    with (AUDIT/'cleanup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.apply:
            if (AUDIT/'deleted_files.jsonl').exists() or (AUDIT/'summary.json').exists():
                raise ValueError('cleanup already applied or journal exists')
            result=apply_plan(OUT,AUDIT,json.loads((AUDIT/'plan.json').read_text()))
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        else:
            if (AUDIT/'plan.json').exists():
                raise ValueError('preserve the existing cache cleanup plan')
            plan=make_plan(OUT)
            write(AUDIT/'plan.json',plan)
            print(json.dumps({k:plan[k] for k in ['counts','by_directory','out_allocated_bytes_before']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
