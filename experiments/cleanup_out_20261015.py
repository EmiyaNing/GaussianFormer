"""User-authorized out cleanup with model detection, latest repair and an audit."""
import argparse
from collections import defaultdict
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time
from zoneinfo import ZoneInfo


REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'out'
AUDIT=REPO/'analysis/out_cleanup_20261015'
DISTILLATION_ROOTS=('sparseworld_fast_motion_distillation','future_distillation_causal','future_distillation_response_based')
CACHE_COMPONENTS=('cache','features','teacher_grids','full_gt_targets','alignment_targets','vis_ep','stream_vis','vectors')


def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    os.replace(tmp,path)


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def metadata(path):
    s=path.lstat()
    return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino,
                mode=stat.S_IFMT(s.st_mode),link=os.readlink(path) if path.is_symlink() else None)


def files(root):
    result=[]
    for directory,dirs,names in os.walk(root,followlinks=False):
        # Directory aliases remain unchanged and are never traversed.
        result.extend(Path(directory)/name for name in names)
        result.extend(Path(directory)/name for name in dirs if (Path(directory)/name).is_symlink())
    return sorted(result)


def weight_tree(value,depth=0):
    if not isinstance(value,dict) or depth>4:
        return False
    import torch
    for key,item in value.items():
        if isinstance(key,str) and torch.is_tensor(item) and re.search(r'(?:^|[._])(weight|bias|running_mean|running_var|num_batches_tracked)$',key):
            return True
    return any(weight_tree(item,depth+1) for item in value.values() if isinstance(item,dict))


def classify(path,root):
    name=path.name.lower()
    suffix=path.suffix.lower()
    if suffix not in ('.pth','.pt','.ckpt','.safetensors','.bin') and not name.endswith(('.pth.tmp','.pt.tmp','.ckpt.tmp')):
        return 'other',None
    relative=path.relative_to(root)
    cache=any(any(marker in part.lower() for marker in CACHE_COMPONENTS) for part in relative.parts[:-1])
    if name!='latest.pth' and (cache or re.fullmatch(r'\d+\.pth',name)):
        return 'tensor_cache',None
    if path.is_symlink() and not path.exists():
        return ('weight_link',None) if name=='latest.pth' else ('unknown','broken_nonlatest_link')
    if path.is_dir():
        return 'other',None
    if (name=='latest.pth' or name.endswith('_latest.pth') or suffix=='.safetensors' or
            re.match(r'^(epoch[_-]?\d+|iter(?:ation)?[_-]?\d*|best(?:[_.-]|$))',name)):
        return 'weight',None
    try:
        import torch
        try:
            value=torch.load(path,map_location='meta',weights_only=False,mmap=True)
        except RuntimeError:
            value=torch.load(path,map_location='meta',weights_only=False)
        return ('weight',None) if weight_tree(value) else ('tensor_cache',None)
    except Exception as error:
        return 'unknown',f'{type(error).__name__}: {error}'


def signature(root,excluded):
    digest=hashlib.sha256()
    count=0
    for path in files(root):
        relative=str(path.relative_to(root))
        if relative in excluded:
            continue
        digest.update(json.dumps([relative,metadata(path)],sort_keys=True).encode())
        count+=1
    return dict(files=count,sha256=digest.hexdigest())


def disk_usage(root):
    return int(subprocess.check_output(['du','-B1','-s',str(root)],text=True).split()[0])


def make_plan(root):
    listed=files(root)
    groups=defaultdict(list)
    nonmodels=[]
    unknown=[]
    npz=[]
    logs=[]
    last=time.monotonic()
    for index,path in enumerate(listed):
        rel=path.relative_to(root)
        if path.is_symlink() and path.is_dir():
            continue
        if path.suffix.lower()=='.npz' and rel.parts[0] in DISTILLATION_ROOTS:
            npz.append(path)
        kind,error=classify(path,root)
        if kind in ('weight','weight_link'):
            groups[path.parent].append(path)
        elif kind=='tensor_cache':
            nonmodels.append(str(rel))
        elif kind=='unknown':
            unknown.append(dict(path=str(rel),reason=error))
        if '.log' in path.name.lower() or path.name.startswith('events.out.tfevents') or path.suffix.lower() in ('.jsonl','.txt','.out','.err'):
            if path.is_file() and not path.is_symlink():
                logs.append(dict(path=str(rel),sha256=sha(path),metadata=metadata(path)))
        if time.monotonic()-last>20:
            print(f'CLEANUP_INVENTORY files={index+1}/{len(listed)} model_directories={len(groups)}',flush=True)
            last=time.monotonic()
    retention=[]
    deletions=[]
    for directory,candidates in sorted(groups.items()):
        latest=directory/'latest.pth'
        if latest.is_file() and not latest.is_symlink():
            keep=latest
            action='keep_regular_latest'
        else:
            available=[p for p in candidates if p.exists() and p.is_file() and p.name!='latest.pth']
            if latest.is_symlink() and latest.exists():
                keep=latest.resolve()
            elif latest.is_symlink() and (directory/Path(os.readlink(latest)).name) in available:
                keep=directory/Path(os.readlink(latest)).name
            else:
                keep=max(available,key=lambda p:(p.stat().st_mtime_ns,p.name)) if available else None
            if keep is None:
                unknown.append(dict(path=str(latest.relative_to(root)),reason='dangling latest without any surviving model checkpoint'))
                continue
            action='repair_or_materialize_latest' if latest.is_symlink() else 'create_latest_from_newest_model'
        retention.append(dict(directory=str(directory.relative_to(root)),latest=str(latest.relative_to(root)),
            source=str(keep.absolute()),source_metadata=metadata(keep),source_resolved=str(keep.resolve()),
            latest_metadata=metadata(latest) if latest.exists() or latest.is_symlink() else None,action=action))
        for path in candidates:
            if path.name!='latest.pth':
                deletions.append(dict(path=str(path.relative_to(root)),reason='non_latest_model_checkpoint',metadata=metadata(path)))
    deletions.extend(dict(path=str(p.relative_to(root)),reason='SparseWorld_distillation_npz',metadata=metadata(p)) for p in npz)
    changed_latest={r['latest'] for r in retention if r['action']!='keep_regular_latest'}
    excluded={r['path'] for r in deletions}|changed_latest
    return dict(status='planned',root=str(root.resolve()),created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        distillation_roots=list(DISTILLATION_ROOTS),model_directories=len(retention),retentions=retention,deletions=deletions,
        model_checkpoint_files_to_delete=sum(r['reason']=='non_latest_model_checkpoint' for r in deletions),
        distillation_npz_to_delete=len(npz),logical_bytes_to_unlink=sum(r['metadata']['size'] for r in deletions),
        tensor_cache_pth_preserved=len(nonmodels),unknown_preserved=unknown,logs=logs,
        protected_non_target_signature=signature(root,excluded),signature_exclusions=sorted(excluded),
        physical_out_bytes_before=disk_usage(root),filesystem_free_before=shutil.disk_usage(root).free)


def apply_plan(root,audit,plan):
    if plan['root']!=str(root.resolve()):
        raise ValueError('cleanup plan root differs')
    for entry in plan['deletions']:
        path=root/entry['path']
        if not path.parent.resolve().is_relative_to(root.resolve()) or metadata(path)!=entry['metadata']:
            raise ValueError(f'deletion target escaped root or changed: {path}')
    for row in plan['retentions']:
        source=Path(row['source'])
        if metadata(source)!=row['source_metadata']:
            raise ValueError(f'retention source changed: {source}')
    if signature(root,set(plan['signature_exclusions']))!=plan['protected_non_target_signature']:
        raise ValueError('non-target files changed since inventory')
    delete_paths={str((root/r['path']).absolute()) for r in plan['deletions']}
    materialized=[]
    for row in plan['retentions']:
        if row['action']=='keep_regular_latest':
            continue
        source=Path(row['source_resolved'])
        latest=root/row['latest']
        temporary=latest.with_name('latest.pth.cleanup_tmp')
        if temporary.exists() or temporary.is_symlink():
            raise ValueError('unexpected cleanup temporary checkpoint')
        if str(source) in delete_paths and source.stat().st_dev==latest.parent.stat().st_dev:
            os.link(source,temporary)
            method='hardlink_before_unlinking_original'
        else:
            shutil.copy2(source,temporary)
            if sha(source)!=sha(temporary):
                raise ValueError('latest checkpoint materialization changed bytes')
            method='verified_copy'
        os.replace(temporary,latest)
        if not latest.is_file() or latest.is_symlink():
            raise ValueError('latest must be a real readable file after materialization')
        materialized.append(dict(latest=row['latest'],source=row['source'],method=method,size=latest.stat().st_size))
    write(audit/'latest_materialization.json',dict(status='passed',files=materialized))
    count=0
    with (audit/'deleted_files.jsonl').open('x',buffering=1) as journal:
        for entry in plan['deletions']:
            path=root/entry['path']
            if metadata(path)!=entry['metadata']:
                raise ValueError(f'deletion target changed before unlink: {path}')
            path.unlink()
            journal.write(json.dumps(entry,ensure_ascii=False)+'\n')
            count+=1
            if count%500==0:
                print(f'CLEANUP_DELETED files={count}/{len(plan["deletions"])}',flush=True)
    protected=signature(root,set(plan['signature_exclusions']))
    if protected!=plan['protected_non_target_signature']:
        raise ValueError('protected files changed during cleanup')
    for log in plan['logs']:
        if sha(root/log['path'])!=log['sha256']:
            raise ValueError(f'log content changed: {log["path"]}')
    for row in plan['retentions']:
        latest=root/row['latest']
        if not latest.is_file() or latest.is_symlink():
            raise ValueError(f'latest checkpoint missing: {latest}')
    leftovers=[str(path.relative_to(root)) for path in files(root)
               if path.suffix.lower()=='.npz' and path.relative_to(root).parts[0] in DISTILLATION_ROOTS]
    if leftovers or any((root/r['path']).exists() or (root/r['path']).is_symlink() for r in plan['deletions']):
        raise ValueError('cleanup left targeted files behind')
    after=disk_usage(root)
    result=dict(status='completed',finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        deleted_files=count,deleted_model_checkpoints=plan['model_checkpoint_files_to_delete'],
        deleted_distillation_npz=plan['distillation_npz_to_delete'],retained_latest_checkpoints=len(plan['retentions']),
        repaired_or_created_latest=len(materialized),logs_verified_unchanged=len(plan['logs']),
        all_non_target_files_verified_unchanged=protected['files'],tensor_cache_pth_preserved=plan['tensor_cache_pth_preserved'],
        unknown_preserved=plan['unknown_preserved'],distillation_npz_remaining=0,
        physical_out_bytes_before=plan['physical_out_bytes_before'],physical_out_bytes_after=after,
        physical_out_bytes_released=plan['physical_out_bytes_before']-after,
        filesystem_free_before=plan['filesystem_free_before'],filesystem_free_after=shutil.disk_usage(root).free)
    write(audit/'summary.json',result)
    lines=['# out目录清理结果','',f"释放{result['physical_out_bytes_released']/2**30:.3f} GiB；out占用{result['physical_out_bytes_before']/2**30:.3f}→{after/2**30:.3f} GiB。",'',
        f"删除历史模型checkpoint {result['deleted_model_checkpoints']}个，保留latest.pth {result['retained_latest_checkpoints']}个；修复或创建latest.pth {len(materialized)}个。",'',
        f"删除SparseWorld蒸馏相关.npz {result['deleted_distillation_npz']}个，三个匹配目录下剩余0个。",'',
        '蒸馏目录：'+ '、'.join(DISTILLATION_ROOTS)+'。','',
        f"核验{len(plan['logs'])}个日志/事件/文本文件内容不变，{protected['files']}个非清理目标的文件和目录别名元数据不变。特征缓存等非模型.pth保留{plan['tensor_cache_pth_preserved']}个。",'',
        '已有latest为软链接时先物化真实文件，失效软链接优先使用同目录的原目标文件；无latest时保留修改时间最新的模型checkpoint并命名latest.pth。原历史epoch/best权重按用户要求删除，历史报告中的相关路径仅作为原实验溯源。','',
        '- [完整计划及日志校验值](plan.json)','- [逐文件删除记录](deleted_files.jsonl)','- [latest修复来源](latest_materialization.json)','- [清理汇总](summary.json)','']
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
            plan=json.loads((AUDIT/'plan.json').read_text())
            if (AUDIT/'summary.json').exists() or (AUDIT/'deleted_files.jsonl').exists():
                raise ValueError('cleanup already applied or journal already exists')
            print(json.dumps(apply_plan(ROOT,AUDIT,plan),ensure_ascii=False,indent=2),flush=True)
        else:
            if (AUDIT/'plan.json').exists():
                raise ValueError('preserve the existing cleanup plan')
            plan=make_plan(ROOT)
            write(AUDIT/'plan.json',plan)
            print(json.dumps({key:plan[key] for key in ['model_directories','model_checkpoint_files_to_delete','distillation_npz_to_delete',
                'logical_bytes_to_unlink','tensor_cache_pth_preserved','unknown_preserved','physical_out_bytes_before']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
