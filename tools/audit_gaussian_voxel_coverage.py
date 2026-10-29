"""Final integrity, provenance and report checks for the full coverage census."""
import json
import re
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.stat_boundary_gaussians import read,write,sha
from tools.stat_gaussian_voxel_coverage import RUN,OLD,COUNT_NAMES,FIELDS,FLAGS,COUNT_CAP,HISTOGRAMS


def main():
    protocol=read(RUN/'protocol.json')
    summary=read(RUN/'summary.json')
    assert summary['status']=='complete'
    assert summary['completed']==protocol['expected']
    assert read(RUN/'execution_complete.json')['status']=='complete'
    assert read(RUN/'pilot.json')['status']=='passed'
    for name,digest in {**protocol['source_sha256'],**protocol['inherited_source_sha256']}.items():
        assert sha(ROOT/name)==digest,name
    assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']
    old=read(OLD/'summary.json')
    with np.load(RUN/'global_arrays.npz') as z:
        for split in ('train','val','total'):
            n=z[split+'__voxels']
            baseline=old['statistics'][split]['conditions']['baseline']['native']
            assert n.tolist()==[baseline['misses'],baseline['gt_foreground']-baseline['misses']]
            hist=z[split+'__count_hist']
            sums=z[split+'__field_sum']
            maximum=z[split+'__count_max']
            np.testing.assert_array_equal(hist.sum(2),np.broadcast_to(n[:,None],(2,len(COUNT_NAMES))))
            for i in range(2):
                for j in range(len(COUNT_NAMES)):
                    lower=np.sum(hist[i,j]*np.arange(COUNT_CAP+2,dtype=np.int64))
                    upper=lower+hist[i,j,-1]*max(int(maximum[i,j])-COUNT_CAP-1,0)
                    assert lower<=round(sums[i,j])<=upper
            assert (sums[:,1]<=sums[:,2]).all() and (sums[:,2]<=sums[:,3]).all() and (sums[:,3]<=sums[:,0]).all()
            assert (sums[:,4]<=sums[:,0]).all() and (sums[:,6]<=sums[:,5]).all()
            assert (sums[:,FIELDS.index('opacity_mass')]<=sums[:,FIELDS.index('kernel_mass')]+1e-6).all()
            valid=z[split+'__field_n']
            share=sums[:,[FIELDS.index(k) for k in ('share_0to1','share_1to2','share_2to3','share_over3')]].sum(1)
            np.testing.assert_allclose(share,valid[:,FIELDS.index('share_0to1')],rtol=2e-6,atol=1e-5)
            flags=z[split+'__flags']
            assert (flags[:,FLAGS.index('no_native_support')]<=flags[:,FLAGS.index('no_positive_winning_class')]).all()
    checks={}
    for name in ('gaussian_voxel_coverage_full_design.md','gaussian_voxel_coverage_full_results.md'):
        p=ROOT/'analysis'/name
        text=p.read_text()
        assert text.count('$$')%2==0
        assert not re.search(r'\{\{[a-z_]+\}\}',text)
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            assert (p.parent/link.split('#')[0]).exists(),(name,link)
        lines=text.splitlines()
        for i,line in enumerate(lines):
            if line.startswith('#'):
                assert i==0 or not lines[i-1].strip()
                assert i+1==len(lines) or not lines[i+1].strip()
            if line.startswith('|') and (i==0 or not lines[i-1].startswith('|')):
                j=i
                while j<len(lines) and lines[j].startswith('|'):
                    assert line.count('|')==lines[j].count('|'),(name,j)
                    j+=1
        checks[name]=dict(lines=len(lines),links_valid=True,table_columns_consistent=True,
                         display_math_blocks=text.count('$$')//2)
    (RUN/'partial_summary.json').unlink(missing_ok=True)
    write(RUN/'final_audit.json',dict(status='passed',source_and_checkpoint_hashes_match=True,
        all_foreground_voxels_match_previous_census=True,histogram_moment_bounds_passed=True,
        core_count_ordering_passed=True,evidence_band_sums_passed=True,documents=checks))
    complete=dict(status='complete',completed_at=datetime.now().astimezone().isoformat(),
        scope=summary['completed'],sampling=protocol['sampling'],
        checkpoint_sha256=protocol['checkpoint_sha256'],frame_identities=34149,
        report='analysis/gaussian_voxel_coverage_full_results.md',
        design='analysis/gaussian_voxel_coverage_full_design.md',source_version=protocol['version'])
    write(RUN/'complete.json',complete)
    # audit.log may still be receiving this process's stdout; exclude that live log.
    files=[p for p in RUN.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json','audit.log')]
    files += [ROOT/'analysis'/name for name in checks]
    files += list((ROOT/'analysis/figures/gaussian_voxel_coverage_full').glob('*'))
    files += [Path(__file__),ROOT/'tools/report_gaussian_voxel_coverage.py',ROOT/'tools/plot_gaussian_voxel_coverage.py']
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
    write(RUN/'artifact_manifest.json',manifest)
    print(json.dumps(dict(status='passed',artifacts_hashed=len(manifest),completed=summary['completed'])))


if __name__=='__main__':
    main()
