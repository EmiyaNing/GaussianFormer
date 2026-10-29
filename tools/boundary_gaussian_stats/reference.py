"""Small, independent NumPy reference for tests/audits (never used for full runs)."""
import numpy as np


def reference(means,scales,inverse,alpha,semantics,xyz,labels,camera,winner,prediction,
              shape,step,multiplier,pc_min):
    ng,nv=len(means),len(labels)
    bits=np.zeros((4,ng),np.int32)
    flags=np.zeros(ng,np.int32)
    unions=np.zeros(nv,np.int32)
    owner_id=np.full((2,nv),-1,np.int64)
    best=np.zeros((2,nv),np.float32)
    edges=[]
    for g in range(ng):
        center=((means[g]-np.asarray(pc_min))/step).astype(int)
        radius=int(np.ceil(np.max(scales[g])*np.float32(multiplier)/np.float32(step)))
        lo=np.maximum(0,center-radius); hi=np.minimum(shape,center+radius+1)
        for x in range(lo[0],hi[0]):
            for y in range(lo[1],hi[1]):
                for z in range(lo[2],hi[2]):
                    v=(x*shape[1]+y)*shape[2]+z
                    if not 0<=labels[v]<18: continue
                    dx,dy,dz=means[g]-xyz[v]; cov=inverse[g]
                    power=-.5*(cov[0]*dx*dx+cov[1]*dy*dy+cov[2]*dz*dz)-(cov[3]*dx*dy+cov[4]*dy*dz+cov[5]*dx*dz)
                    weight=np.float32(alpha[g]*np.exp(np.float32(power)))
                    if not weight>0: continue
                    core=power>=-4.5
                    scopes=[0]+([1] if camera[v] else [])
                    if core: scopes += [2]+([3] if camera[v] else [])
                    for s in scopes: bits[s,g] |= 1<<int(labels[v])
                    edges.append((g,v,weight,core))
        for s in range(4):
            value=int(bits[s,g]); fg=value&((1<<17)-1)
            flag=(value.bit_count()>=2) | ((fg.bit_count()>=2)<<1) | ((bool(fg) and bool(value&(1<<17)))<<2)
            flags[g] |= flag<<(s*3)
    for g,v,weight,core in edges:
        unions[v] |= flags[g] & (4095 if core else 63)
        contribution=np.float32(weight*semantics[g,winner[v]])
        for variant in ([0,1] if core else [0]):
            if contribution>best[variant,v]:
                best[variant,v]=contribution; owner_id[variant,v]=g
    counts=np.zeros((4,9,5),np.int64)
    valid=(labels>=0)&(labels<18)
    outcomes=np.where(labels==17,np.where(prediction==17,4,3),
                      np.where(prediction==17,2,np.where(prediction==labels,0,1)))
    for s in range(4):
        mask=valid & (camera if s%2 else True)
        owner=owner_id[s//2]; owned=owner>=0
        oflags=np.zeros(nv,np.int32); oflags[owned]=(flags[owner[owned]]>>(s*3))&7
        covered=(unions>>(s*3))&7
        cohorts=[mask]+[mask&((covered&(1<<t))!=0) for t in range(3)]
        cohorts += [mask&((oflags&(1<<t))!=0) for t in range(3)]
        cohorts += [mask&owned&((oflags&1)==0),mask&~owned]
        for c,selected in enumerate(cohorts):
            counts[s,c]=np.bincount(outcomes[selected],minlength=5)
    return bits,flags,unions,owner_id,counts
