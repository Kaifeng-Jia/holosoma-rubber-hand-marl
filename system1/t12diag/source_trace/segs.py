import csv, numpy as np
def segs_of(m, t):
    out=[]; s=None
    for i,v in enumerate(m):
        if v and s is None: s=i
        if not v and s is not None: out.append((round(t[s],2),round(t[i-1],2))); s=None
    if s is not None: out.append((round(t[s],2),round(t[len(m)-1],2)))
    return out
r=list(csv.DictReader(open('hand_table_per_source_frame.csv')))
col=lambda c: np.array([float(x[c]) for x in r])
t=col('t_s')
for p in (1,2):
    for kind in ('mesh','box'):
        for side in ('left','right'):
            print(f'person{p} {kind} {side} within1.5cm (s):', segs_of(col(f'p{p}_{side}_{kind}_cm')>=-1.5, t))
rr=list(csv.DictReader(open('/home/kevin/duet-system1/results_system1/T12_diag/reference_hand_table_distance_per_frame.csv')))
t50=np.arange(len(rr))/50
for rb in (0,1):
    for side in ('left','right'):
        print(f'runtime robot{rb} {side} within1.5cm (s):', segs_of(np.array([float(x[f"urdf_r{rb}_{side}_cm"]) for x in rr])>=-1.5, t50))
