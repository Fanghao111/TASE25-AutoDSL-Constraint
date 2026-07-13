"""Diagnose duration distributions and any anomalies in ta75/ta78 vs ta71/ta79."""
import json
from collections import Counter

def stats(m, tag):
    durs = [t[1] for job in m for t in job]
    total = sum(durs)
    zeros = sum(1 for d in durs if d == 0)
    negs = sum(1 for d in durs if d < 0)
    huge = sum(1 for d in durs if d > 200)
    duplicated_tasks = 0
    for job in m:
        seen = set()
        for i, (mach, d, pre) in enumerate(job):
            key = (mach, d, tuple(pre))
            if key in seen:
                duplicated_tasks += 1
            seen.add(key)
    # LB1 = max processing per machine
    from collections import defaultdict
    per_m = defaultdict(int)
    for job in m:
        for (mach, d, pre) in job:
            per_m[mach] += d
    LB1 = max(per_m.values()) if per_m else 0
    LB2 = max(sum(t[1] for t in job) for job in m)
    n_machines = len(per_m)
    print(f"[{tag}] n_tasks={sum(len(j) for j in m)}  total_dur={total}  zero_dur={zeros}  neg_dur={negs}  huge={huge}  dup_tasks_in_job={duplicated_tasks}")
    print(f"       n_machines={n_machines}  LB1(machine-load)={LB1}  LB2(longest-job)={LB2}  theoretical LB=max(LB1,LB2)={max(LB1,LB2)}")

for i in [71, 75, 78, 79]:
    m = json.load(open(f"outputs/FB-gt-ceiling/from_gt_rs/instance ta{i}/s5_or_matrix.json"))
    stats(m, f"from_gt_rs ta{i}")
    print()
