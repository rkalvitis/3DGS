import json
import os
import sys
import glob
from collections import defaultdict

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else '/output'
ITERATIONS = [7000, 30000]

summary = {itr: defaultdict(lambda: defaultdict(list)) for itr in ITERATIONS}

for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, '**/results.json'), recursive=True)):
    run_name = os.path.basename(os.path.dirname(f))
    if '_seed' in run_name:
        scene, seed = run_name.rsplit('_seed', 1)
    else:
        scene, seed = run_name, '0'   # single-seed runs (e.g. FineView species)
    with open(f) as fh:
        data = json.load(fh)
    for itr in ITERATIONS:
        key = 'ours_{}'.format(itr)
        if key not in data:
            continue
        m = data[key]
        summary[itr][scene]['PSNR'].append(m.get('PSNR'))
        summary[itr][scene]['SSIM'].append(m.get('SSIM'))
        summary[itr][scene]['LPIPS'].append(m.get('LPIPS'))

for itr in ITERATIONS:
    print('=== ours_{} ==='.format(itr))
    print('{:<15} {:>8} {:>8} {:>8}  (mean over {} seeds)'.format(
        'Scene', 'PSNR', 'SSIM', 'LPIPS', 5))
    print('-' * 52)
    for scene in sorted(summary[itr]):
        v = summary[itr][scene]
        n = len(v['PSNR'])
        psnr  = sum(x for x in v['PSNR']  if x is not None) / n
        ssim  = sum(x for x in v['SSIM']  if x is not None) / n
        lpips = sum(x for x in v['LPIPS'] if x is not None) / n
        print('{:<15} {:>8.2f} {:>8.4f} {:>8.4f}  (n={})'.format(
            scene, psnr, ssim, lpips, n))
    print()
