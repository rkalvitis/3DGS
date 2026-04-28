import json
import os
import glob
from collections import defaultdict

OUTPUT_DIR = '/output'

summary = defaultdict(lambda: defaultdict(list))

for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, '*/results.json'))):
    run_name = os.path.basename(os.path.dirname(f))
    scene, seed = run_name.rsplit('_seed', 1)
    with open(f) as fh:
        data = json.load(fh)
    key = sorted(data.keys())[-1]  # e.g. 'ours_30000'
    m = data[key]
    summary[scene]['PSNR'].append(m.get('PSNR'))
    summary[scene]['SSIM'].append(m.get('SSIM'))
    summary[scene]['LPIPS'].append(m.get('LPIPS'))

header = '{:<15} {:>8} {:>8} {:>8}  (mean over {} seeds)'.format(
    'Scene', 'PSNR', 'SSIM', 'LPIPS', 5)
print(header)
print('-' * 52)

for scene in sorted(summary):
    v = summary[scene]
    psnr  = sum(x for x in v['PSNR']  if x is not None) / len(v['PSNR'])
    ssim  = sum(x for x in v['SSIM']  if x is not None) / len(v['SSIM'])
    lpips = sum(x for x in v['LPIPS'] if x is not None) / len(v['LPIPS'])
    n = len(v['PSNR'])
    print('{:<15} {:>8.2f} {:>8.4f} {:>8.4f}  (n={})'.format(
        scene, psnr, ssim, lpips, n))
