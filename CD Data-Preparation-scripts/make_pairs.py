"""Driver: crop the selected boxes from the source maps (one streaming pass per source file), then level/centre/co-register
each pair.  Output layout mirrors Pairs/Pair1..8:  <source> - Cloud.ply (raw crops) + 2023.ply / 2025.ply (aligned) + pair_info.json
Usage: python make_pairs.py [crop|align|all]"""
import json, os, sys, time
import crop, align

OUT = "E:/Subiaco Change Detection point cloud data/Pairs"
MARGIN = 10.0
sel = json.load(open('selected.json'))
mode = sys.argv[1] if len(sys.argv) > 1 else 'all'


def raw_name(s, yr):
    if yr == '2023':
        return os.path.basename(s['src2023'])[:-4] + ' - Cloud.ply'
    return f"20250226_{s['time2025']}_{s['loop']}_OS-1-128_122313001294map_no_underground - Cloud.ply"


jobs = []
for s in sel:
    d = f"{OUT}/{s['pair']}"
    jobs.append({'name': f"{s['pair']}/2023", 'src': s['src2023'], 'xmin': s['xmin'], 'xmax': s['xmax'], 'ymin': s['ymin'], 'ymax': s['ymax'],
                 'out': f"{d}/{raw_name(s, '2023')}"})
    (lx0, ly0), (lx1, ly1) = s['loop_bbox']
    jobs.append({'name': f"{s['pair']}/2025", 'src': s['src2025'], 'xmin': lx0 - MARGIN, 'xmax': lx1 + MARGIN, 'ymin': ly0 - MARGIN, 'ymax': ly1 + MARGIN,
                 'out': f"{d}/{raw_name(s, '2025')}"})

if mode in ('crop', 'all'):
    todo = [j for j in jobs if not os.path.exists(j['out'])]
    print(f"{len(todo)} crops to do over {len(set(j['src'] for j in todo))} source files", flush=True)
    crop.run(todo)

if mode in ('align', 'all'):
    for s in sel:
        d = f"{OUT}/{s['pair']}"
        if os.path.exists(f"{d}/2025.ply") and os.path.exists(f"{d}/pair_info.json"):
            print(s['pair'], 'already aligned'); continue
        t0 = time.time()
        info = align.main(f"{d}/{raw_name(s, '2023')}", f"{d}/{raw_name(s, '2025')}", d, s['pair'], pre=(s['yaw_deg'], s['t']), clip=True)
        info['crop_box_2023_map_frame'] = {k: s[k] for k in ('xmin', 'xmax', 'ymin', 'ymax')}
        info['crop_box_2025_loop_frame'] = {'xmin': jobs[2 * sel.index(s) + 1]['xmin'], 'xmax': jobs[2 * sel.index(s) + 1]['xmax'],
                                            'ymin': jobs[2 * sel.index(s) + 1]['ymin'], 'ymax': jobs[2 * sel.index(s) + 1]['ymax']}
        info['box_centre_in_loop_frame'] = s['centre_loop']
        info['selection'] = {'joint_coverage_frac': s['joint'], 'relief_ncc_score': s['score'], 'relief_ncc_second_peak': s['second'],
                             'prediction_error_m': s['pred_err_m'], 'duplicate_ncc_max': s['dup_ncc']}
        info['loop2025'] = s['loop']; info['time2025'] = s['time2025']; info['map2023'] = s['map'][6:]
        info['files'] = {'raw2023': raw_name(s, '2023'), 'raw2025': raw_name(s, '2025'), 'aligned2023': '2023.ply', 'aligned2025': '2025.ply'}
        json.dump(info, open(f"{d}/pair_info.json", 'w'), indent=1)
        try: os.remove(f"{d}/alignment.json")
        except OSError: pass
        print(f"{s['pair']} aligned in {time.time()-t0:.0f}s: nn median {info['nn_median_2023to2025_m']} / {info['nn_median_2025to2023_m']} m, "
              f"coarse shift {info['coarse_xy_shift']}", flush=True)
