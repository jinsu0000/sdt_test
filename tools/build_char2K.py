
# tools/build_char2K.py
import os, lmdb, pickle, argparse, json, numpy as np
from collections import defaultdict, Counter

def detect_cols(coords):
    x_idx, y_idx = 0, 1
    cand = [j for j in range(coords.shape[1]) if j not in (x_idx,y_idx)]
    binlike = [j for j in cand if set(np.unique(coords[:,j]).tolist()) <= {0,1}]
    pm, up, eoc = None, None, None
    if len(binlike) >= 1:
        tails = [j for j in binlike if coords[-1,j]==1]
        if tails:
            eoc = tails[0]; binlike.remove(eoc)
    if len(binlike) >= 1:
        counts = [(j, int(coords[:,j].sum())) for j in binlike]
        counts.sort(key=lambda x: abs(x[1]-coords.shape[0]/10))
        up = counts[0][0]; binlike.remove(up)
    if len(binlike) >= 1:
        pm = binlike[0]
    return dict(x=0, y=1, pm=pm, up=up, eoc=eoc)

def to_abs(coords, col):
    xy = coords[:, [col['x'], col['y']]].astype(np.float32)
    if len(xy)<=1: 
        return xy
    mean_step = np.mean(np.abs(xy[1:] - xy[:-1]))
    mean_lvl  = np.mean(np.abs(xy))
    if mean_step < 0.1 * max(mean_lvl, 1e-3):
        xy = np.cumsum(xy, axis=0)   # delta → absolute
    return xy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True, help="LMDB path (train split)")
    ap.add_argument("--out", required=True, help="char2K.json path")
    ap.add_argument("--minK", type=int, default=32)
    ap.add_argument("--maxK", type=int, default=128)
    ap.add_argument("--stat", choices=["median","p75","mean"], default="median")
    args = ap.parse_args()

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False, meminit=False)
    lengths = defaultdict(list)
    with env.begin(write=False) as txn:
        n = int(txn.get(b'num_sample').decode())
        for i in range(n):
            rec = pickle.loads(txn.get(str(i).encode('utf-8')))
            coords = np.array(rec['coordinates'])
            col = detect_cols(coords)
            xy = to_abs(coords, col)
            T = int(xy.shape[0])
            ch = rec.get('tag_char','*')
            lengths[ch].append(T)

    char2K = {}
    for ch, arr in lengths.items():
        if args.stat == "median":
            val = int(np.median(arr))
        elif args.stat == "p75":
            val = int(np.percentile(arr, 75))
        else:
            val = int(np.mean(arr))
        val = max(args.minK, min(args.maxK, val))
        # round to multiple of 2 (stable for chunk H=6/stride)
        if val % 2 == 1: val += 1
        char2K[ch] = val
    if "*" not in char2K:
        # fallback default as global median
        all_vals = [v for vs in lengths.values() for v in vs]
        fallback = int(np.median(all_vals)) if all_vals else 64
        fallback = max(args.minK, min(args.maxK, fallback))
        if fallback % 2 == 1: fallback += 1
        char2K["*"] = fallback

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(char2K, f, ensure_ascii=False, indent=2)
    print(f"Saved {args.out} with {len(char2K)} entries.")

if __name__ == "__main__":
    main()
