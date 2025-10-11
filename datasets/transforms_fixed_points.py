
# datasets/transforms_fixed_points.py
import numpy as np

def _rdp(points, epsilon):
    """
    Ramer–Douglas–Peucker for 2D points.
    points: [N,2] ndarray
    """
    if len(points) < 3 or epsilon <= 0:
        return points
    # find index of point with max distance to line p0-pn
    p0, pn = points[0], points[-1]
    v = pn - p0
    v_norm = np.linalg.norm(v) + 1e-12
    max_d = -1.0
    idx = -1
    for i in range(1, len(points)-1):
        w = points[i] - p0
        # distance from point to line
        d = np.linalg.norm(np.cross(np.append(v,0.0), np.append(w,0.0))) / v_norm
        if d > max_d:
            max_d = d; idx = i
    if max_d > epsilon:
        left  = _rdp(points[:idx+1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return np.concatenate([left[:-1], right], axis=0)
    else:
        return np.stack([p0, pn], axis=0)

def _arc_len(P):
    if len(P) <= 1:
        return np.zeros(len(P), dtype=np.float32)
    d = np.sqrt(((P[1:] - P[:-1])**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    s /= max(s[-1], 1e-8)
    return s.astype(np.float32)

def _resample_uniform(P, K):
    if len(P) == 0:
        return np.zeros((K,2), np.float32)
    if len(P) == 1:
        return np.repeat(P.astype(np.float32), K, axis=0)
    s = _arc_len(P)                 # [N] in [0,1]
    u = np.linspace(0.0, 1.0, K)    # [K]
    X = np.interp(u, s, P[:,0])
    Y = np.interp(u, s, P[:,1])
    return np.stack([X,Y], axis=1).astype(np.float32)

def detect_cols(coords):
    """
    coords: [T,K]
    heuristic for x,y + binary pen channels
    """
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

def split_strokes(coords, col):
    up_col = col.get('up')
    if up_col is None:
        # single stroke
        return [(0, coords.shape[0]-1)]
    up = coords[:, up_col]
    idx = np.where(up==1)[0].tolist()
    spans, s = [], 0
    for e in idx:
        spans.append((s, e))
        s = e+1
    if s < coords.shape[0]:
        spans.append((s, coords.shape[0]-1))
    return spans

def resample_coords_to_K(coords, tag_char, K, rdp_epsilon=0.0, min_pts_per_stroke=3):
    """
    coords: [T,5] (abs xy + one-hot(3)) or variant with abs xy first two cols
    returns: [K,5] (delta xy + one-hot(3)), with last frame EOC=1
    NOTE: THIS FUNCTION expects absolute xy in 'coords'.
    """
    col = detect_cols(coords)
    xy_abs = to_abs(coords, col)
    spans = split_strokes(coords, col)
    # stroke lengths
    lens = []
    strokes = []
    for (s,e) in spans:
        P = xy_abs[s:e+1]
        if rdp_epsilon > 0 and len(P) > 3:
            P = _rdp(P, rdp_epsilon)
        strokes.append(P)
        # arc length
        if len(P) >= 2:
            L = np.sqrt(((P[1:]-P[:-1])**2).sum(axis=1)).sum()
        else:
            L = 0.0
        lens.append(L)
    Lsum = sum(lens) + 1e-8
    if len(strokes)==0:
        strokes=[xy_abs]; lens=[1.0]; Lsum=1.0
    # allocate K per stroke
    Ks = [max(min_pts_per_stroke, int(round(K * (L/Lsum)))) for L in lens]
    # fix sum
    diff = K - sum(Ks)
    i = 0
    while diff != 0 and len(Ks)>0:
        step = 1 if diff>0 else -1
        if Ks[i] + step >= min_pts_per_stroke:
            Ks[i] += step; diff -= step
        i = (i+1) % len(Ks)

    # resample and concat
    canon = []
    for P,Ki in zip(strokes, Ks):
        canon.append(_resample_uniform(P, Ki))
    canon = np.concatenate(canon, axis=0).astype(np.float32) if len(canon)>0 else _resample_uniform(xy_abs, K)

    # pen states: move/up only; EOC at last for output
    pen = np.zeros((K,3), np.float32)
    # set Up at stroke ends
    off = 0
    for Ki in Ks:
        end = min(off + Ki - 1, K-1)
        pen[end,1] = 1.0
        off += Ki
    # default Move elsewhere
    pen[:,0] = 1.0 - pen[:,1]
    # EOC on last frame only
    pen[-1] = np.array([0,0,1], np.float32)

    # convert to deltas
    delta = np.zeros_like(canon, dtype=np.float32)
    if K>1:
        delta[1:] = canon[1:] - canon[:-1]
        delta[0] = 0.0
    out = np.concatenate([delta, pen], axis=1).astype(np.float32)  # [K,5]
    return out
