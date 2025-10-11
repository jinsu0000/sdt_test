
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
    P = coords.astype(np.float32)
    # 1) 입력은 '절대좌표' + one-hot(3)라고 가정
    xy_abs = P[:, :2]

    # 2) stroke split: Up 채널(=1) 기준 (고정 포맷 전제)
    up = (P[:, 3] == 1).astype(np.int32)  # P[:, 2:5] -> [Move, Up, EOC], index 3가 Up
    spans, s = [], 0
    for e in np.where(up == 1)[0].tolist():
        spans.append((s, e))
        s = e + 1
    if s < len(xy_abs):
        spans.append((s, len(xy_abs) - 1))
    if not spans:
        spans = [(0, len(xy_abs) - 1)]

    # 3) (옵션) RDP 간략화
    def _rdp(points, eps):
        if len(points) < 3 or eps <= 0: return points
        p0, pn = points[0], points[-1]
        v = pn - p0; vn = np.linalg.norm(v) + 1e-12
        idx = -1; md = -1.0
        for i in range(1, len(points)-1):
            d = np.linalg.norm(np.cross(np.append(v,0.0), np.append(points[i]-p0,0.0))) / vn
            if d > md: md, idx = d, i
        if md > eps:
            L = _rdp(points[:idx+1], eps)
            R = _rdp(points[idx:], eps)
            return np.concatenate([L[:-1], R], axis=0)
        return np.stack([p0, pn], axis=0)

    strokes, lens = [], []
    for s,e in spans:
        seg = xy_abs[s:e+1]
        if rdp_epsilon > 0 and len(seg) > 3:
            seg = _rdp(seg, rdp_epsilon)
        strokes.append(seg)
        L = 0.0 if len(seg) < 2 else np.sqrt(((seg[1:]-seg[:-1])**2).sum(axis=1)).sum()
        lens.append(L)
    Lsum = float(sum(lens)) + 1e-8

    # 4) K 분배(길이 비례, 최소 길이 보장) + 합 맞추기
    Ks = [max(min_pts_per_stroke, int(round(K * (L/Lsum)))) for L in lens]
    diff = K - sum(Ks); i = 0
    while diff != 0 and Ks:
        step = 1 if diff > 0 else -1
        if Ks[i] + step >= min_pts_per_stroke:
            Ks[i] += step; diff -= step
        i = (i + 1) % len(Ks)

    # 5) 등간격(arc-length) 보간
    def _arc_len(P2):
        if len(P2) <= 1: return np.zeros(len(P2), np.float32)
        d = np.sqrt(((P2[1:] - P2[:-1])**2).sum(axis=1))
        s = np.concatenate([[0.0], np.cumsum(d)])
        s /= max(s[-1], 1e-8)
        return s.astype(np.float32)
    def _resample(P2, K2):
        if len(P2) == 0: return np.zeros((K2,2), np.float32)
        if len(P2) == 1: return np.repeat(P2.astype(np.float32), K2, axis=0)
        s = _arc_len(P2); u = np.linspace(0.0, 1.0, K2)
        X = np.interp(u, s, P2[:,0]); Y = np.interp(u, s, P2[:,1])
        return np.stack([X,Y], axis=1).astype(np.float32)

    canon = np.concatenate([_resample(seg, Ki) for seg,Ki in zip(strokes, Ks)], axis=0).astype(np.float32)

    # 6) pen(one-hot) 구성: stroke 끝 Up=1, 마지막 EOC=1
    pen = np.zeros((K,3), np.float32)
    off = 0
    for Ki in Ks:
        end = min(off + Ki - 1, K-1)
        pen[end,1] = 1.0    # Up
        off += Ki
    pen[:,0] = 1.0 - pen[:,1]   # Move
    pen[-1] = np.array([0,0,1], np.float32)  # EOC (마지막 강제)

    # 7) 절대좌표 + one-hot 반환 (정규화/Δ 변환은 기존 normalize_xys가 수행)
    return np.concatenate([canon, pen], axis=1).astype(np.float32)
