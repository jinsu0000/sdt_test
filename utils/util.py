import numpy as np
import torch
from torchvision import transforms
import random
from PIL import ImageDraw, Image

transform_data = transforms.Compose([
    transforms.Resize((64, 64)),        # PIL 이미지 크기 변경
    transforms.ToTensor(),              # [H, W] → [1, H, W] & [0,1] 범위
    transforms.Normalize((0.5,), (0.5,))# SDT와 동일하게 [-1, 1] 범위로 변환
])

'''
description: Normalize the xy-coordinates into a standard interval.
Refer to "Drawing and Recognizing Chinese Characters with Recurrent Neural Network".
'''
def normalize_xys(xys):
    stroken_state = np.cumsum(np.concatenate((np.array([0]), xys[:, -2]))[:-1])
    px_sum = py_sum = len_sum = 0
    for ptr_idx in range(0, xys.shape[0] - 2):
        if stroken_state[ptr_idx] == stroken_state[ptr_idx + 1]:
            xy_1, xy = xys[ptr_idx][:2], xys[ptr_idx + 1][:2]
            temp_len = np.sqrt(np.sum(np.power(xy - xy_1, 2)))
            temp_px, temp_py = temp_len * (xy_1 + xy) / 2
            px_sum += temp_px
            py_sum += temp_py
            len_sum += temp_len
    if len_sum==0:
        raise Exception("Broken online characters")
    else:
        pass
    
    mux, muy = px_sum / len_sum, py_sum / len_sum
    dx_sum, dy_sum = 0, 0
    for ptr_idx in range(0, xys.shape[0] - 2):
        if stroken_state[ptr_idx] == stroken_state[ptr_idx + 1]:
            xy_1, xy = xys[ptr_idx][:2], xys[ptr_idx + 1][:2]
            temp_len = np.sqrt(np.sum(np.power(xy - xy_1, 2)))
            temp_dx = temp_len * (
                    np.power(xy_1[0] - mux, 2) + np.power(xy[0] - mux, 2) + (xy_1[0] - mux) * (xy[0] - mux)) / 3
            temp_dy = temp_len * (
                    np.power(xy_1[1] - muy, 2) + np.power(xy[1] - muy, 2) + (xy_1[1] - muy) * (xy[1] - muy)) / 3
            dx_sum += temp_dx
            dy_sum += temp_dy
    sigma = np.sqrt(dx_sum / len_sum)
    if sigma == 0:
        sigma = np.sqrt(dy_sum / len_sum)
    xys[:, 0], xys[:, 1] = (xys[:, 0] - mux) / sigma, (xys[:, 1] - muy) / sigma
    return xys

'''
description: Rendering offline character images by connecting coordinate points
'''
def coords_render(coordinates, split, width, height, thickness, board=5, show_pen_state=False):
    canvas_w = width  
    canvas_h = height  
    board_w = board  
    board_h = board
    # preprocess canvas size
    p_canvas_w = canvas_w - 2*board_w
    p_canvas_h = canvas_h - 2*board_h

    # find original character size to fit with canvas
    min_x = 635535
    min_y = 635535
    max_x = -1
    max_y = -1
    
    coordinates[:, 0] = np.cumsum(coordinates[:, 0])
    coordinates[:, 1] = np.cumsum(coordinates[:, 1])
    if split:
        ids = np.where(coordinates[:, -1] == 1)[0] 
        if len(ids) < 1:  ### if not exist [0, 0, 1]
            print("[Warning] coords_render: no [0,0,1] in the sequence!")
            ids = np.where(coordinates[:, 3] == 1)[0] + 1
            if len(ids) < 1: ### if not exist [0, 1, 0]
                ids = np.array([len(coordinates)])
                xys_split = np.split(coordinates, ids, axis=0)[:-1] # remove the blank list
            else:
                xys_split = np.split(coordinates, ids, axis=0)
        else:  ### if exist [0, 0, 1]
            remove_end = np.split(coordinates, ids+1, axis=0)[0]
            ids = np.where(remove_end[:, 3] == 1)[0] + 1 ### break in [0, 1, 0]
            xys_split = np.split(remove_end, ids, axis=0)
    else:
        pass
    for stroke in xys_split:
        for (x, y) in stroke[:, :2].reshape((-1, 2)):
            min_x = min(x, min_x)
            max_x = max(x, max_x)
            min_y = min(y, min_y)
            max_y = max(y, max_y)
    original_size = max(max_x-min_x, max_y-min_y)
    canvas = Image.new(mode='L', size=(canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(canvas)

    # show_pen_state면 색 점을 찍기 위해 RGB로 전환
    if show_pen_state:
        canvas = canvas.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        r = 1
        # class → color (현재 파이프라인엔 EOC 없음: 0/1/2만 존재)  :contentReference[oaicite:3]{index=3}
        color_map = {
            0: (0, 128, 255),  # move: blue
            1: (0, 255, 0),      # up/neutral: green
            2: (255, 0, 0),    # EOS: red
        }


    for stroke in xys_split:
        xs, ys = stroke[:, 0], stroke[:, 1]
        xys = np.stack([xs, ys], axis=-1).reshape(-1)
        xys[::2] = (xys[::2]-min_x) / original_size * p_canvas_w + board_w 
        xys[1::2] = (xys[1::2] - min_y) / original_size * p_canvas_h + board_h
        xys = np.round(xys)
        draw.line(xys.tolist(), fill=0, width=thickness)
        # --- pen state 점 덧그리기 (옵션) ---
        if show_pen_state:
            xs_n = np.round((xs - min_x) / original_size * p_canvas_w + board_w).astype(int)
            ys_n = np.round((ys - min_y) / original_size * p_canvas_h + board_h).astype(int)

        # --- pen state 점 덧그리기 (옵션; one-hot 값으로 결정) ---
        if show_pen_state:
            xs_n = np.round((xs - min_x) / original_size * p_canvas_w + board_w)
            ys_n = np.round((ys - min_y) / original_size * p_canvas_h + board_h)
            # 안전: finite 체크 + int 변환
            xs_n = xs_n.astype(np.int64, copy=False)
            ys_n = ys_n.astype(np.int64, copy=False)

            if stroke.shape[1] >= 5:
                pen_idx = stroke[:, 2:].argmax(axis=-1)  # [len(stroke)]
            else:
                pen_idx = np.zeros(len(stroke), dtype=np.int64)

            for i in range(len(stroke)):
                cx, cy = int(xs_n[i]), int(ys_n[i])
                c = int(pen_idx[i])
                color = color_map.get(c, (255, 255, 255))
                # PIL 안정성: 사각형 좌표 정렬(최소/최대)로 ValueError 예방
                x0, x1 = cx - r, cx + r
                y0, y1 = cy - r, cy + r
                if x0 > x1: x0, x1 = x1, x0
                if y0 > y1: y0, y1 = y1, y0
                draw.ellipse([x0, y0, x1, y1], fill=color)
    return canvas

# fix random seeds for reproducibility
def fix_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.manual_seed(random_seed)

### model loads specific parameters (i.e., par) from pretrained_model 
def load_specific_dict(model, pretrained_model, par):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model)
    if par in list(pretrained_dict.keys())[0]:
        count = len(par) + 1
        pretrained_dict = {k[count:]: v for k, v in pretrained_dict.items() if k[count:] in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    if len(pretrained_dict) > 0:
        model_dict.update(pretrained_dict)
    else:
        return ValueError
    return model_dict


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


'''
description: convert the np version of coordinates to the list counterpart
'''
def dxdynp_to_list(coordinates, eos_trunc: bool = False):
    ids = np.where(coordinates[:, -1] == 1)[0]  # Pen_end(EOC) 위치
    # ---- NEW: 잘라낼 시퀀스 선택 (원본 유지 + eos_trunc 추가) ----
    if len(ids) < 1:  ### if not exist [0, 0, 1]
        seq = coordinates
        if eos_trunc:
            eos_ids = np.where(seq[:, 3] == 1)[0]  # Pen_up(EOS) 위치
            if len(eos_ids) > 0:
                seq = seq[:eos_ids[0] + 1]        # 첫 EOS '포함'까지 자르기
        ids = np.where(seq[:, 3] == 1)[0] + 1
        if len(ids) < 1: ### if not exist [0, 1, 0]
            ids = np.array([len(seq)])
            xys_split = np.split(seq, ids, axis=0)[:-1] # remove the blank list
        else:
            xys_split = np.split(seq, ids, axis=0)
    else:  ### if exist [0, 0, 1]
        # 원본과 동일: EOC '직전'까지만 사용 (EOC 프레임 제외)
        remove_end = np.split(coordinates, ids, axis=0)[0]
        seq = remove_end
        ids = np.where(remove_end[:, 3] == 1)[0] + 1 ### break in [0, 1, 0]
        xys_split = np.split(remove_end, ids, axis=0)[:-1] # split from the remove_end

    # ---- 원본 그대로: 스트로크를 [x0,y0,x1,y1,...]로 변환 ----
    coord_list = []
    for stroke in xys_split:
        xs, ys = stroke[:, 0], stroke[:, 1]
        if len(xs) > 0:
            xys = np.stack([xs, ys], axis=-1).reshape(-1)
            coord_list.append(xys)
        else:
            pass

    length = len(seq)  # NEW: 실제 사용한 구간 길이
    return coord_list, length

'''
description: 
    [x, y] --> [x, y, p1, p2, p3]
    see 'A NEURAL REPRESENTATION OF SKETCH DRAWINGS' for more details
'''
def corrds2xys(coordinates):
    new_strokes = []
    for stroke in coordinates:
        for (x, y) in np.array(stroke).reshape((-1, 2)):
            p = np.array([x, y, 1, 0, 0], np.float32)
            new_strokes.append(p)
        try:   
            new_strokes[-1][2:] = [0, 1, 0]  # set the end of a stroke
        except IndexError:
            print(stroke)
            return None
    new_strokes = np.stack(new_strokes, axis=0)
    return new_strokes