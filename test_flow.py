# test_flow.py
import argparse, os, pickle, torch, tqdm, lmdb
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from data_loader.loader import ScriptDataset
from utils.util import writeCache, dxdynp_to_list, coords_render
from models.model import SDT_Generator
from models.sdt_flow_wrapper import SDT_FlowWrapper

def lengths_from_eos_or_nz(coords: torch.Tensor) -> torch.Tensor:
    """
    coords: [B, T, 5]  (dx, dy, pen one-hot(3))
    return: [B] (각 시퀀스의 유효 길이)
    규칙:
      1) pen one-hot에서 index==2(EOS)인 **첫 위치 + 1**을 길이로 사용
      2) EOS가 없으면, **비-제로 프레임 수**를 길이로 사용(전부 0이면 0)
    """
    device = coords.device
    B, T, _ = coords.shape

    pen_idx = coords[..., 2:].argmax(-1)    # [B, T]
    eos_mask = (pen_idx == 2)               # EOS 위치
    idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    first_eos = torch.where(eos_mask, idxs, torch.full_like(idxs, T)).min(dim=1).values  # [B]
    has_eos = (first_eos < T)

    # EOS가 없으면: 패딩이 0이라 가정하고 비-제로 프레임 수로 대체
    nonzero_mask = (coords.abs().sum(dim=-1) > 0)  # [B, T]
    nz_len = nonzero_mask.sum(dim=1)               # [B]

    lengths = torch.where(has_eos, first_eos + 1, nz_len)
    return lengths.clamp(max=T)

def _valid_seq_list(lst):
    # dxdynp_to_list가 반환하는 리스트가 비어있지 않고,
    # 각 토막 길이가 짝수( x,y 쌍 )이며 최소 한 토막은 존재하는지 체크
    if not isinstance(lst, list) or len(lst) == 0:
        return False
    for seg in lst:
        if not isinstance(seg, (list, tuple)) or len(seg) < 2 or len(seg) % 2 != 0:
            return False
    return True


def main(opt):
    # ----- cfg / device -----
    cfg_from_file(opt.cfg_file); assert_and_infer_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- data -----
    ds = ScriptDataset(cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, cfg.TEST.ISTRAIN, cfg.MODEL.NUM_IMGS)
    dl = torch.utils.data.DataLoader(ds,
                                     batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                     shuffle=True, drop_last=False,
                                     collate_fn=ds.collate_fn_,
                                     num_workers=cfg.DATA_LOADER.NUM_THREADS)
    
    # --- char2K mapping (optional) ---
    import json
    char2K = None
    if hasattr(opt, 'char2K') and opt.char2K:
        try:
            with open(opt.char2K, 'r', encoding='utf-8') as f:
                char2K = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to load char2K: {e}")
    char_dict, writer_dict = ds.char_dict, ds.writer_dict

    # ----- save dir / lmdb -----
    os.makedirs(os.path.join(opt.save_dir, 'test'), exist_ok=True)
    env = lmdb.open(os.path.join(opt.save_dir, 'test'), map_size=1099511627776)
    pickle.dump(writer_dict, open(os.path.join(opt.save_dir, 'writer_dict.pkl'), 'wb'))
    pickle.dump(char_dict,  open(os.path.join(opt.save_dir, 'character_dict.pkl'), 'wb'))

    # ----- model -----
    sdt = SDT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
                        num_head_layers=cfg.MODEL.NUM_HEAD_LAYERS,
                        wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
                        gly_dec_layers=cfg.MODEL.GLY_DEC_LAYERS).to(device)
    flow = SDT_FlowWrapper(sdt, H=opt.H, stride_default=opt.stride).to(device)

    # checkpoint 로딩 (flow 래퍼/SDT 단독 모두 허용)
    state = torch.load(opt.pretrained_model, map_location=device)
    model_state = state.get('model', state)
    try:
        flow.load_state_dict(model_state, strict=False)
    except Exception:
        cleaned = {k.replace('module.', ''): v for k, v in model_state.items()}
        flow.load_state_dict(cleaned, strict=False)
    flow.eval()

    # ----- 배치 수 -----
    if opt.sample_size == 'all':
        total_batches = len(dl)
    else:
        total_batches = int(opt.sample_size) * len(writer_dict) // cfg.TRAIN.IMS_PER_BATCH
        total_batches = min(total_batches, len(dl))

    num_written = 0
    data_iter = iter(dl)
    with torch.no_grad():
        for _ in tqdm.tqdm(range(total_batches)):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                data = next(data_iter)

            coords      = data['coords'].to(device)           # [B,T,5] (GT Δ, one-hot)
            character_id= data['character_id'].long().to(device)
            writer_id   = data['writer_id'].long().to(device)
            img_list    = data['img_list'].to(device)         # [B,N,1,H,W]
            char_img    = data['char_img'].to(device)

            B = coords.size(0)
            coords_len = lengths_from_eos_or_nz(coords)

            #T = min(opt.T, coords.size(1))                    # 원본과 동일한 길이 컨트롤

            # Per-sample T determined by char2K (fallback to opt.T)
            preds = []
            for i in range(B):
                ch = char_dict[character_id[i].item()]
                T_i = int(char2K.get(ch, char2K.get('*', opt.T))) if char2K is not None else int(opt.T)
                pred_i = flow.flow_infer(
                    img_list[i:i+1], char_img[i:i+1], T=T_i, steps=opt.steps,
                    stride=opt.stride, replan=opt.replan,
                    solver=opt.solver,
                    micro_pen_ensemble=opt.micro_pen, micro_pen_weight=opt.micro_pen_weight,
                    temporal_ensemble=False
                )
                preds.append(pred_i)
            pred = torch.cat(preds, dim=0)

