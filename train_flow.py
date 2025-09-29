# train_flow.py
import argparse, os, torch, random, numpy as np
import torch.backends.cudnn as cudnn
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed
from utils.logger import set_log, print_once
from data_loader.loader import ScriptDataset
from torch.utils.data import DataLoader

from models.model import SDT_Generator
from models.sdt_flow_wrapper import SDT_FlowWrapper
from trainer.trainer_flow import TrainerFlow

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True; cudnn.benchmark = False
    print(f"[SEED] {seed}")

def get_dataloaders(cfg):
    train_dataset = ScriptDataset(cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET,
                                  cfg.TRAIN.ISTRAIN, cfg.MODEL.NUM_IMGS)
    test_dataset  = ScriptDataset(cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET,
                                  cfg.TEST.ISTRAIN,  cfg.MODEL.NUM_IMGS)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH,
                              shuffle=True, drop_last=False, collate_fn=train_dataset.collate_fn_,
                              num_workers=cfg.DATA_LOADER.NUM_THREADS)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg.TRAIN.IMS_PER_BATCH,
                              shuffle=True, drop_last=False, collate_fn=test_dataset.collate_fn_,
                              num_workers=cfg.DATA_LOADER.NUM_THREADS)
    return train_dataset, test_dataset, train_loader, test_loader

def main(opt):
    # --- cfg / seed ---
    cfg_from_file(opt.cfg_file); assert_and_infer_cfg()
    fix_seed(cfg.TRAIN.SEED); seed_all(cfg.TRAIN.SEED)

    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)
    train_dataset, test_dataset, train_loader, test_loader = get_dataloaders(cfg)
    print_once(f"Number of train images: {len(train_dataset)} | test images: {len(test_dataset)}")
    char_dict = test_dataset.char_dict

    # --- model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sdt = SDT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
                        num_head_layers=cfg.MODEL.NUM_HEAD_LAYERS,
                        wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
                        gly_dec_layers=cfg.MODEL.GLY_DEC_LAYERS).to(device)

    # 래퍼: condition_mode/keep_k 없이 cond=1토큰 경로 고정, 학습 stride는 stride_default로 전달
    flow = SDT_FlowWrapper(
        sdt,
        H=opt.H,
        n_layers=opt.layers,
        n_head=opt.heads,
        ffn_mult=opt.ffn_mult,
        stride_default=opt.stride
    ).to(device)

    if torch.cuda.device_count() > 1:
        flow = torch.nn.DataParallel(flow).to(device)

    optm = torch.optim.AdamW(flow.parameters(), lr=opt.lr, weight_decay=1e-4)

    # (선택) 프리트레인 로드
    if opt.pretrained_model:
        state = torch.load(opt.pretrained_model, map_location=device)
        model = flow.module if isinstance(flow, torch.nn.DataParallel) else flow
        try:
            model.load_state_dict(state.get("model", state), strict=False)
        except Exception:
            model.load_state_dict({k.replace("module.", ""): v for k, v in state.get("model", state).items()},
                                  strict=False)

    # --- trainer ---
    trainer = TrainerFlow(
        flow, optm, train_loader, logs, char_dict, valid_data_loader=test_loader,
        nce_writer_weight=opt.nce_w_writer, nce_glyph_weight=opt.nce_w_glyph, nce_temp=opt.nce_temp,
        nce_w_writer_init=opt.nce_w_writer_init, nce_w_glyph_init=opt.nce_w_glyph_init,
        nce_w_writer_final=opt.nce_w_writer_final, nce_w_glyph_final=opt.nce_w_glyph_final,
        nce_warm_steps=opt.nce_warm_steps, nce_decay_steps=opt.nce_decay_steps
    )

    max_iter = getattr(cfg.SOLVER, "MAX_ITER", opt.max_iter)
    print_once(f"[INFO] max_iter: {max_iter}")
    trainer.train(max_iter=max_iter)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', dest='cfg_file', default='configs/English_CASIA.yml')
    ap.add_argument('--log', dest='log_name', default='flow_run')
    ap.add_argument('--pretrained_model', default='')
    # FlowPolicy / chunk 하이퍼
    ap.add_argument('--H', type=int, default=6)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--ffn_mult', type=int, default=4)
    ap.add_argument('--stride', type=int, default=3, help='학습 윈도우 stride (= R 기본값)')
    # Optim
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--max_iter', type=int, default=200000)
    # (선택) NCE 관련 — TrainerFlow가 사용하면 활성, 아니면 무시
    ap.add_argument('--nce_w_writer', type=float, default=0.1)
    ap.add_argument('--nce_w_glyph',  type=float, default=0.1)
    ap.add_argument('--nce_temp',     type=float, default=0.07)
    ap.add_argument('--nce_w_writer_init', type=float, default=0.5)
    ap.add_argument('--nce_w_glyph_init',  type=float, default=0.5)
    ap.add_argument('--nce_w_writer_final', type=float, default=0.1)
    ap.add_argument('--nce_w_glyph_final',  type=float, default=0.1)
    ap.add_argument('--nce_warm_steps', type=int, default=30000)
    ap.add_argument('--nce_decay_steps', type=int, default=60000)
    args = ap.parse_args(); main(args)
