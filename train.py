import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed, load_specific_dict
from models.loss import SupConLoss, get_pen_loss
from models.model import SDT_Generator
from utils.logger import set_log
from data_loader.loader import ScriptDataset
import os
import re
import torch
import torch.nn as nn  # ✅ 추가: DP 사용
from trainer.trainer import Trainer
from utils.logger import print_once
import collections

# ✅ DP/비-DP 간 state_dict 키(prefix) 자동 정규화
def _smart_load_state_dict(model, state_dict, strict=False):
    """
    model: nn.Module 또는 nn.DataParallel
    state_dict: torch.load(...) 결과(dict 또는 {'model': dict, ...})
    """
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    is_dp = isinstance(model, nn.DataParallel)
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    new_sd = collections.OrderedDict()

    if is_dp and not has_module:
        for k, v in state_dict.items():
            new_sd["module." + k] = v
    elif (not is_dp) and has_module:
        for k, v in state_dict.items():
            new_sd[k[len("module."):]] = v
    else:
        new_sd = state_dict

    msg = model.load_state_dict(new_sd, strict=strict)
    try:
        missing = getattr(msg, "missing_keys", [])
        unexpected = getattr(msg, "unexpected_keys", [])
        if missing or unexpected:
            print_once(f"[load] missing={missing[:5]}{' ...' if len(missing)>5 else ''}")
            print_once(f"[load] unexpected={unexpected[:5]}{' ...' if len(unexpected)>5 else ''}")
    except Exception:
        pass


def get_dataset(cfg, is_train_dataset=True):
    dataset = ScriptDataset(
        root=cfg.DATA_LOADER.PATH,
        dataset=cfg.DATA_LOADER.DATASET,
        is_train=cfg.TRAIN.ISTRAIN if is_train_dataset else cfg.TEST.ISTRAIN,
        num_img=cfg.MODEL.NUM_IMGS
    )
    return dataset


def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()

    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    """ prepare log file """
    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)

    """ set dataset"""
    train_dataset = get_dataset(cfg, True)
    print('number of training images: ', len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        shuffle=True,
        drop_last=False,
        collate_fn=train_dataset.collate_fn_,  # ✅ 오타 수정: 뒤의 ',f' 제거
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        pin_memory=True
    )
    test_dataset = get_dataset(cfg, False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        shuffle=True,
        sampler=None,
        drop_last=False,
        collate_fn=test_dataset.collate_fn_,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        pin_memory=True
    )
    print_once(f"Number of test images: {len(test_dataset)}, Number of train images: {len(train_dataset)}")

    char_dict = test_dataset.char_dict

    """ build model, criterion and optimizer"""
    model = SDT_Generator(
        num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
        num_head_layers=cfg.MODEL.NUM_HEAD_LAYERS,
        wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
        gly_dec_layers=cfg.MODEL.GLY_DEC_LAYERS
    ).cuda()

    # ✅ Multi-GPU(DataParallel) 조건부 적용
    use_multi = (getattr(cfg, "NUM_GPUS", 1) > 1) and (torch.cuda.device_count() > 1)
    if use_multi:
        device_ids = list(range(min(cfg.NUM_GPUS, torch.cuda.device_count())))
        print_once(f"[train] Using DataParallel on device_ids={device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        print("[train] Single-GPU mode (no DataParallel).")

    # ✅ DP/비-DP 모두 안전한 optimizer 대상
    net = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.SOLVER.BASE_LR)

    start_checkpoint_step = 0

    ### load checkpoint
    if len(opt.pretrained_model) > 0:
        print('load pretrained model from {}'.format(opt.pretrained_model))
        state_dict = torch.load(opt.pretrained_model, map_location="cpu")

        # ✅ 모델 파라미터 로드 (DP/비-DP 자동 정규화, strict=False로 신규 VQ 파라미터 무시)
        _smart_load_state_dict(model, state_dict, strict=False)

        # 옵티마이저 및 스텝 복원(있을 때만)
        if isinstance(state_dict, dict):
            if 'optimizer' in state_dict:
                try:
                    optimizer.load_state_dict(state_dict['optimizer'])
                    print_once("[Resume] Optimizer state loaded.")
                except Exception as e:
                    print_once(f"[Resume] Optimizer load skipped: {e}")

            if 'step' in state_dict:
                start_checkpoint_step = int(state_dict['step'])
                print(f"[Resume] Resuming from step {start_checkpoint_step}")

    elif len(opt.content_pretrained) > 0:
        # content encoder만 로드 (DP 고려)
        target = net.content_encoder
        model_dict = load_specific_dict(target, opt.content_pretrained, "feature_ext")
        target.load_state_dict(model_dict)
        print('load content pretrained model from {}'.format(opt.content_pretrained))
    else:
        pass

    criterion = dict(NCE=SupConLoss(contrast_mode='all'), PEN=get_pen_loss)

    """start training iterations"""
    trainer = Trainer(model, criterion, optimizer, train_loader, logs, char_dict, test_loader)
    trainer.train(start_step=start_checkpoint_step)


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', default='',
                        dest='pretrained_model', required=False, help='continue to train model')
    # parser.add_argument('--content_pretrained', default='model_zoo/position_layer2_dim512_iter138k_test_acc0.9443.pth',
    #                   dest='content_pretrained', required=False, help='continue to train content encoder')
    parser.add_argument('--content_pretrained', default='',
                        dest='content_pretrained', required=False, help='continue to train content encoder')
    parser.add_argument('--cfg', dest='cfg_file', default='configs/CHINESE_CASIA.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--log', default='debug',
                        dest='log_name', required=False, help='the filename of log')
    opt = parser.parse_args()
    main(opt)
