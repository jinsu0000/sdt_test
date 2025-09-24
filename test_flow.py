import argparse
import os
import pickle
import lmdb
import tqdm
import torch

from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from data_loader.loader import ScriptDataset
from models.model import SDT_Generator
from utils.util import writeCache, dxdynp_to_list, coords_render


def main(opt):
    # ---- config ----
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()

    # ---- device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # ---- data ----
    test_dataset = ScriptDataset(
        cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, cfg.TEST.ISTRAIN, cfg.MODEL.NUM_IMGS
    )
    print('number of test images: ', len(test_dataset))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        shuffle=True,
        sampler=None,
        drop_last=False,
        collate_fn=test_dataset.collate_fn_,
        num_workers=cfg.DATA_LOADER.NUM_THREADS
    )
    char_dict = test_dataset.char_dict
    writer_dict = test_dataset.writer_dict

    # ---- output (LMDB) ----
    os.makedirs(os.path.join(opt.save_dir, 'test'), exist_ok=True)
    test_env = lmdb.open(os.path.join(opt.save_dir, 'test'), map_size=1099511627776)
    pickle.dump(writer_dict, open(os.path.join(opt.save_dir, 'writer_dict.pkl'), 'wb'))
    pickle.dump(char_dict, open(os.path.join(opt.save_dir, 'character_dict.pkl'), 'wb'))

    # ---- model ----
    model = SDT_Generator(
        num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
        num_head_layers=cfg.MODEL.NUM_HEAD_LAYERS,
        wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
        gly_dec_layers=cfg.MODEL.GLY_DEC_LAYERS
    ).to(device)

    if len(opt.pretrained_model) == 0:
        raise IOError('input the correct checkpoint path')

    print('load pretrained model from {}'.format(opt.pretrained_model))
    state = torch.load(opt.pretrained_model, map_location=device)
    state = state.get('model', state)
    try:
        model.load_state_dict(state, strict=True)
    except:
        fixed = { (k.replace('module.', '') if k.startswith('module.') else k): v for k, v in state.items() }
        model.load_state_dict(fixed, strict=True)
    model.eval()

    # ---- how many batches to generate (원본과 동일 규약) ----
    if opt.sample_size == 'all':
        batch_samples = len(test_loader)
    else:
        batch_samples = int(opt.sample_size) * len(writer_dict) // cfg.TRAIN.IMS_PER_BATCH
        batch_samples = min(batch_samples, len(test_loader))

    # ---- chunked inference 고정 하이퍼파라미터 (원본 포맷 유지에 최적) ----
    TOTAL_LEN = 120         # 원본 inference와 동일 길이
    CHUNK     = 4           # H
    REPLAN    = 1           # R (매 스텝 재계획)
    DECAY     = 1.0         # 감쇠 없음 (가장 단순/안정)
    USE_SIGMA = True        # 원본과 가장 유사한 샘플 성질 유지

    # ---- loop ----
    data_iter = iter(test_loader)
    num_count = 0

    with torch.no_grad():
        for _ in tqdm.tqdm(range(batch_samples)):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(test_loader)
                data = next(data_iter)

            # prepare inputs
            coords         = data['coords'].to(device)                       # [B, T, 5] (GT)
            coords_len     = data['coords_len'].to(device)                   # [B]
            character_id   = data['character_id'].long().to(device)          # [B]
            writer_id      = data['writer_id'].long().to(device)             # [B]
            img_list       = data['img_list'].to(device)                     # [B, N, 1, H, W]
            char_img       = data['char_img'].to(device)                     # [*, B, *] (원본과 동일)

            # ---- *** 핵심: chunk+ensemble but original-compatible output ***
            preds = model.inference_chunk_ensemble(
                style_imgs=img_list,
                char_img=char_img,
                total_len=TOTAL_LEN,
                chunk_size=CHUNK,
                replan=REPLAN,
                age_decay=DECAY,
                use_sigma_sampling=USE_SIGMA
            )  # [B, T(=TOTAL_LEN), 5] float32

            # ---- SOS 토큰 추가 (원본과 동일) ----
            bs = character_id.shape[0]
            SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]], device=device).unsqueeze(1).to(preds)
            preds = torch.cat((SOS, preds), 1)  # [B, T+1, 5]
            preds = preds.detach().cpu().numpy()

            # ---- 저장 (원본과 완전히 동일 포맷/경로) ----
            test_cache = {}
            coords_np = coords.detach().cpu().numpy()

            if opt.store_type == 'online':
                for i in range(bs):
                    pred_list, _  = dxdynp_to_list(preds[i])      # (dx,dy,pen) 리스트 포맷
                    coord_list, _ = dxdynp_to_list(coords_np[i])  # GT도 동일 포맷
                    rec = {
                        'coordinates': pred_list,
                        'writer_id': writer_id[i].item(),
                        'character_id': character_id[i].item(),
                        'coords_gt': coord_list
                    }
                    data_byte = pickle.dumps(rec)
                    data_id   = str(num_count).encode('utf-8')
                    test_cache[data_id] = data_byte
                    num_count += 1
                test_cache['num_sample'.encode('utf-8')] = str(num_count).encode()
                writeCache(test_env, test_cache)

            elif opt.store_type == 'img':
                for i in range(bs):
                    sk_pil   = coords_render(preds[i], split=True, width=256, height=256, thickness=8, board=0)
                    character = char_dict[character_id[i].item()]
                    save_path = os.path.join(opt.save_dir, 'test', f"{writer_id[i].item()}_{character}.png")
                    try:
                        sk_pil.save(save_path)
                    except:
                        print('error. %s, %s, %s' % (save_path, str(writer_id[i].item()), character))
            else:
                raise NotImplementedError('only support online or img format')


if __name__ == '__main__':
    # *** 원본과 동일한 CLI ***
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/CHINESE_CASIA.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated/Chinese',
                        help='target dir for storing the generated characters')
    parser.add_argument('--pretrained_model', dest='pretrained_model', default='', required=True,
                        help='continue train model')
    parser.add_argument('--store_type', dest='store_type', required=True, default='online',
                        help='online or img')
    parser.add_argument('--sample_size', dest='sample_size', default='500', required=True,
                        help='randomly generate a certain number of characters for each writer')
    opt = parser.parse_args()
    main(opt)
