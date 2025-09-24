import argparse, os, pickle, lmdb, tqdm, torch
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from data_loader.loader import ScriptDataset
from utils.util import writeCache, dxdynp_to_list, coords_render

# 모델들
from models.model import SDT_Generator
from models.sdt_flow_wrapper import SDT_FlowWrapper   # 래퍼 파일이 models/ 아래 있어야 합니다.

def load_flow_wrapper(cfg, ckpt_path, device, H=64):
    """SDT 기본 모델 위에 Flow Wrapper 세팅 + 체크포인트 로드"""
    sdt = SDT_Generator(
        num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
        num_head_layers=cfg.MODEL.NUM_HEAD_LAYERS,
        wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
        gly_dec_layers=cfg.MODEL.GLY_DEC_LAYERS
    ).to(device)
    flow = SDT_FlowWrapper(sdt, H=H).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model_state = state.get("model", state)

    # DataParallel 호환
    if any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}

    # wrapper에 바로 로드 (내부에서 필요한 키만 가져가게 설계)
    missing, unexpected = flow.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        # 너무 장황한 프린트 방지 – 요약만 출력
        print(f"[load] missing: {len(missing)} keys, unexpected: {len(unexpected)} keys")

    flow.eval()
    return flow

def main(opt):
    # ----- Config & Device -----
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device = {device}")

    # ----- Dataloader (원본 test.py와 동일) -----
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

    # ----- 출력 폴더 & LMDB env (test_env) -----
    os.makedirs(os.path.join(opt.save_dir, 'test'), exist_ok=True)
    test_env = lmdb.open(os.path.join(opt.save_dir, 'test'), map_size=1099511627776)
    pickle.dump(writer_dict, open(os.path.join(opt.save_dir, 'writer_dict.pkl'), 'wb'))
    pickle.dump(char_dict, open(os.path.join(opt.save_dir, 'character_dict.pkl'), 'wb'))

    # ----- 모델 로드 -----
    if not opt.pretrained_model or not os.path.isfile(opt.pretrained_model):
        raise IOError('input the correct checkpoint path')
    print(f'load pretrained model from {opt.pretrained_model}')
    flow = load_flow_wrapper(cfg, opt.pretrained_model, device, H=opt.H)

    # ----- 몇 배치 생성할지 (원본 test.py 로직과 동일하게) -----
    if opt.sample_size == 'all':
        batch_samples = len(test_loader)
    else:
        batch_samples = int(opt.sample_size) * len(writer_dict) // cfg.TRAIN.IMS_PER_BATCH
        batch_samples = min(batch_samples, len(test_loader))

    data_iter = iter(test_loader)
    num_count = 0

    with torch.no_grad():
        for _ in tqdm.tqdm(range(batch_samples)):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(test_loader)
                data = next(data_iter)

            # ----- 입력 준비 -----
            coords       = data['coords'].to(device)              # [B, T_gt, 5]
            character_id = data['character_id'].long().to(device) # [B]
            writer_id    = data['writer_id'].long().to(device)    # [B]
            img_list     = data['img_list'].to(device)            # [B, N, 1, H, W]
            char_img     = data['char_img'].to(device)            # content encoder 입력

            B = coords.size(0)

            # ----- Flow inference: Δx,Δy 생성 -----
            # T_body는 생성할 본문 길이(맨 앞 SOS 제외). 너무 길면 평가가 꼬일 수 있으니 opt.T 사용.
            T_body = opt.T
            delta = flow.flow_infer(img_list, char_img, T=T_body, steps=opt.steps)  # [B, T_body, 2]
            assert delta.dim() == 3 and delta.size(-1) == 2, "flow_infer must return [B,T,2]"

            # ----- 원본 포맷으로 preds 조립 -----
            preds = torch.zeros(B, T_body, 5, device=device, dtype=torch.float32)
            preds[..., :2] = delta
            # 본문 구간은 pen-down (0,1,0)
            preds[..., 2] = 0.0  # P0
            preds[..., 3] = 1.0  # P1
            preds[..., 4] = 0.0  # P2

            # SOS 추가
            SOS = torch.tensor([[0., 0., 1., 0., 0.]], device=device, dtype=torch.float32).expand(B, 1, 5)
            preds = torch.cat([SOS, preds], dim=1)  # [B, T_body+1, 5]

            # 마지막 프레임 EOS 보장
            preds[:, -1, :2] = 0.0
            preds[:, -1, 2:] = torch.tensor([0., 0., 1.], device=device, dtype=torch.float32)

            # ----- numpy 변환 (dxdynp_to_list는 numpy 기대) -----
            preds_np  = preds.detach().cpu().numpy()
            coords_np = coords.detach().cpu().numpy()

            # ----- 저장 (원본 test.py와 동일한 LMDB 구조) -----
            test_cache = {}
            if opt.store_type == 'online':
                for i in range(B):
                    pred_list, _  = dxdynp_to_list(preds_np[i])   # numpy OK
                    coord_list, _ = dxdynp_to_list(coords_np[i])
                    rec = {
                        'coordinates': pred_list,
                        'writer_id': writer_id[i].item(),
                        'character_id': character_id[i].item(),
                        'coords_gt': coord_list
                    }
                    data_byte = pickle.dumps(rec)
                    data_id = str(num_count).encode('utf-8')
                    test_cache[data_id] = data_byte
                    num_count += 1
                # num_sample 매 배치 갱신 (원본과 동일)
                test_cache['num_sample'.encode('utf-8')] = str(num_count).encode()
                writeCache(test_env, test_cache)

            elif opt.store_type == 'img':
                for i in range(B):
                    sk_pil = coords_render(preds_np[i], split=True, width=256, height=256, thickness=8, board=0)
                    character = char_dict[character_id[i].item()]
                    save_path = os.path.join(opt.save_dir, 'test', f"{writer_id[i].item()}_{character}.png")
                    try:
                        sk_pil.save(save_path)
                    except Exception as e:
                        print(f'[Save Error] {save_path}: {e}')
            else:
                raise NotImplementedError('only support online or img format')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', dest='cfg_file', default='configs/English_CASIA.yml',
                    help='Config file for testing')
    ap.add_argument('--pretrained_model', required=True,
                    help='flow 래퍼 체크포인트 (.pth)')
    ap.add_argument('--dir', dest='save_dir', default='Generated/Flow',
                    help='결과를 저장할 폴더(안에 test/ LMDB 생성)')
    ap.add_argument('--store_type', default='online', choices=['online', 'img'])
    ap.add_argument('--sample_size', default='500',
                    help="'all' 또는 정수 문자열 (원본 test.py와 동일 로직)")
    # Flow inference 관련
    ap.add_argument('--H', type=int, default=64, help='flow wrapper 내부 horizon 크기(모델 구성 파라미터)')
    ap.add_argument('--T', type=int, default=120, help='생성 본문 길이 (SOS 제외)')
    ap.add_argument('--steps', type=int, default=10, help='flow ODE/solver 스텝 수(래퍼 구현에 맞게)')
    args = ap.parse_args()
    main(args)
