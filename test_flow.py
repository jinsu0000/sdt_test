
import argparse, os, pickle, torch, tqdm, lmdb
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from data_loader.loader import ScriptDataset
from utils.util import writeCache, dxdynp_to_list, coords_render
from model import SDT_Generator
from sdt_flow_wrapper import SDT_FlowWrapper

def main(opt):
    cfg_from_file(opt.cfg_file); assert_and_infer_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = ScriptDataset(cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, cfg.TEST.ISTRAIN, cfg.MODEL.NUM_IMGS)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH, shuffle=True, drop_last=False, collate_fn=test_dataset.collate_fn_, num_workers=cfg.DATA_LOADER.NUM_THREADS)
    char_dict = test_dataset.char_dict; writer_dict = test_dataset.writer_dict
    os.makedirs(os.path.join(opt.save_dir, 'test'), exist_ok=True)
    test_env = lmdb.open(os.path.join(opt.save_dir, 'test'), map_size=1099511627776)
    pickle.dump(writer_dict, open(os.path.join(opt.save_dir, 'writer_dict.pkl'), 'wb'))
    pickle.dump(char_dict, open(os.path.join(opt.save_dir, 'character_dict.pkl'), 'wb'))
    sdt = SDT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS, num_head_layers=cfg.MODEL.NUM_HEAD_LAYERS, wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS, gly_dec_layers=cfg.MODEL.GLY_DEC_LAYERS).to(device)
    flow = SDT_FlowWrapper(sdt, H=opt.H).to(device)
    state = torch.load(opt.pretrained_model, map_location=device); model_state = state.get("model", state)
    try: flow.load_state_dict(model_state)
    except: flow.load_state_dict({k.replace("module.", ""): v for k,v in model_state.items()})
    flow.eval()
    if opt.sample_size == 'all': batch_samples = len(test_loader)
    else: batch_samples = int(opt.sample_size)*len(writer_dict)//cfg.TRAIN.IMS_PER_BATCH
    data_iter = iter(test_loader); num_count = 0
    with torch.no_grad():
        for _ in tqdm.tqdm(range(batch_samples)):
            try: data = next(data_iter)
            except StopIteration: data_iter = iter(test_loader); data = next(data_iter)
            coords = data['coords'].to(device); character_id = data['character_id'].long().to(device)
            writer_id = data['writer_id'].long().to(device); img_list = data['img_list'].to(device); char_img = data['char_img'].to(device)
            B, T, _ = coords.shape
            delta = flow.flow_infer(img_list, char_img, T=min(opt.T, T), steps=opt.steps)  # [B,T,2]
            preds = torch.zeros(B, delta.size(1), 5, device=device); preds[..., :2] = delta
            SOS = torch.tensor(B * [[0, 0, 1, 0, 0]], device=device).unsqueeze(1); preds = torch.cat((SOS, preds), 1).cpu().numpy()
            if opt.store_type == 'online':
                test_cache = {}; coords_np = coords.detach().cpu().numpy()
                for i in range(B):
                    pred, _ = dxdynp_to_list(preds[i]); coord, _ = dxdynp_to_list(coords_np[i])
                    rec = {'coordinates': pred, 'writer_id': writer_id[i].item(), 'character_id': character_id[i].item(), 'coords_gt': coord}
                    data_byte = pickle.dumps(rec); data_id = str(num_count).encode('utf-8'); test_cache[data_id] = data_byte; num_count += 1
                test_cache['num_sample'.encode('utf-8')] = str(num_count).encode(); writeCache(test_env, test_cache)
            elif opt.store_type == 'img':
                for i in range(B):
                    sk_pil = coords_render(preds[i], split=True, width=256, height=256, thickness=8, board=0)
                    character = char_dict[character_id[i].item()]
                    save_path = os.path.join(opt.save_dir, 'test', f"{writer_id[i].item()}_{character}.png")
                    try: sk_pil.save(save_path)
                    except Exception as e: print(f"[Save Error] {save_path}: {e}")
            else: raise NotImplementedError('only support online or img format')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', dest='cfg_file', default='configs/English_CASIA.yml')
    ap.add_argument('--pretrained_model', required=True)
    ap.add_argument('--save_dir', default='Generated/Flow')
    ap.add_argument('--store_type', default='online')
    ap.add_argument('--sample_size', default='500')
    ap.add_argument('--H', type=int, default=64)
    ap.add_argument('--T', type=int, default=120)
    ap.add_argument('--steps', type=int, default=10)
    args = ap.parse_args(); main(args)
