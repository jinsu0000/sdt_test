import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import ScriptDataset
import pickle
from models.model import SDT_Generator
import tqdm
from utils.util import writeCache, dxdynp_to_list, coords_render
import lmdb

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()

    """setup device"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    """setup data_loader instances"""
    test_dataset = ScriptDataset(
       cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, cfg.TEST.ISTRAIN, cfg.MODEL.NUM_IMGS)
    print('number of test images: ', len(test_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                              shuffle=True,
                                              sampler=None,
                                              drop_last=False,
                                              collate_fn=test_dataset.collate_fn_,
                                              num_workers=cfg.DATA_LOADER.NUM_THREADS)
    char_dict = test_dataset.char_dict
    writer_dict = test_dataset.writer_dict

    os.makedirs(os.path.join(opt.save_dir, 'test'), exist_ok=True)
    test_env = lmdb.open(os.path.join(opt.save_dir, 'test'), map_size=1099511627776)
    pickle.dump(writer_dict, open(os.path.join(opt.save_dir, 'writer_dict.pkl'), 'wb'))
    pickle.dump(char_dict, open(os.path.join(opt.save_dir, 'character_dict.pkl'), 'wb'))

    """build model architecture"""
    model = SDT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
            num_head_layers= cfg.MODEL.NUM_HEAD_LAYERS,
            wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
            gly_dec_layers= cfg.MODEL.GLY_DEC_LAYERS).to(device)
    if len(opt.pretrained_model) > 0:
        print('load pretrained model from {}'.format(opt.pretrained_model))
        model_weight = torch.load(opt.pretrained_model, map_location=device)

        if isinstance(model_weight, dict) and 'model' in model_weight:
            model_state_dict = model_weight['model']
            print(f"Type of checkpoint['model']: {type(model_state_dict)}")

            # model_state_dict 안의 key 몇 개만 출력
            for i, key in enumerate(model_state_dict.keys()):
                if i >= 10:
                    break
                print(f"{i}: {key}")

            new_state_dict = {}
            for k, v in model_state_dict.items():
                new_key = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(model_weight)
        #model.load_state_dict(model_weight)
        print('load pretrained model from {}'.format(opt.pretrained_model))
    else:
        raise IOError('input the correct checkpoint path')
    model.eval()

    """calculate the total batches of generated samples"""
    if opt.sample_size == 'all':
        batch_samples = len(test_loader)
    else:
        batch_samples = int(opt.sample_size)*len(writer_dict)//cfg.TRAIN.IMS_PER_BATCH

    num_count = 0
    # --- 수정된 부분 1: data_iter = iter(test_loader) 라인 삭제 ---
    # 수동으로 이터레이터를 만들 필요가 없습니다.

    with torch.no_grad():
        # --- 수정된 부분 2: for 루프를 dataloader에 직접 사용하고, enumerate로 배치 번호를 관리 ---
        # tqdm.tqdm으로 감싸서 진행률 표시는 그대로 유지합니다.
        for batch_num, data in enumerate(tqdm.tqdm(test_loader)):
            
            # --- 수정된 부분 3: 생성하려는 샘플 수를 초과하면 루프를 빠져나옴 ---
            # 기존의 batch_num > batch_samples 로직을 루프의 시작점으로 옮겼습니다.
            if batch_num >= batch_samples:
                break
            
            # --- 수정된 부분 4: data = next(data_iter) 라인 삭제 ---
            # for 루프가 자동으로 다음 데이터를 'data' 변수에 할당해줍니다.

            # prepare input
            coords, coords_len, character_id, writer_id, img_list, char_img = data['coords'].to(device), \
                data['coords_len'].to(device), \
                data['character_id'].long().to(device), \
                data['writer_id'].long().to(device), \
                data['img_list'].to(device), \
                data['char_img'].to(device)
            preds = model.inference(img_list, char_img, 120)
            bs = character_id.shape[0]
            SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]], device=device).unsqueeze(1).to(preds)
            preds = torch.cat((SOS, preds), 1)  # add the SOS token like GT
            preds = preds.detach().cpu().numpy()

            test_cache = {}
            coords = coords.detach().cpu().numpy()
            if opt.store_type == 'online':
                for i, pred in enumerate(preds):
                    pred, _ = dxdynp_to_list(preds[i])
                    coord, _ = dxdynp_to_list(coords[i])
                    data = {'coordinates': pred, 'writer_id': writer_id[i].item(),
                            'character_id': character_id[i].item(), 'coords_gt': coord}
                    data_byte = pickle.dumps(data)
                    data_id = str(num_count).encode('utf-8')
                    test_cache[data_id] = data_byte
                    num_count += 1
                test_cache['num_sample'.encode('utf-8')] = str(num_count).encode()
                writeCache(test_env, test_cache)
            elif opt.store_type == 'img':
                for i, pred in enumerate(preds):
                    """intends to blur the boundaries of each sample to fit the actual using situations,
                        as suggested in 'Deep imitator: Handwriting calligraphy imitation via deep attention networks'"""
                    sk_pil = coords_render(preds[i], split=True, width=256, height=256, thickness=8, board=0)
                    character = char_dict[character_id[i].item()]
                    save_path = os.path.join(opt.save_dir, 'test',
                                           str(writer_id[i].item()) + '_' + character + '.png')
                    try:
                        sk_pil.save(save_path)
                    except:
                        print('error. %s, %s, %s' % (save_path, str(writer_id[i].item()), character))
            else:
                raise NotImplementedError('only support online or img format')

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/CHINESE_CASIA.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated/Chinese', help='target dir for storing the generated characters')
    parser.add_argument('--pretrained_model', dest='pretrained_model', default='', required=True, help='continue train model')
    parser.add_argument('--store_type', dest='store_type', required=True, default='online', help='online or img')
    parser.add_argument('--sample_size', dest='sample_size', default='500', required=True, help='randomly generate a certain number of characters for each writer')
    opt = parser.parse_args()
    main(opt)

    '''
    python test.py --cfg configs/English_CASIA.yml --pretrained_model Saved/English_CASIA/English_log-20251124_075009/model/checkpoint-iter199999.pth --store_type online --sample_size 500 --dir Generated/English > inference.log 2>&1 &
    python test.py --cfg configs/English_CASIA.yml --pretrained_model Saved/English_CASIA/English_log-20251124_075009/model/checkpoint-iter199999.pth --store_type img --sample_size 500 --dir Generated/English > inference.log 2>&1 &
    '''