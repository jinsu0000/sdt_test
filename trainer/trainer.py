import torch
from tensorboardX import SummaryWriter
import time
from parse_config import cfg
from models.gmm import get_mixture_coef
import os
import datetime
import sys
from utils.util import coords_render
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.logger import print_once

class Trainer:
    def __init__(self, model, criterion, optimizer, data_loader, 
                logs, char_dict, valid_data_loader=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.char_dict = char_dict
        self.valid_data_loader = valid_data_loader
        self.nce_criterion = criterion['NCE']
        self.pen_criterion = criterion['PEN']
        self.tb_summary = SummaryWriter(logs['tboard'])
        self.save_model_dir = logs['model']
        self.save_sample_dir = logs['sample']
      
    def _train_iter(self, data, step):
        self.model.train()
        prev_time = time.time()
        # prepare input
        coords, coords_len, character_id, writer_id, img_list, char_img = data['coords'].cuda(), \
            data['coords_len'].cuda(), \
            data['character_id'].long().cuda(), \
            data['writer_id'].long().cuda(), \
            data['img_list'].cuda(), \
            data['char_img'].cuda()
        
        # forward
        input_seq = coords[:, 1:-1] # GT sequence에서 SOS/END 제외?

        if step == 1:
            #writer_ids = torch.cat([writer_id, torch.full_like(writer_id, -1)])
            print(" # _visualize_input_images_tb img_list:", img_list.shape)
            print(" # _visualize_input_images_tb writer_id:", writer_id.shape)
            self._visualize_input_images_tb(img_list, writer_id, step)

        if (step+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (step+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
            print(" # Style img_list shape from PNG:", img_list.shape)

        vq_alpha = min(1.0, step / cfg.SOLVER.WARMUP_ITERS)
        out = self.model(img_list, input_seq, char_img, vq_alpha)

        if isinstance(out, (tuple, list)) and len(out) >= 5:
            preds, nce_emb, nce_emb_patch, vq_loss, vq_codes = out[:5]
            print_once(f"train_iter vq_loss.shape : {vq_loss.shape}, VQ codes : {vq_codes.shape}, unique : {torch.unique(vq_codes)}")
            vq_loss = vq_loss.mean()
        else:
            preds, nce_emb, nce_emb_patch = out
        print_once(f"train_iter preds : {preds.shape}, nce_emb : {nce_emb.shape}, nce_emb_patch : {nce_emb_patch.shape}")
        
        if step % 10000 == 0: #(step+1) > cfg.TRAIN.VALIDATE_BEGIN  and (step+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
            self._plot_nce_embedding_2d(nce_emb, writer_id, step)

        # calculate loss
        gt_coords = coords[:, 1:, :]
        print_once(f"train_iter GT coords : {gt_coords.shape[0]}, T, {gt_coords.shape[1]}, {gt_coords.shape[2]}], preds : {preds.shape[0]}, T, {preds.shape[1]}, {preds.shape[2]}]")
        nce_loss_writer = self.nce_criterion(nce_emb, labels=writer_id)
        nce_loss_glyph = self.nce_criterion(nce_emb_patch)
        preds = preds.view(-1, 123)
        gt_coords = gt_coords.reshape(-1, 5)
        print_once(f"train_iter preds w/view(-1, 123) : {preds.shape}, gt_coords w/reshape(-1, 5) : {gt_coords.shape}")
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits] = get_mixture_coef(preds)
        moving_loss_all, state_loss = self.pen_criterion(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, \
                                      o_corr, o_pen_logits, gt_coords[:,0].unsqueeze(-1), gt_coords[:,1].unsqueeze(-1), gt_coords[:,2:], step)
        moving_loss = torch.sum(moving_loss_all) / torch.sum(coords_len)
        pen_loss = moving_loss + 2*state_loss
        
                
        # [VQ-PATCH] 가중치
        lambda_vq     = getattr(cfg.TRAIN, "LAMBDA_VQ", 0.25)
        lambda_writer = getattr(cfg.TRAIN, "LAMBDA_WRITER_NCE", 1.0)   # 없으면 1.0
        lambda_glyph  = getattr(cfg.TRAIN, "LAMBDA_GLYPH_NCE", 1.0)

        loss = pen_loss + lambda_writer * nce_loss_writer + lambda_glyph * nce_loss_glyph + lambda_vq * vq_loss

        #loss = pen_loss + nce_loss_writer + nce_loss_glyph
        
        # nan/infinity 체크 추가
        if (torch.isnan(loss) | torch.isinf(loss)).any().item():
            print(f"[!!! NaN Detected] at Step {step}")
            print(f"  loss={loss.detach().float().mean().item():.6f}")
            print(f"  moving_loss={moving_loss.detach().float().mean().item():.6f}, state_loss={state_loss.detach().float().mean().item():.6f}")
            print(f"  nce_writer={nce_loss_writer.detach().float().mean().item():.6f}, nce_glyph={nce_loss_glyph.detach().float().mean().item():.6f}")
            print(f"  vq_loss={float(vq_loss) if not isinstance(vq_loss, torch.Tensor) else vq_loss.detach().float().mean().item():.6f}")
            raise ValueError(f"NaN detected in loss at step {step}")

        # backward and update trainable parameters
        self.model.zero_grad()
        loss.backward()
        if cfg.SOLVER.GRAD_L2_CLIP > 0:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
        self.optimizer.step()

        # log files
        loss_dict = {"pen_loss": pen_loss.item(), "moving_loss": moving_loss.item(),
                    "state_loss": state_loss.item(), "nce_loss_writer": nce_loss_writer.item(),
                    "nce_loss_glyph": nce_loss_glyph.item(), "vq_loss": float(vq_loss)}
        self.tb_summary.add_scalars("loss", loss_dict, step)
        iter_left = cfg.SOLVER.MAX_ITER - step
        time_left = datetime.timedelta(
                    seconds=iter_left * (time.time() - prev_time))
        self._progress(step, loss.item(), time_left)

        del data, preds, loss
        torch.cuda.empty_cache()


    def _valid_iter(self, step):
        self.model.eval()
        print('loading test dataset, the number is', len(self.valid_data_loader))
        try:
            test_loader_iter = iter(self.valid_data_loader)
            test_data = next(test_loader_iter)
        except StopIteration:
            test_loader_iter = iter(self.valid_data_loader)
            test_data = next(test_loader_iter)
        # prepare input
        coords, coords_len, character_id, writer_id, img_list, char_img = test_data['coords'].cuda(), \
            test_data['coords_len'].cuda(), \
            test_data['character_id'].long().cuda(), \
            test_data['writer_id'].long().cuda(), \
            test_data['img_list'].cuda(), \
            test_data['char_img'].cuda()
        
        print_once(f"valid_iter GT coords : {coords.shape[0]}, T, {coords.shape[1]}, {coords.shape[2]}], img_list : {img_list.shape}, char_img : {char_img.shape}")
        print_once(f"valid_iter character_id : {character_id.shape}, writer_id : {writer_id.shape}")
         # forward
        with torch.no_grad():
            preds = self.model.module.inference(img_list, char_img, 120)
            bs = character_id.shape[0]
            print_once(f"bs : {bs}, preds shape : {preds.shape}")
            SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds)
            preds = torch.cat((SOS, preds), 1)  # add the first token
            preds = preds.cpu().numpy()
            gt_coords = coords.cpu().numpy()  # [N, T, C]
            self._vis_genarate_samples(gt_coords, preds, character_id, char_img, step)

    def train(self, start_step=0):
        """start training iterations"""    
        train_loader_iter = iter(self.data_loader)
        
        for step in range(start_step, cfg.SOLVER.MAX_ITER):
            try:
                data = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(self.data_loader)
                data = next(train_loader_iter)
            self._train_iter(data, step)

            if (step+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (step+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
               self._save_checkpoint(step)
            else:
                pass
            if self.valid_data_loader is not None:
                if (step+1) > cfg.TRAIN.VALIDATE_BEGIN  and (step+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                    self._valid_iter(step)
            else:
                pass


    def _progress(self, step, loss, time_left):
        terminal_log = 'iter:%d ' % step
        terminal_log += '%s:%.3f ' % ('loss', loss)
        terminal_log += 'ETA:%s\r\n' % str(time_left)
        sys.stdout.write(terminal_log)

    # def _save_checkpoint(self, step):
    #     model_path = '{}/checkpoint-iter{}.pth'.format(self.save_model_dir, step)
    #     torch.save(self.model.state_dict(), model_path)
    #     print('save model to {}'.format(model_path))

    def _save_checkpoint(self, step):
        model_path = '{}/checkpoint-iter{}.pth'.format(self.save_model_dir, step)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step,
        }, model_path)
        print('✅ Saved full checkpoint to {}'.format(model_path))

    def _vis_genarate_samples(self, gt_coords, preds, character_id, char_img_batch, step):
        img_tensors = []
        batch_idx = 0
        font = ImageFont.load_default()
        for i, _ in enumerate(gt_coords):
            gt_img = coords_render(gt_coords[i], split=True, width=64, height=64, thickness=1)
            pred_img = coords_render(preds[i], split=True, width=64, height=64, thickness=1)

            # char_img: torch.Tensor -> PIL
            char_img_np = (char_img_batch[i].cpu().numpy().squeeze() * 255).astype('uint8')
            char_img_pil = Image.fromarray(char_img_np).convert("RGB").resize((64, 64))

            # 하나로 붙이기: [Content | Pred | GT]
            example_img = Image.new("RGB", (cfg.TEST.IMG_W * 3, cfg.TEST.IMG_H + 12), (255, 255, 255))
            example_img.paste(char_img_pil, (0, 0))                      # Content image
            example_img.paste(pred_img, (cfg.TEST.IMG_W, 0))            # Generated image
            example_img.paste(gt_img, (cfg.TEST.IMG_W * 2, 0))          # Ground-truth image

            # 문자 텍스트 쓰기
            character = self.char_dict[character_id[i].item()]
            draw = ImageDraw.Draw(example_img)
            draw.text((2, cfg.TEST.IMG_H), character, fill=(0, 0, 0), font=font)

            # 저장
            save_path = os.path.join(
                self.save_sample_dir,
                f'ite.{step//100000}-{step//100000 + 100000}',
                f'{character}_{step}_.jpg'
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            try:
                example_img.save(save_path)
            except Exception as e:
                print(f"[Save Error] {save_path}, error: {e}")

            # TensorBoard용 버퍼
            if self.tb_summary is not None:
                try:
                    img_tensor = ToTensor()(example_img)
                    img_tensors.append(img_tensor)
                except Exception as e:
                    print(f"[ToTensor Error] {save_path}, error: {e}")

            # 10개씩 TensorBoard에 기록
            if len(img_tensors) == 10:
                try:
                    grid = make_grid(img_tensors, nrow=5, padding=4)
                    self.tb_summary.add_image(f"Samples/Step_{step}_batch{batch_idx}", grid, step)
                    batch_idx += 1
                except Exception as e:
                    print(f"[Grid Write Error] step {step} batch {batch_idx}, error: {e}")
                img_tensors = []

        # 마지막 버퍼 처리
        if self.tb_summary is not None and img_tensors:
            try:
                grid = make_grid(img_tensors, nrow=5, padding=4)
                self.tb_summary.add_image(f"Samples/Step_{step}_batch{batch_idx}", grid, step)
            except Exception as e:
                print(f"[Final Grid Write Error] step {step} batch {batch_idx}, error: {e}")

    def _plot_nce_embedding_2d(self, nce_emb, labels, step, var_threshold=1e-2):
        tag = "WriterNCE/query_vs_positive"
        B = nce_emb.size(0)

        emb_all = nce_emb.view(-1, nce_emb.size(-1))  # [2B, 256]
        emb_all_np = emb_all.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy().astype(int)
        labels_all = np.concatenate([labels_np, labels_np], axis=0)  # [2B]
        pair_type = ["query"] * B + ["positive"] * B

        # t-SNE
        embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(emb_all_np)
        cmap = plt.cm.get_cmap('nipy_spectral', labels.max().item() + 1)

        # mean & variance
        variances = emb_all_np.var(axis=0)  # [256]
        means = emb_all_np.mean(axis=0)     # [256]
        low_var_dims = [i for i, v in enumerate(variances) if v < var_threshold]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9), gridspec_kw={'height_ratios': [3, 1]})

        # 💠 t-SNE SCATTER
        for i in range(2 * B):
            color = cmap(labels_all[i])
            marker = 'o' if pair_type[i] == "query" else 'x'
            ax1.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c=[color], marker=marker, s=20)
        ax1.set_title(f"Writer NCE - Step {step}")
        ax1.axis('off')

        # 📊 MEAN & VARIANCE PLOT
        dim_range = np.arange(256)
        ax2.plot(dim_range, means, color='blue', label='Mean (μ)')
        ax2.plot(dim_range, variances, color='red', label='Variance (σ²)')
        ax2.axhline(var_threshold, color='gray', linestyle='--', linewidth=0.8)

        for i in low_var_dims:
            ax2.scatter(i, variances[i], color='red', marker='x', s=25)
            ax2.annotate(f"{i}", (i, variances[i]), fontsize=6, ha='center')

        ax2.set_title(f"Low-var dims: {len(low_var_dims)} / 256")
        ax2.set_xlabel("Embedding Dimension")
        ax2.set_ylabel("μ / σ²")
        ax2.legend(fontsize=7, loc="upper right")

        # SAVE & LOG
        os.makedirs(self.save_sample_dir, exist_ok=True)
        img_path = os.path.join(self.save_sample_dir, f"nce_query_positive_step_{step}.png")
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

        if self.tb_summary:
            image = Image.open(img_path).convert("RGB")
            tensor_img = ToTensor()(image)
            self.tb_summary.add_image(tag, tensor_img, step)

    def _visualize_input_images_tb(self, style_imgs, writer_id, step, nrow=15):
        tag = "Input style_imgs"

        B, N, C, H, W = style_imgs.shape
        imgs = style_imgs.view(B * N, C, H, W).cpu()
        imgs_rgb = imgs.repeat(1, 3, 1, 1)

        # 라벨 텍스트: writer+char+batch idx
        labels = [
            f"w{writer_id[b].item()}\nb{b},n{n}"
            for b in range(B) for n in range(N)
        ]

        # 색상: 앞 N//2 → Blue(Positive), 뒤 N//2 → Red(Anchor)
        colors = [(0, 0, 255)] * (N // 2) + [(255, 0, 0)] * (N - N // 2)
        colors = colors * B

        # 그리드 이미지 생성
        grid = make_grid(imgs_rgb, nrow=nrow, padding=4)
        grid_np = TF.to_pil_image(grid)

        # 텍스트 덧붙이기
        draw = ImageDraw.Draw(grid_np)
        font = ImageFont.load_default()

        for idx, label in enumerate(labels):
            x = (idx % nrow) * (W + 4) + 2
            y = (idx // nrow) * (H + 4) + 2
            draw.text((x, y), label, font=font, fill=colors[idx])

        grid_tensor = TF.to_tensor(grid_np)
        self.tb_summary.add_image(tag, grid_tensor, step)
