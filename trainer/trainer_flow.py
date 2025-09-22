import time, os, datetime, sys, torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.util import coords_render
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont

class TrainerFlow:
    """
    - DataParallel 안전 호출(_call)
    - TensorBoard 경로 출력 + 매 스텝 flush
    - 50스텝마다 [문자 | Pred | GT] 3패널 이미지
    - 체크포인트 저장 시 .module 안전 처리
    """
    def __init__(self, model_flow, optimizer, data_loader, logs, char_dict, valid_data_loader=None):
        self.model_flow = model_flow
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.char_dict = char_dict

        self.tb_summary = SummaryWriter(logs['tboard'])
        print(f"[TB] writing to: {logs['tboard']}")
        os.makedirs(logs['tboard'], exist_ok=True)

        self.save_model_dir = logs['model']
        self.save_sample_dir = logs['sample']

    def _call(self, fn, *args, **kwargs):
        model = self.model_flow.module if isinstance(self.model_flow, torch.nn.DataParallel) else self.model_flow
        return getattr(model, fn)(*args, **kwargs)

    def _progress(self, step, loss, time_left):
        sys.stdout.write(f"iter:{step} loss:{loss:.3f} ETA:{str(time_left)}\r\n")

    def _save_checkpoint(self, step):
        os.makedirs(self.save_model_dir, exist_ok=True)
        path = f"{self.save_model_dir}/checkpoint-flow-iter{step}.pth"
        model = self.model_flow.module if isinstance(self.model_flow, torch.nn.DataParallel) else self.model_flow
        torch.save({"model": model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "step": step}, path)

    @torch.no_grad()
    def _tb_samples(self, gt_coords, preds, character_id, char_img_batch, step, img_w=64, img_h=64):
        img_tensors = []; batch_idx = 0; font = ImageFont.load_default()
        for i, _ in enumerate(gt_coords):
            gt_img = coords_render(gt_coords[i], split=True, width=img_w, height=img_h, thickness=1)
            pred_img = coords_render(preds[i],   split=True, width=img_w, height=img_h, thickness=1)
            char_img_np = (char_img_batch[i].cpu().numpy().squeeze() * 255).astype('uint8')
            char_img_pil = Image.fromarray(char_img_np).convert("RGB").resize((img_w, img_h))
            canvas = Image.new("RGB", (img_w*3, img_h + 12), (255,255,255))
            canvas.paste(char_img_pil, (0,0)); canvas.paste(pred_img, (img_w,0)); canvas.paste(gt_img, (img_w*2,0))
            character = self.char_dict[character_id[i].item()]
            ImageDraw.Draw(canvas).text((2, img_h), character, fill=(0,0,0), font=font)
            os.makedirs(self.save_sample_dir, exist_ok=True)
            path = os.path.join(self.save_sample_dir, f'{character}_{step}_.jpg')
            try: canvas.save(path)
            except Exception as e: print(f"[Save Error] {path}: {e}")
            try: img_tensors.append(ToTensor()(canvas))
            except Exception as e: print(f"[ToTensor Error] {path}: {e}")
            if len(img_tensors) == 10:
                try:
                    grid = make_grid(img_tensors, nrow=5, padding=4)
                    self.tb_summary.add_image(f"Samples/Step_{step}_batch{batch_idx}", grid, step)
                    self.tb_summary.flush()
                    batch_idx += 1
                except Exception as e:
                    print(f"[TB Image Error] step {step} batch {batch_idx}: {e}")
                img_tensors = []
        if img_tensors:
            try:
                grid = make_grid(img_tensors, nrow=5, padding=4)
                self.tb_summary.add_image(f"Samples/Step_{step}_batch{batch_idx}", grid, step)
                self.tb_summary.flush()
            except Exception as e:
                print(f"[TB Final Image Error] step {step} batch {batch_idx}: {e}")

    def train(self, max_iter):
        train_iter = iter(self.data_loader)
        for step in range(max_iter):
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(self.data_loader); data = next(train_iter)

            coords = data['coords'].cuda()
            character_id = data['character_id'].long().cuda()
            writer_id = data['writer_id'].long().cuda()
            img_list = data['img_list'].cuda()
            char_img = data['char_img'].cuda()

            t0 = time.time()
            self.model_flow.train()

            loss_flow, loss_pen = self._call('flow_match_loss', img_list, coords, char_img)
            loss = loss_flow + loss_pen

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_flow.parameters(), 1.0)
            self.optimizer.step()

            losses = {"flow": float(loss_flow.detach().cpu()),
                      "pen":  float(loss_pen.detach().cpu()),
                      "total": float(loss.detach().cpu())}
            self.tb_summary.add_scalars("loss_flow", losses, step)
            self.tb_summary.flush()

            diag = self._call('get_diag_attn')  # DataParallel-safe
            if diag is not None:
                if diag.get("style") is not None:
                    self.tb_summary.add_scalar("diag/attn_style", float(diag["style"]), step)
                if diag.get("content") is not None:
                    self.tb_summary.add_scalar("diag/attn_content", float(diag["content"]), step)
                if diag.get("xattn") is not None:
                    self.tb_summary.add_scalar("diag/attn_xattn", float(diag["xattn"]), step)
                self.tb_summary.flush()

            dt = time.time() - t0
            left = datetime.timedelta(seconds=int((max_iter - step) * max(dt, 1e-3)))
            self._progress(step, losses["total"], left)

            if (step+1) % 100 == 0:
                self.model_flow.eval()
                with torch.no_grad():
                    B, T, _ = coords.shape
                    delta = self._call('flow_infer', img_list[:1], char_img[:1], T=min(120, T))
                    pred = torch.zeros(1, delta.size(1), 5, device=delta.device)
                    pred[..., :2] = delta
                    self._tb_samples(coords[:1].cpu().numpy(), pred.cpu().numpy(),
                                     character_id[:1].cpu(), char_img[:1].cpu(), step)

            if (step+1) % 10000 == 0:
                self._save_checkpoint(step)
