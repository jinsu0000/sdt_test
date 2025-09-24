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
    - 100스텝마다 [문자 | Pred | GT] 3패널 이미지
    - 체크포인트 저장 시 .module 안전 처리
    - 콘솔 프린트 안전(float 캐스팅 + flush)
    - NCE(Writer/Glyph) 가중치: warm→cosine decay 스케줄
    """
    def __init__(self, model_flow, optimizer, data_loader, logs, char_dict,
                 valid_data_loader=None,
                 # 기본 가중치(미사용 시 0으로 둬도 됨)
                 nce_writer_weight=0.1, nce_glyph_weight=0.1, nce_temp=0.07,
                 # 스케줄 파라미터
                 nce_w_writer_init=0.5, nce_w_glyph_init=0.5,
                 nce_w_writer_final=0.1, nce_w_glyph_final=0.1,
                 nce_warm_steps=5000, nce_decay_steps=15000):
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

        # base weights (옵션)
        self.w_writer_base = nce_writer_weight
        self.w_glyph_base  = nce_glyph_weight
        self.nce_temp = nce_temp

        # schedule params
        self.nce_w_writer_init  = nce_w_writer_init
        self.nce_w_glyph_init   = nce_w_glyph_init
        self.nce_w_writer_final = nce_w_writer_final
        self.nce_w_glyph_final  = nce_w_glyph_final
        self.nce_warm_steps     = nce_warm_steps
        self.nce_decay_steps    = nce_decay_steps

        # 모델 내부 SupCon 온도 동기화
        model = self.model_flow.module if isinstance(self.model_flow, torch.nn.DataParallel) else self.model_flow
        if hasattr(model, 'supcon'):
            model.supcon.temperature = self.nce_temp

    # ---- utils ----
    def _call(self, fn, *args, **kwargs):
        model = self.model_flow.module if isinstance(self.model_flow, torch.nn.DataParallel) else self.model_flow
        return getattr(model, fn)(*args, **kwargs)

    def _progress(self, step, loss, time_left, extra=None):
        try:
            loss_val = float(loss) if not hasattr(loss, "item") else float(loss.item())
        except Exception:
            loss_val = float(loss)
        line = f"iter:{step} loss:{loss_val:.3f} ETA:{str(time_left)}"
        if extra:
            try:
                extras = " ".join([f"{k}:{float(v if not hasattr(v,'item') else v.item()):.3f}" for k, v in extra.items()])
                line += " " + extras
            except Exception:
                pass
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

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
            from PIL import Image as _Image
            char_img_pil = _Image.fromarray(char_img_np).convert("RGB").resize((img_w, img_h))
            canvas = _Image.new("RGB", (img_w*3, img_h + 12), (255,255,255))
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

    # ---- NCE weight scheduler (warm -> cosine decay -> final) ----
    def _sched(self, step, w_init, w_final):
        if step < self.nce_warm_steps:
            return float(w_init)
        # t in [0,1]
        t = min(1.0, max(0.0, (step - self.nce_warm_steps) / max(1, self.nce_decay_steps)))
        # cosine from init to final
        import math
        return float(w_final + 0.5*(w_init - w_final)*(1 + math.cos(math.pi * t)))

    # ---- train loop ----
    def train(self, max_iter):
        train_iter = iter(self.data_loader)
        for step in range(max_iter):
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(self.data_loader); data = next(train_iter)

            coords = data['coords'].cuda()
            character_id = data['character_id'].long().cuda()
            writer_id = data['writer_id'].long().cuda()   # (현재는 사용 X, 필요시 라벨로 활용)
            img_list = data['img_list'].cuda()
            char_img = data['char_img'].cuda()

            t0 = time.time()
            self.model_flow.train()

            # 유동 NCE 가중치
            wW = self._sched(step, self.nce_w_writer_init, self.nce_w_writer_final)
            wG = self._sched(step, self.nce_w_glyph_init,  self.nce_w_glyph_final)

            # ---- Flow ----
            loss_flow, loss_pen = self._call('flow_match_loss', img_list, coords, char_img)
            loss = loss_flow + loss_pen

            # ---- NCE (SupCon, 학습 포함) ----
            writer_supcon, glyph_supcon = self._call('nce_losses_supcon', img_list)
            loss = loss + wW * writer_supcon + wG * glyph_supcon

            # ---- Optimize ----
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_flow.parameters(), 1.0)
            self.optimizer.step()

            # ---- TensorBoard ----
            self.tb_summary.add_scalars("loss_flow", {
                "flow":  float(loss_flow.detach().cpu()),
                "pen":   float(loss_pen.detach().cpu()),
                "total": float(loss.detach().cpu()),
            }, step)
            self.tb_summary.add_scalars("loss_nce", {
                "writer_supcon": float(writer_supcon.detach().cpu()),
                "glyph_supcon":  float(glyph_supcon.detach().cpu()),
            }, step)
            self.tb_summary.add_scalars("nce/weights", {
                "writer_w": wW,
                "glyph_w":  wG,
            }, step)

            # 어텐션 다이애그
            diag = self._call('get_diag_attn')
            if diag is not None:
                if diag.get("style") is not None:
                    self.tb_summary.add_scalar("diag/attn_style", float(diag["style"]), step)
                if diag.get("content") is not None:
                    self.tb_summary.add_scalar("diag/attn_content", float(diag["content"]), step)
                if diag.get("xattn") is not None:
                    self.tb_summary.add_scalar("diag/attn_xattn", float(diag["xattn"]), step)
            self.tb_summary.flush()

            # ---- Console ----
            dt = time.time() - t0
            left = datetime.timedelta(seconds=int((max_iter - step) * max(dt, 1e-3)))
            if (step % 10) == 0:
                extra = {"flow": loss_flow.detach(), "pen": loss_pen.detach(),
                         "wNCE": writer_supcon.detach(), "gNCE": glyph_supcon.detach(),
                         "wW": wW, "wG": wG}
                self._progress(step, loss.detach(), left, extra=extra)
            else:
                self._progress(step, loss.detach(), left)

            # ---- Samples & Checkpoint ----
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
