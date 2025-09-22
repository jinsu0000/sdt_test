
# SDT ↔ Flow Matching Integration (Action Chunking, π₀-style)

This folder adds Flow Matching on top of your existing **SDT_Generator** without modifying its style encoders.
- We **reuse Writer/Glyph encoders** inside `SDT_Generator` to build **context prefix tokens** (Writer, Glyph).
- A π₀-style **FlowPolicy** predicts a **vector field** over action chunks (Δx,Δy), with optional pen logits.
- A minimal **TrainerFlow** mirrors your TensorBoard image logging style.

## Files
- `flow_policy.py` — shared self-attn + expert FFN, `ActionEmbed`, `FlowPolicy`
- `sdt_flow_wrapper.py` — `SDT_FlowWrapper` to extract style tokens from SDT and run Flow Matching
- `trainer_flow.py` — Trainer variant for Flow Matching (keeps your TB image layout)
- (this) README

## How to wire
1) Construct your original SDT model as usual (to keep NCE paths available).
2) Wrap it:
```python
from sdt_flow_wrapper import SDT_FlowWrapper
flow_model = SDT_FlowWrapper(sdt_model, H=64, n_layers=6, n_head=8, ffn_mult=4)
opt = torch.optim.AdamW(flow_model.parameters(), lr=2e-4, weight_decay=1e-4)
```
3) Use `TrainerFlow` with your existing dataloader/char_dict/log dirs:
```python
from trainer_flow import TrainerFlow
trainer = TrainerFlow(flow_model, opt, train_loader, logs, char_dict, valid_loader)
trainer.train(max_iter=cfg.SOLVER.MAX_ITER)
```
4) To keep your **NCE** losses/plots exactly as-is, still run your original `Trainer` in AR mode for NCE,
   or add those terms in `TrainerFlow` where marked.

## Notes
- Context prefix width is `512` to match your SDT `d_model`. If different, change `self.d_model` in wrapper.
- Sliding windows / overlap blending can be added easily; this wrapper currently trains on the **first H window** per batch for simplicity.
- The TB image routine uses the same canvas layout as your original trainer.

Enjoy!
