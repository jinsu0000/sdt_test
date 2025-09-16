import torch
import torch.nn as nn
import torchvision.models as models
from models.transformer import *
from models.encoder import Content_TR
from models.vq import VectorQuantizerEMA
from einops import rearrange, repeat
from utils.logger import print_once, log_stats
from models.gmm import get_seq_from_gmm, get_seq_from_gmm_with_sigma

'''
the overall architecture of our style-disentangled Transformer (SDT).
the input of our SDT is the gray image with 1 channel.
'''
class SDT_Generator(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2, num_head_layers= 1,
                 wri_dec_layers=2, gly_dec_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True, return_intermediate_dec=True):
        super(SDT_Generator, self).__init__()
        ### style encoder with dual heads
        self.Feat_Encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(models.resnet18(pretrained=True).children())[1:-2]))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)
        writer_norm = nn.LayerNorm(d_model) if normalize_before else None
        glyph_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.writer_head = TransformerEncoder(encoder_layer, num_head_layers, writer_norm)
        self.glyph_head = TransformerEncoder(encoder_layer, num_head_layers, glyph_norm)

        ### content ecoder
        self.content_encoder = Content_TR(d_model, num_encoder_layers)

        ### decoder for receiving writer-wise and character-wise styles
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        wri_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.wri_decoder = TransformerDecoder(decoder_layer, wri_dec_layers, wri_decoder_norm,
                                              return_intermediate=return_intermediate_dec)
        gly_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.gly_decoder = TransformerDecoder(decoder_layer, gly_dec_layers, gly_decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        
        ### two mlps that project style features into the space where nce_loss is applied
        self.pro_mlp_writer = nn.Sequential(
            nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))
        self.pro_mlp_character = nn.Sequential(
            nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))

        # [VQ-PATCH] 하이퍼파라미터 (원하면 cfg로 빼세요)
        self.vq_dim   = 64
        self.vq_K     = 1024
        self.vq_beta  = 0.25
        self.proto_scale = 1.0   # r_t 크기 조절용(불안정하면 0.5로 시작)

        # [VQ-PATCH] VQ 토크나이저 + 프로젝션/프로토타입 헤드
        self.vq = VectorQuantizerEMA(K=self.vq_K, D=self.vq_dim, beta=self.vq_beta, decay=0.99)

        self.vq_proj = nn.Linear(d_model, self.vq_dim)

        self.proto_head = nn.Linear(self.vq_dim, 2)    # Δx_base, Δy_base


        self.SeqtoEmb = SeqtoEmb(hid_dim=d_model)
        self.EmbtoSeq = EmbtoSeq(hid_dim=d_model)
        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model)        
        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def random_double_sampling(self, x, ratio=0.25):
        """
        Sample the positive pair (i.e., o and o^+) within a character by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [L, B, N, D], sequence
        return o [B, N, 1, D], o^+ [B, N, 1, D]
        """
        L, B, N, D = x.shape  # length, batch, group_number, dim
        x = rearrange(x, "L B N D -> B N L D")
        noise = torch.rand(B, N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)

        anchor_tokens, pos_tokens = int(L*ratio), int(L*2*ratio)
        ids_keep_anchor, ids_keep_pos = ids_shuffle[:, :, :anchor_tokens], ids_shuffle[:, :, anchor_tokens:pos_tokens]
        x_anchor = torch.gather(
            x, dim=2, index=ids_keep_anchor.unsqueeze(-1).repeat(1, 1, 1, D))
        x_pos = torch.gather(
            x, dim=2, index=ids_keep_pos.unsqueeze(-1).repeat(1, 1, 1, D))
        return x_anchor, x_pos

    # the shape of style_imgs is [B, 2*N, C, H, W] during training
    def forward(self, style_imgs, seq, char_img):
        print_once("SDT_Generator::forward, style_imgs:", style_imgs.shape)
        batch_size, num_imgs, in_planes, h, w = style_imgs.shape
        print_once("SDT_Generator::forward, batch_size:", batch_size, ", num_imgs:", num_imgs, ", in_planes:", in_planes, ", h:", h, ", w:", w)
        
        # style_imgs: [B, 2*N, C:1, H, W] -> FEAT_ST_ENC: [4*N, B, C:512]
        style_imgs = style_imgs.view(-1, in_planes, h, w)  # [B*2N, C:1, H, W]
        print_once("SDT_Generator::forward style_imgs.view():", style_imgs.shape)
        style_embe = self.Feat_Encoder(style_imgs)  # [B*2N, C:512, 2, 2]
        print_once("SDT_Generator::forward Feat_Encoder_ResNet output:", style_embe.shape)  

        anchor_num = num_imgs//2
        style_embe = style_embe.view(batch_size*num_imgs, 512, -1).permute(2, 0, 1)  # [4, B*2N, C:512]
        print_once("SDT_Generator::forward style_embe.view():", style_embe.shape)
        FEAT_ST_ENC = self.add_position(style_embe)

        memory = self.base_encoder(FEAT_ST_ENC)  # [4, B*2N, C]
        print_once("SDT_Generator::forward base_encoder output memory:", memory.shape)
        writer_memory = self.writer_head(memory)
        glyph_memory = self.glyph_head(memory)
        print_once("SDT_Generator::forward WRITER memory:", writer_memory.shape, ", GLYPH memory:", glyph_memory.shape)

        writer_memory = rearrange(writer_memory, 't (b p n) c -> t (p b) n c',
                           b=batch_size, p=2, n=anchor_num)  # [4, 2*B, N, C]
        glyph_memory = rearrange(glyph_memory, 't (b p n) c -> t (p b) n c',
                           b=batch_size, p=2, n=anchor_num)  # [4, 2*B, N, C]
        print_once("SDT_Generator::forward rearrange writer_memory:", writer_memory.shape, ", glyph_memory:", glyph_memory.shape)

        # writer NCE: [4, 2B, N, C] -> [4N, 2B, C]
        memory_fea = rearrange(writer_memory, 't b n c ->(t n) b c')  # [4*N, 2*B, C]

        compact_fea = torch.mean(memory_fea, 0) # [2*B, C]
        print_once("SDT_Generator::forward [writer] memory_fea:", memory_fea.shape, ", compact_fea:", compact_fea.shape)
        # compact_fea:[2*B, C:512] ->  nce_emb: [B, 2, C:128]
        pro_emb = self.pro_mlp_writer(compact_fea)
        print_once("SDT_Generator::forward [writer] pro_emb:", pro_emb.shape)
        query_emb = pro_emb[:batch_size, :]
        print_once("SDT_Generator::forward [writer] query_emb:", query_emb.shape)
        pos_emb = pro_emb[batch_size:, :]
        print_once("SDT_Generator::forward [writer] pos_emb:", pos_emb.shape)
        nce_emb = torch.stack((query_emb, pos_emb), 1) # [B, 2, C]
        print_once("SDT_Generator::forward [writer] nce_emb:", nce_emb.shape)
        nce_emb = nn.functional.normalize(nce_emb, p=2, dim=2)
        print_once("SDT_Generator::forward [writer] normalize NCE nce_emb:", nce_emb.shape)

        # glyph-nce
        patch_emb = glyph_memory[:, :batch_size]  # [4, B, N, C]
        print_once("SDT_Generator::forward [glyph] patch_emb:", patch_emb.shape)
        # sample the positive pair
        anc, positive = self.random_double_sampling(patch_emb)
        n_channels = anc.shape[-1]
        print_once("SDT_Generator::forward [glyph] random_double_sampling result anc:", anc.shape, ", positive:", positive.shape, ", n_channels:", n_channels)
        anc = anc.reshape(batch_size, -1, n_channels)
        print_once("SDT_Generator::forward [glyph] anc reshape:", anc.shape)
        anc_compact = torch.mean(anc, 1, keepdim=True)
        print_once("SDT_Generator::forward [glyph] anc_compact:", anc_compact.shape)
        anc_compact = self.pro_mlp_character(anc_compact) # [B, 1, C]
        print_once("SDT_Generator::forward [glyph] anc_compact after pro_mlp_character:", anc_compact.shape)
        positive = positive.reshape(batch_size, -1, n_channels)
        print_once("SDT_Generator::forward [glyph] positive reshape:", positive.shape)
        positive_compact = torch.mean(positive, 1, keepdim=True)
        print_once("SDT_Generator::forward [glyph] positive_compact:", positive_compact.shape)
        positive_compact = self.pro_mlp_character(positive_compact) # [B, 1, C]

        print_once("SDT_Generator::forward [glyph] NCE anc_compact:", anc_compact.shape, ", positive_compact:", positive_compact.shape)
        nce_emb_patch = torch.cat((anc_compact, positive_compact), 1) # [B, 2, C]
        nce_emb_patch = nn.functional.normalize(nce_emb_patch, p=2, dim=2)
        print_once("SDT_Generator::forward [glyph] normalize nce_emb_patch:", nce_emb_patch.shape)

        # input the writer-wise & character-wise styles into the decoder
        writer_style = memory_fea[:, :batch_size, :]  # [4*N, B, C]
        print_once("SDT_Generator::forward [KV] writer_style:", writer_style.shape)
        glyph_style = glyph_memory[:, :batch_size]  # [4, B, N, C]
        print_once("SDT_Generator::forward [KV] glyph_style:", glyph_style.shape)
        glyph_style = rearrange(glyph_style, 't b n c -> (t n) b c') # [4*N, B, C]
        print_once("SDT_Generator::forward [KV] glyph_style rearranged:", glyph_style.shape)
        #print_once("SDT_Generator::forward glyph_style after rearrange:", glyph_style.shape)

        # QUERY: [char_emb, seq_emb]
        
        print_once(f"SDT_Generator::forward [Decoder] seq: [{seq.shape[0]}, T, {seq.shape[2]}] (T : length of the sequence)")
        # seq: [T, B, 5], T is the length of the sequence
        seq_emb = self.SeqtoEmb(seq).permute(1, 0, 2)
        T, N, C = seq_emb.shape
        print_once(f"SDT_Generator::forward [Decoder] seq_emb: [T, {N}, {C}]")

        char_emb = self.content_encoder(char_img) # [4, N, 512]
        print_once(f"SDT_Generator::forward [Content Encoder] char_emb: [4, T, {char_emb.shape[1]}]")
        char_emb = torch.mean(char_emb, 0) #[N, 512]
        print_once(f"SDT_Generator::forward [Content Encoder] char_emb after mean: [T, {char_emb.shape[1]}]")
        char_emb = repeat(char_emb, 'n c -> t n c', t = 1)
        print_once(f"SDT_Generator::forward [Content Encoder] char_emb after repeat: [1, T, {char_emb.shape[1]}]")
        tgt = torch.cat((char_emb, seq_emb), 0) # [1+T], put the content token as the first token
        print_once(f"SDT_Generator::forward [Decoder] tgt: [T, {tgt.shape[1]}, {tgt.shape[2]}] (T=T+1 : length of the sequence + 1 for content token)")
        tgt_mask = generate_square_subsequent_mask(sz=(T+1)).to(tgt)
        tgt = self.add_position(tgt)
        print_once(f"SDT_Generator::forward [Decoder] tgt after masking & add_position: [1+T, {tgt.shape[1]}, {tgt.shape[2]}]")

        # [wri_dec_layers, T, B, C]
        wri_hs = self.wri_decoder(tgt, writer_style, tgt_mask=tgt_mask)
        print_once(f"SDT_Generator::forward [Writer Decoder] wri_hs: {wri_hs.shape[0]}, T, {wri_hs.shape[2]}, {wri_hs.shape[3]}] (wri_dec_layers, T, B, C)")
        # [gly_dec_layers, T, B, C]
        hs = self.gly_decoder(wri_hs[-1], glyph_style, tgt_mask=tgt_mask) 
        print_once(f"SDT_Generator::forward [Glyph Decoder] hs: [{hs.shape[0]}, T, {hs.shape[2]}, {hs.shape[3]}] (gly_dec_layers, T, B, C)")

        h = hs.transpose(1, 2)[-1]  # B T C
        print_once(f"SDT_Generator::forward [Decoder] h after transpose: {h.shape[0]}, T, {h.shape[2]}] (B, T, C)")

        # [VQ-PATCH] 디코더 히든 -> VQ 임베딩 -> 코드북 양자화 -> 프로토타입 r_t
        z_e = self.vq_proj(h)                                 # [B,T,vq_dim]
        z_q, vq_loss, vq_codes = self.vq(z_e)                 # [B,T,vq_dim], scalar, [B,T]
        r_t = self.proto_head(z_q) * self.proto_scale         # [B,T,2]  (Δx_base, Δy_base)

        pred_sequence = self.EmbtoSeq(h)                      # [B,T,123] (기존)
        # [VQ-PATCH] pred_sequence를 재조립: μ에 r_t를 더해 "프로토타입+잔차"
        B, T, Ctot = pred_sequence.shape
        assert (Ctot - 3) % 6 == 0, "GMM 파라미터 차원이 3 + 6*M 형태여야 합니다."
        M = (Ctot - 3) // 6

        pred = pred_sequence.view(B, T, 3 + 6*M)
        pen_logits = pred[:, :, :3]                     # [B,T,3]
        mix = pred[:, :, 3:].view(B, T, 6, M)           # [B,T,6,M]
        pi, mu1_r, mu2_r, s1, s2, corr = mix[:, :, 0], mix[:, :, 1], mix[:, :, 2], mix[:, :, 3], mix[:, :, 4], mix[:, :, 5]

        # r_t(Δx_base, Δy_base)를  M개의 mixture로 broadcast
        rx = r_t[..., 0].unsqueeze(-1).expand_as(mu1_r) # [B,T,M]
        ry = r_t[..., 1].unsqueeze(-1).expand_as(mu2_r) # [B,T,M]
        mu1 = mu1_r + rx
        mu2 = mu2_r + ry

        out = torch.cat([
            pen_logits,
            torch.stack([pi, mu1, mu2, s1, s2, corr], dim=2).reshape(B, T, 6*M)
        ], dim=-1).contiguous()                          # [B,T,3+6*M] == [B,T,123]
        pred_sequence = out

        print_once(f"SDT_Generator::forward [Decoder Output] pred_sequence: [{pred_sequence.shape[0]}, T, {pred_sequence.shape[2]}] (B, T, C)")

        # [VQ-PATCH] vq_loss / codes를 로깅용으로 보낼 수 있게 추가 반환
        return pred_sequence, nce_emb, nce_emb_patch, vq_loss, vq_codes

    # style_imgs: [B, N, C, H, W]
    def inference(self, style_imgs, char_img, max_len):
        print_once("SDT_Generator::inference style_imgs:", style_imgs.shape)
        batch_size, num_imgs, in_planes, h, w = style_imgs.shape
        # [B, N, C, H, W] -> [B*N, C, H, W]
        style_imgs = style_imgs.view(-1, in_planes, h, w)
        print_once("SDT_Generator::inference style_imgs.view():", style_imgs.shape)
        # [B*N, 1, 64, 64] -> [B*N, 512, 2, 2]
        style_embe = self.Feat_Encoder(style_imgs)
        print_once("SDT_Generator::inference Feat_Encoder_ResNet output:", style_embe.shape)
        FEAT_ST = style_embe.reshape(batch_size*num_imgs, 512, -1).permute(2, 0, 1)  # [4, B*N, C]
        print_once("SDT_Generator::inference FEAT_ST:", FEAT_ST.shape)
        FEAT_ST_ENC = self.add_position(FEAT_ST)  # [4, B*N, C:512]
        memory = self.base_encoder(FEAT_ST_ENC)  # [5, B*N, C]
        memory_writer = self.writer_head(memory)  # [4, B*N, C]
        memory_glyph = self.glyph_head(memory)  # [4, B*N, C]
        memory_writer = rearrange(
            memory_writer, 't (b n) c ->(t n) b c', b=batch_size)  # [4*N, B, C]
        memory_glyph = rearrange(
            memory_glyph, 't (b n) c -> (t n) b c', b=batch_size)  # [4*N, B, C]

        char_emb = self.content_encoder(char_img)
        char_emb = torch.mean(char_emb, 0) #[N, 256]
        src_tensor = torch.zeros(max_len + 1, batch_size, 512).to(char_emb)
        pred_sequence = torch.zeros(max_len, batch_size, 5).to(char_emb)
        src_tensor[0] = char_emb
        tgt_mask = generate_square_subsequent_mask(sz=max_len + 1).to(char_emb)
        for i in range(max_len):
            src_tensor[i] = self.add_position(src_tensor[i], step=i)

            wri_hs = self.wri_decoder(
                src_tensor, memory_writer, tgt_mask=tgt_mask)
            hs = self.gly_decoder(wri_hs[-1], memory_glyph, tgt_mask=tgt_mask)

            output_hid = hs[-1][i]
            gmm_pred = self.EmbtoSeq(output_hid)
            pred_sequence[i] = get_seq_from_gmm(gmm_pred)
            pen_state = pred_sequence[i, :, 2:]
            seq_emb = self.SeqtoEmb(pred_sequence[i])
            src_tensor[i + 1] = seq_emb
            if sum(pen_state[:, -1]) == batch_size:
                break
            else:
                pass
        return pred_sequence.transpose(0, 1)  # N, T, C        

'''
project the handwriting sequences to the transformer hidden space
'''
class SeqtoEmb(nn.Module):
    def __init__(self, hid_dim, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(5, 256)
        self.fc_2 = nn.Linear(256, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.dropout(torch.relu(self.fc_1(seq)))
        x = self.fc_2(x)
        return x

'''
project the transformer hidden space to handwriting sequences
'''
class EmbtoSeq(nn.Module):
    def __init__(self, hid_dim, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 123)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        x = self.dropout(torch.relu(self.fc_1(seq)))
        x = self.fc_2(x)
        return x


''' 
generate the attention mask, i.e. [[0, inf, inf],
                                   [0, 0, inf],
                                   [0, 0, 0]].
The masked positions are filled with float('-inf').
Unmasked positions are filled with float(0.0).                                     
'''
def generate_square_subsequent_mask(sz: int) -> Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    print_once("generate_square_subsequent_mask:", mask.shape)
    return mask