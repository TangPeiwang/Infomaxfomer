import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer, series_decomp, moving_avg
from models.attn import FullAttention, SparseAttention, AttentionLayer
from models.embed import DataEmbedding


class Infomaxformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='sparse', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), avg=25, constant=30):
        super(Infomaxformer, self).__init__()
        self.pred_len = out_len
        self.seq_len = seq_len
        self.label_len = label_len
        self.attn = attn
        self.device = device
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = SparseAttention if attn == 'sparse' else FullAttention

        self.avg = avg
        self.decomp = series_decomp(avg)
        self.decomp1 = series_decomp(avg)
        self.moving_avg = moving_avg(avg, 1)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, constant=constant, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False, seq_len=self.seq_len),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, constant=constant, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix, seq_len=self.seq_len),
                    AttentionLayer(Attn(False, factor, constant=constant, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False, seq_len=self.seq_len),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=avg,
                    dropout=dropout,
                    activation=activation,
                )for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # encoder
        trend_init, seasonal_init = self.decomp(x_enc)
        enc_out = self.enc_embedding(trend_init, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        trend_part = self.projection(trend_part[:, -self.pred_len:, :])

        # final
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # mean = trend_init[:, -self.pred_len-self.avg:-self.avg, :]
        dec_out = trend_part + mean

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :] , attns[:, -self.pred_len:, :]
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

