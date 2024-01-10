"""
This code is based on huggingface,
https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py

MIT License

Copyright (c) 2018 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OFS CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
# Arxiv Link https://arxiv.org/pdf/1907.00235.pdf


import numpy as np
import torch
import torch.nn as nn
import math
import copy
from torch.nn.parameter import Parameter
from typing import Dict
from math import sqrt
from ContrastiveModules import ContrastiveModel, pn_rul_compute

import torch


class PositionEmbedding(nn.Module):
    def __init__(self, dim, window_size, dropout=0.5, device="cuda:0"):
        super(PositionEmbedding, self).__init__()
        self.pe = torch.zeros(window_size, dim)
        position = torch.arange(0, window_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                             -(np.log(10000.0) / dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU(),
    'swish': swish,
    'gelu': gelu
}


class LogSparseAttention(nn.Module):
    """
    Args:
        n_time_series: Number of time series present in input
        n_head: Number of heads in the MultiHeadAttention mechanism
        seq_num: The number of targets to forecast
        sub_len: sub_len of the sparse attention
        num_layer: The number of transformer blocks in the model.
        n_embd: The dimention of Position embedding and time series ID embedding
        forecast_history: The number of historical steps fed into the time series model
        dropout: The dropout for the embedding of the model.
        additional_params: Additional parameters used to initalize the attention model. Can inc
    """

    def __init__(self, n_head, n_embd, win_len, scale: bool, q_len: int, sub_len, sparse=True, attn_pdrop=0.1,
                 resid_pdrop=0.1):
        super(LogSparseAttention, self).__init__()

        if sparse:
            print('Activate log sparse!')
            mask = self.log_mask(win_len, sub_len)
        else:
            mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)

        self.register_buffer('mask_tri', mask)
        self.n_head = n_head
        self.split_size = n_embd * self.n_head
        self.scale = scale
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd * n_head * 2, self.q_len)
        self.value = Conv1D(n_embd * n_head, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_embd * self.n_head)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.

        2 . Our default setting here use Local attention and Restart attention.

        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros((win_len), dtype=torch.float)
        if ((win_len // sub_len) * 2 * (log_l) > index):
            mask[:(index + 1)] = 1
        else:
            while (index >= 0):
                if ((index - log_l + 1) < 0):
                    mask[:index] = 1
                    break
                mask[index - log_l + 1:(index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2 ** i
                    if ((index - new_index) <= sub_len and new_index >= 0):
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def attn(self, query: torch.Tensor, key, value: torch.Tensor):
        activation = nn.Softmax(dim=-1)
        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = activation(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return attn

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):

        value = self.value(x)
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len - 1, 0))
        query_key = self.query_key(qk_x).permute(0, 2, 1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn = self.attn(query, key, value)
        attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)
        return attn


class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.out_dim,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        return context.transpose(2, 1).contiguous(), attn


class ProbAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(ProbAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = ProbAttention(True, attention_dropout=0.1, output_attention=False)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, L, H, -1)
        values = self.value_projection(x).view(B, L, H, -1)

        out, _ = self.inner_attention(
            queries,
            keys,
            values
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)


class Encoder(nn.Module):
    def __init__(self, window_size, hidden_dim, attention):
        super(Encoder, self).__init__()
        self.attention = attention
        self.window_size = window_size
        self.ln1 = nn.LayerNorm(normalized_shape=hidden_dim)
        self.ln2 = nn.LayerNorm(normalized_shape=hidden_dim)
        self.ffl = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        )

    def forward(self, x):
        # x.shape = (b, w, h)
        att_x = self.attention(x)
        att_x = self.ln1(att_x + x)
        f_x = self.ln2(self.ffl(att_x) + att_x)
        return f_x


class IMDSSN(ContrastiveModel):
    def __init__(self,
                 window_size,
                 in_features,
                 hidden_dim,
                 encoder_nums,
                 n_heads,
                 pe=True,
                 label_norm=True, model_flag="IMDSSN", device="cuda:0", filter_size=0):
        super(IMDSSN, self).__init__(label_norm=label_norm, model_flag=model_flag, device=device)
        if filter_size > 0:
            self.window_size = window_size // filter_size
            self.MaV = nn.AvgPool1d(kernel_size=filter_size, stride=filter_size)
        else:
            self.window_size = window_size
            self.MaV = None

        self.input_mapper = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.pe = PositionEmbedding(dim=hidden_dim,
                                    window_size=self.window_size,
                                    dropout=0,
                                    device=device) if pe else None
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.MLSNEncoders = nn.Sequential()
        self.MPSNEncoders = nn.Sequential()
        for _ in range(encoder_nums):
            self.MLSNEncoders.append(Encoder(
                self.window_size, hidden_dim, attention=LogSparseAttention(
                    n_head=n_heads,
                    n_embd=hidden_dim,
                    win_len=self.window_size,
                    q_len=5,
                    sub_len=10,
                    scale=True,
                )
            ))
            self.MPSNEncoders.append(Encoder(
                self.window_size, hidden_dim, attention=ProbAttentionLayer(
                    n_heads=n_heads,
                    d_model=hidden_dim
                )
            ))
        self.fuse = nn.Linear(in_features=hidden_dim * 2,
                              out_features=hidden_dim,
                              bias=False)
        self.output = nn.Sequential(
            nn.Linear(in_features=self.window_size * hidden_dim, out_features=1)
        )
        self.to(device)

    def feature_extractor(self, x):
        # x.shape = (b, w, f)
        if self.MaV:
            x = self.MaV(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.input_mapper(x)
        x = self.pe(x) if self.pe else x
        f1 = self.MLSNEncoders(x)  # (b, w, h)
        f2 = self.MPSNEncoders(x)  # (b, w, h)
        f = torch.concat([f1, f2], dim=-1)  # (b, w, 2*h)
        f = self.fuse(f)  # (b, w, h)
        return torch.flatten(f, start_dim=-2, end_dim=-1)

    def forward(self, x, label=None):
        if len(x.shape) < 4:  # the normal forward, default shape with (b, l, f)
            x = self.feature_extractor(x)
            return self.output(x)
        else:  # the forward with negative samples, default shape with (b, num, l, f)
            f_pos, f_apos, f_neg, w = self.generate_contrastive_samples(x, label)
            return pn_rul_compute(self.output, f_pos, f_neg), f_pos, f_apos, f_neg, w


if __name__ == '__main__':
    log_att = LogSparseAttention(n_head=2, n_embd=512, win_len=30, scale=True, q_len=5, sub_len=10, sparse=True)
    pro_att = ProbAttentionLayer(
        d_model=512, n_heads=2
    )
    encoder = Encoder(30, 512, pro_att)
    net = IMDSSN(window_size=30, in_features=14, hidden_dim=512, encoder_nums=1,
                 n_heads=2, device="cpu", pe=True)
    inp = torch.randn(2, 30, 512)
    pro_out = pro_att(inp)
    log_out = log_att(inp)
    enc_out = encoder(inp)
    inp = torch.randn(2, 30, 14)
    lab = torch.randn(2, 5)
    net_out = net(inp)
