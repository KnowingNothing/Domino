import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
import argparse
import torch.onnx as tonnx


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.w_2(self.activation(self.w_1(x)))) + x


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # p_attn = F.softmax(scores, dim=-1)
        p_attn = scores

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)]
        )
        # self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        inputs = query
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # x = query + key + value

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return x + inputs


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()

        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        # self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        # x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        # x = self.output_sublayer(x, self.feed_forward)
        # return self.dropout(x)
        x = self.attention(x, x, x)
        x = self.feed_forward(x)
        return x


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(
        self, vocab_size=None, hidden=768, n_layers=12, attn_heads=12, dropout=0.1
    ):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = min(hidden * 4, 3072)

        # embedding for BERT, sum of positional, segment, token embeddings
        # self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, segment_info=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        # x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)

        return x


def main(batch, seq_len, hidden, n_heads, layers, dtype, only_once=False):
    in_dtype = dtype

    repeat = 600
    inputs_np = np.random.uniform(-1, 1, [batch, seq_len, hidden]).astype(in_dtype)

    model = BERT(hidden=hidden, n_layers=layers, attn_heads=n_heads)
    if dtype == "float16":
        model = model.half()
    model = model.cuda()
    if only_once:
        inputs_torch = torch.tensor(inputs_np).cuda()
        output = model(inputs_torch)
        cost = -1
    else:
        inputs_torch = [torch.tensor(inputs_np).cuda() for i in range(repeat)]
        # measure time
        # warm up
        for i in range(repeat):
            output = model(inputs_torch[i])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        # beg = time.time()
        start.record()
        for i in range(repeat):
            output = model(inputs_torch[i])
        end.record()
        torch.cuda.synchronize()
        # stop = time.time()
        total = start.elapsed_time(end)
        cost = total / repeat
        # print(f"Average time cost is {cost} ms.")
    return cost


def export_model(batch, seq_len, hidden, n_heads, layers, dtype, path):
    in_dtype = dtype

    inputs_np = np.random.uniform(-1, 1, [batch, seq_len, hidden]).astype(in_dtype)

    model = BERT(hidden=hidden, n_layers=layers, attn_heads=n_heads)
    if dtype == "float16":
        model = model.half()
    model = model.cuda()
    inputs_torch = torch.tensor(inputs_np).cuda()
    output = model(inputs_torch)
    tonnx.export(
        model,
        inputs_torch,
        path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


example_text = """
 example:
    python bert_inference_cuda.py --dtype float32 --begin 0 --num 1
"""


def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)


shapes = [
    # (batch, seq_len, hidden, n_heads, n_layers)
    # (1, 1024, 512, 8, 4),  # Transformer-Small
    # (1, 1024, 768, 12, 12),  # Transformer-Base
    # (1, 1024, 1024, 16, 24),  # Transformer-Large
    # (1, 512, 512, 8, 4),  # Bert-Small
    (1, 512, 768, 12, 12),  # Bert-Base
    # (1, 512, 1024, 16, 24),  # Bert-Large
    # (1, 256, 768, 12, 12),  # ViT-Base/14
    # (1, 256, 1024, 16, 24),  # ViT-Large/14
    # (1, 256, 1280, 16, 24),  # ViT-Huge/14
    # (1, 196, 768, 12, 12),  # ViT-Base/16
    # (1, 196, 1024, 16, 24),  # ViT-Large/16
    # (1, 196, 1280, 16, 24),  # ViT-Huge/16
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only_once", action="store_true")
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "int8"],
        default="float16",
    )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False
    # costs = []
    # for ss in shapes[args.begin : args.begin + args.num]:
    #     batch, seq_len, hidden, n_heads, n_layers = ss
    #     cost = main(
    #         batch, seq_len, hidden, n_heads, n_layers, args.dtype, args.only_once
    #     )
    #     costs.append((ss, cost))
    # print("batch,seq_len,hidden,n_heads,n_layers,dtype,cost")
    # for cc in costs:
    #     print(
    #         f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{args.dtype},{cc[1]}"
    #     )
    for ss in shapes[args.begin : args.begin + args.num]:
        batch, seq_len, hidden, n_heads, n_layers = ss
        export_model(
            batch, seq_len, hidden, n_heads, n_layers, args.dtype, "simplified_bert_base.onnx"
        )