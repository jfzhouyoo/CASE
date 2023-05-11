### TAKEN FROM https://github.com/kolloldas/torchnlp
import os
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F

import numpy as np
import math
from src.models.common import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    NoamOpt,
    _get_attn_subsequent_mask,
    get_input_from_batch,
    get_output_from_batch,
    top_k_top_p_filtering,
    MultiHeadAttention,
    Embeddings,
    Attention
)
from src.utils.config import config
from src.utils.constants import ESC_MAP_EMO, ESC_MAP_STRATEGY, ED_MAP_EMO

from sklearn.metrics import accuracy_score

class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask, cs_enc_outputs, cs_enc_mask, concept_enc_outputs, concept_enc_mask):
        src_mask, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (src_mask, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (src_mask, dec_mask), cs_enc_outputs, cs_enc_mask, concept_enc_outputs, concept_enc_mask))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )  ## extend for all seq
            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  ## extend for all seq
            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            )
            return logit
        else:
            return F.log_softmax(logit, dim=-1)

class RelationMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, weights_dropout=True):
        super(RelationMultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == self.emb_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.query_linear = nn.Linear(emb_dim, emb_dim)
        self.key_linear = nn.Linear(emb_dim, emb_dim)
        self.value_linear = nn.Linear(emb_dim, emb_dim)
        self.relation_linear = nn.Linear(emb_dim, 2*emb_dim, bias=False)
        
        self.output_linear = nn.Linear(emb_dim, emb_dim)
        self.weights_dropout = weights_dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.query_linear.weight, std=0.02)
        nn.init.normal_(self.key_linear.weight, std=0.02)
        nn.init.normal_(self.value_linear.weight, std=0.02)
        nn.init.normal_(self.relation_linear.weight, std=0.02)
        nn.init.normal_(self.output_linear.weight, std=0.02)
        nn.init.constant_(self.query_linear.bias, 0.)
        nn.init.constant_(self.key_linear.bias, 0.)
        nn.init.constant_(self.value_linear.bias, 0.)
        nn.init.constant_(self.output_linear.bias, 0.)
    
    def forward(self, queries, keys, values, adj_mask, relation):
        bsz, tgt_len, emb_dim = queries.size()
        src_len = keys.size(1)
        
        queries = self.query_linear(queries) * self.scaling
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        
        queries = queries.view(bsz, tgt_len, self.num_heads, self.head_dim)
        keys = keys.view(bsz, src_len, self.num_heads, self.head_dim)
        values = values.view(bsz, src_len, self.num_heads, self.head_dim)
        
        relation_queries, relation_keys = self.relation_linear(relation).chunk(2, dim=-1)
        relation_queries = relation_queries.view(bsz, tgt_len, src_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        relation_keys = relation_keys.view(bsz, tgt_len, src_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        queries = queries.unsqueeze(2) + relation_queries
        keys = keys.unsqueeze(1) + relation_keys
        
        attn_weights = torch.einsum('bijhn,bijhn->bijh', [queries, keys])
        assert list(attn_weights.size()) == [bsz, tgt_len, src_len, self.num_heads]
        
        adj_mask = adj_mask.data.eq(config.PAD_idx)
        attn_weights.masked_fill_(
            adj_mask.unsqueeze(-1),
            float("-inf")
        )
        
        attn_weights = F.softmax(attn_weights, dim=2)
        
        if self.weights_dropout:
            attn_weights = self.attn_dropout(attn_weights)
        
        # attn_weights: bsz x tgt_len x src_len x heads
        # values: bsz x src_len x heads x dim
        outputs = torch.einsum('bijh,bjhn->bihn', [attn_weights, values]).contiguous()

        if not self.weights_dropout:
            outputs = self.attn_dropout(outputs)
        
        assert list(outputs.size()) == [bsz, tgt_len, self.num_heads, self.head_dim]
        
        outputs = self.output_linear(outputs.view(bsz, tgt_len, self.emb_dim))
        
        return outputs, attn_weights

class GraphTransformer(nn.Module):
    def __init__(self, layer_num, emb_dim, ffn_emb_dim, num_heads, dropout, weights_dropout=True):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layer_num):
            self.layers.append(GraphTransformerLayer(emb_dim, ffn_emb_dim, num_heads, dropout, weights_dropout))
    
    def forward(self, queries, keys, values, adj_mask, relation=None):
        for _, layer in enumerate(self.layers):
            outputs, _ = layer(queries, keys, values, adj_mask, relation)
        return outputs

class GraphTransformerLayer(nn.Module):
    def __init__(self, emb_dim, ffn_emb_dim, num_heads, dropout, weights_dropout=True):
        super(GraphTransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(emb_dim, emb_dim, emb_dim, emb_dim, num_heads=num_heads, dropout=dropout)
        self.relation_multi_head_attention = RelationMultiHeadAttention(emb_dim, num_heads, dropout, weights_dropout)
        self.fc1 = nn.Linear(emb_dim, ffn_emb_dim)
        self.fc2 = nn.Linear(ffn_emb_dim, emb_dim)
        
        self.attn_layer_norm = LayerNorm(emb_dim)
        self.ffn_layer_norm = LayerNorm(emb_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)
    
    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
    def forward(self, queries, keys, values, adj_mask, relation=None):
        residual = queries
        
        if relation is None:
            outputs, attn_weights = self.multi_head_attention(queries, keys, values, adj_mask)
        else:
            outputs, attn_weights = self.relation_multi_head_attention(queries, keys, values, adj_mask, relation)
        
        outputs = self.attn_dropout(outputs)
        outputs = self.attn_layer_norm(residual + outputs)
        
        residual = outputs 
        outputs = self.fc2(self.gelu(self.fc1(outputs)))
        outputs = self.ffn_dropout(outputs)
        outputs = self.ffn_layer_norm(residual + outputs)
        return outputs, attn_weights

def create_embedding(num, emb_dim, padding_idx=None):
    """Create and initialize embeddings."""
    embedding = Embeddings(num, emb_dim, padding_idx=padding_idx)
    nn.init.normal_(embedding.lut.weight, mean=0, std=emb_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(embedding.lut.weight[padding_idx], 0)
    return embedding

class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)
        self.sigm = nn.Sigmoid()
        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def forward(self, prior_enc, poster_enc):
        prior_enc = self.sigm(prior_enc)
        poster_enc = self.sigm(poster_enc)
        score = torch.squeeze(self.bilinear(prior_enc, poster_enc), 1)
        return self.sigm(score)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_num = 3 if config.woStrategy else 4
        input_dim = input_num * config.hidden_dim
        hid_num = 1 if config.woStrategy else 2
        hid_dim = hid_num * config.hidden_dim
        out_dim = config.hidden_dim

        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, mask):
        N = h.shape[0]
        assert self.dim == h.shape[2]
        mask=-1e30*mask.float()

        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)
        attention = F.softmax(e+mask.unsqueeze(-1),dim=1)
        return torch.matmul(torch.transpose(attention,1,2), h).squeeze(1),attention

class CASE(nn.Module):
    def __init__(
        self,
        vocab,
        emotion_num,
        strategy_num,
        model_file_path=None,
        is_eval=False,
        load_optim=False
    ):
        super(CASE, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.dataset = config.dataset
        
        self.word_freq = np.zeros(self.vocab_size)
        
        self.is_eval = is_eval
        self.rels = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]
        
        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.relation_embedding = create_embedding(
            config.relation_num, 
            emb_dim=config.emb_dim, 
            padding_idx=config.PAD_idx
        )
        
        # Context encoder
        self.encoder = self.make_encoder(config.emb_dim)
        # Cognition
        self.cognition_encoder = self.make_encoder(config.emb_dim)
        self.cs_graph_encoder = GraphTransformer(
            config.graph_layer_num, 
            config.emb_dim, 
            config.graph_ffn_emb_dim, 
            config.graph_num_heads, 
            config.dropout
        )
        self.cs_prior_attn = Attention(query_size=config.emb_dim,
                                       memory_size=config.emb_dim,
                                       hidden_size=config.emb_dim,
                                       mode="dot"
                                    )
        self.cs_posterior_attn = Attention(query_size=config.emb_dim,
                                       memory_size=config.emb_dim,
                                       hidden_size=config.emb_dim,
                                       mode="dot"
                                    )
        # Affection
        self.react_encoder = self.make_encoder(config.emb_dim)
        self.react_selfattn = SelfAttention(config.emb_dim, config.emb_dim)
        self.react_ctx_encoder = self.make_encoder(2 * config.emb_dim)
        self.react_linear = nn.Linear(config.emb_dim, config.emb_dim)
        self.fine_emotion_selfattn = SelfAttention(config.emb_dim, config.emb_dim)
        self.emotion_gate = nn.Linear(config.emb_dim, 1)
        self.emotion_norm = nn.Linear(2*config.emb_dim, config.emb_dim)
        self.concept_graph_encoder = GraphTransformer(
            config.graph_layer_num, 
            config.emb_dim, 
            config.graph_ffn_emb_dim, 
            config.graph_num_heads, 
            config.dropout
        )
        self.vad_layernorm = nn.LayerNorm(config.emb_dim)
        if self.dataset == "ESConv":
            self.emotion_num = len(ESC_MAP_EMO)
        else:
            self.emotion_num = len(ED_MAP_EMO)
        self.emotion_linear = nn.Linear(config.emb_dim, self.emotion_num)
        self.concept_prior_attn = Attention(query_size=config.emb_dim,
                                            memory_size=config.emb_dim,
                                            hidden_size=config.emb_dim,
                                            mode="dot"
                                        )
        self.concept_posterior_attn = Attention(query_size=config.emb_dim,
                                                memory_size=config.emb_dim,
                                                hidden_size=config.emb_dim,
                                                mode="dot"
                                            )
        # Alignment
        self.bow_output_layer = nn.Sequential(
                    nn.Linear(in_features=2*config.emb_dim, out_features=config.emb_dim),
                    nn.Tanh(),
                    nn.Linear(in_features=config.emb_dim, out_features=self.vocab_size),
                    nn.LogSoftmax(dim=-1))
        # self.discriminator = Discriminator(config.emb_dim)
        
        # Strategy
        if self.dataset == "ESConv":
            self.strategy_encoder = self.make_encoder(config.emb_dim)
            self.strategy_num = len(ESC_MAP_STRATEGY)
            self.strategy_embedding = create_embedding(
                self.strategy_num, 
                emb_dim=config.emb_dim,
                padding_idx=config.PAD_idx
            )
            self.position_embedding = create_embedding(
                self.strategy_num, 
                emb_dim=config.emb_dim,
                padding_idx=None
            )
            self.strategy_layernorm = LayerNorm(config.emb_dim)
            self.strategy_linear = nn.Linear(config.emb_dim, self.strategy_num)
            self.prior_query_linear = nn.Linear(2*config.emb_dim, config.emb_dim)
        else:
            self.prior_query_linear = nn.Linear(config.emb_dim, config.emb_dim)
        
        # Merge
        self.ctx_merge_lin = MLP()
        
        # Decoder
        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )
        
        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.activation = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout)
        
        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight
        
        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx, reduction="sum")
        self.criterion.weight = torch.ones(self.vocab_size)
        self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        self.criterion_kl = nn.KLDivLoss(reduction="mean")
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_bce_fine = nn.BCEWithLogitsLoss(reduction="none")
        # self.criterion_bce = nn.BCELoss()
        # self.criterion_bcs_fine = nn.BCELoss(reduction="none")
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_bow = nn.NLLLoss(ignore_index=config.PAD_idx, reduction='mean')
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim,
                1,
                config.warmup,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )
        
        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=config.device)
            self.load_state_dict(state["model"])
            if load_optim:
                self.optimizer.load_state_dict(state["optimizer"])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""
        
    def make_encoder(self, emb_dim):
        return Encoder(
            emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )

    def save_model(self, running_avg_ppl, iter):
        state = {
            "iter": iter,
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_ppl,
            "model": self.state_dict(),
        }
        model_save_path = os.path.join(
            self.model_dir,
            "CASE_{}_{:.4f}".format(iter, running_avg_ppl),
        )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def clean_preds(self, preds):
        res = []
        preds = preds.cpu().tolist()
        for pred in preds:
            if config.EOS_idx in pred:
                ind = pred.index(config.EOS_idx) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            if pred[0] == config.SOS_idx:
                pred = pred[1:]
            res.append(pred)
        return res

    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)
        for k, v in curr.items():
            if k != config.EOS_idx:
                self.word_freq[k] += v

    def calc_weight(self):
        RF = self.word_freq / self.word_freq.sum()
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)

        return torch.FloatTensor(weight).to(config.device)
    
    def add_position_embedding(self, sequence, seq_embeddings):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embedding(position_ids)
        sequence_emb = seq_embeddings + position_embeddings
        sequence_emb = self.strategy_layernorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb
    
    def infomax(self, cs_enc, cs_fake_enc, concept_enc, concept_fake_enc):
        cs_enc = self.dropout(cs_enc)
        concept_enc = self.dropout(concept_enc)
        logits = self.discriminator(cs_enc, concept_enc)
        
        cs_fake_enc = self.dropout(cs_fake_enc)
        cs_fake_logits = self.discriminator(cs_fake_enc, concept_enc)
        
        concept_fake_enc = self.dropout(concept_enc)
        concept_fake_logits = self.discriminator(cs_enc, concept_fake_enc)
        
        mim_logits = torch.cat((logits, cs_fake_logits, concept_fake_logits))
        mim_lables = torch.cat((torch.ones_like(logits),
                                torch.zeros_like(cs_fake_logits),
                                torch.zeros_like(concept_fake_logits)))
        mim_loss = self.criterion_bce(mim_logits, mim_lables)
        return mim_loss
    
    def infomax_score(self, cs_enc, cs_fake_enc, concept_enc, concept_fake_enc):
        pos_score = torch.mul(cs_enc, concept_enc)
        logits = torch.sigmoid(torch.sum(pos_score, -1))
        cs_fake_score = torch.mul(cs_fake_enc, concept_enc)
        cs_fake_logits = torch.sigmoid(torch.sum(cs_fake_score, -1))
        concept_fake_score = torch.mul(cs_enc, concept_fake_enc)
        concept_fake_logits = torch.sigmoid(torch.sum(concept_fake_score, -1))
        cs_distance = torch.sigmoid(logits - cs_fake_logits)
        cs_loss = self.criterion_bce(cs_distance, torch.ones_like(cs_distance, dtype=torch.float32))
        concept_distance = torch.sigmoid(logits - concept_fake_logits)
        concept_loss = self.criterion_bce(concept_distance, torch.ones_like(concept_distance, dtype=torch.float32))
        mim_loss = cs_loss + concept_loss
        return mim_loss
    
    def fine_grained_infomax_score(self, cs_enc, cs_fake_enc, react_enc, react_fake_enc, mask):
        bsz, num, _ = cs_enc.size()
        cs_enc = cs_enc.reshape(bsz*num, -1)
        cs_fake_enc = cs_fake_enc.reshape(bsz*num, -1)
        react_enc = react_enc.view(bsz*num, -1)
        react_fake_enc = react_fake_enc.view(bsz*num, -1)
       
        pos_score = torch.mul(cs_enc, react_enc)
        logits = torch.sigmoid(torch.sum(pos_score, -1))
        
        cs_fake_score = torch.mul(cs_enc, react_fake_enc)
        cs_fake_logits = torch.sigmoid(torch.sum(cs_fake_score, -1))
        cs_distance = torch.sigmoid(logits - cs_fake_logits)
        cs_loss = self.criterion_bce_fine(cs_distance, torch.ones_like(cs_distance, dtype=torch.float32))
        cs_loss = torch.sum(cs_loss * mask.flatten()) / torch.sum(mask.flatten())
        
        react_fake_score = torch.mul(cs_fake_enc, react_enc)
        react_fake_logits = torch.sigmoid(torch.sum(react_fake_score, -1))
        react_distance = torch.sigmoid(logits - react_fake_logits)
        react_loss = self.criterion_bce_fine(react_distance, torch.ones_like(react_distance, dtype=torch.float32))
        react_loss = torch.sum(react_loss * mask.flatten()) / torch.sum(mask.flatten())
        
        fine_mim_loss = cs_loss + react_loss
        return fine_mim_loss
    
    def train_one_batch(self, batch, iter, train=True):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        enc_vad_batch = batch["context_vad"]
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        
        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        
        # Encode Context
        src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_emb = self.embedding(batch["mask_input"])
        src_emb = self.embedding(enc_batch) + mask_emb
        enc_outputs = self.encoder(src_emb, src_mask)  # batch_size * seq_len * 300
        
        # Affection: Encode Concept
        concept_input = batch["concept_batch"]
        concept_mask = concept_input.data.eq(config.PAD_idx).unsqueeze(1)
        concept_vad_batch = batch["concept_vad_batch"]
        mask_concept = batch["mask_concept"]
        concept_adj_mask = batch["concept_adjacency_mask_batch"]
        # mask_concept = concept_input.data.eq(config.PAD_idx).unsqueeze(1)  # real mask
        # concept_mask = self.embedding(mask_concept)  # KG_idx embedding
        concept_emb = self.embedding(concept_input) + self.embedding(mask_concept)  # KG_idx embedding
        src_concept_input_emb = torch.cat((src_emb, concept_emb), dim=1)
        src_concept_outputs = self.concept_graph_encoder(src_concept_input_emb, 
                                                         src_concept_input_emb,
                                                         src_concept_input_emb,
                                                         concept_adj_mask)
        src_concept_mask = torch.cat((enc_batch, concept_input), dim=1).data.eq(config.PAD_idx)
        src_concept_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1)
        src_concept_vad = torch.softmax(src_concept_vad, dim=-1).unsqueeze(2).repeat(1, 1, config.emb_dim)
        src_concept_outputs = self.vad_layernorm(src_concept_vad * src_concept_outputs)
        
        # Cognition: Encode Commonsense
        bsz, uttr_num, uttr_length = batch["uttr_batch_concat"].size()
        uttr_batch_concat = batch["uttr_batch_concat"].view(bsz*uttr_num, -1)
        uttr_batch_mask = uttr_batch_concat.data.eq(config.PAD_idx).unsqueeze(1)
        
        bsz, cs_num, cs_length = batch["cs_batch"].size()
        cs_batch = batch["cs_batch"].view(bsz*cs_num, -1)
        cs_batch_mask = cs_batch.data.eq(config.PAD_idx).unsqueeze(1)
        cs_mask = batch["cs_mask"] # bsz, cs_num
        
        cs_adj_mask = batch["cs_adjacency_mask_batch"]
        
        uttr_batch_emb = self.embedding(uttr_batch_concat)
        uttr_batch_outputs = self.cognition_encoder(uttr_batch_emb, uttr_batch_mask)[:,0].view(bsz, uttr_num, -1)
        
        cs_batch_emb = self.embedding(cs_batch)
        cs_batch_outputs = self.cognition_encoder(cs_batch_emb, cs_batch_mask)[:,0].view(bsz, cs_num, -1)
        uttr_cs_outputs = torch.cat((uttr_batch_outputs, cs_batch_outputs), dim=1)
        
        assert uttr_cs_outputs.size(1) == cs_adj_mask.size(1)
        relation_emb = self.relation_embedding(cs_adj_mask)
        uttr_cs_graph_outputs = self.cs_graph_encoder(uttr_cs_outputs,
                                                  uttr_cs_outputs,
                                                  uttr_cs_outputs,
                                                  cs_adj_mask,
                                                  relation_emb)
        
        commonsense_outputs = uttr_cs_graph_outputs[:,-cs_batch_outputs.size(1):,:]
        commonsense_mask = cs_mask
        
        # Strategy: Encode Strategy Sequence
        if self.dataset == "ESConv":
            strategy_seqs = batch["strategy_seqs_batch"]
            mask_strategy = strategy_seqs.data.eq(config.PAD_idx).unsqueeze(1)
            strategy_seqs_emb = self.strategy_embedding(strategy_seqs)
            strategy_seqs_emb = self.add_position_embedding(strategy_seqs, strategy_seqs_emb)
            strategy_enc_outputs = self.strategy_encoder(strategy_seqs_emb, mask_strategy)
            strategy_enc_outputs = strategy_enc_outputs[:,0,:]
            
            prior_query = self.tanh(self.prior_query_linear(torch.cat((enc_outputs[:,0,:], strategy_enc_outputs), dim=-1)))
        else:
            prior_query = self.tanh(self.prior_query_linear(enc_outputs[:,0,:]))
        
        # concept
        prior_concept_enc, prior_concept_attn = self.concept_prior_attn(
            query = prior_query.unsqueeze(1), # enc_outputs[:,0,:].unsqueeze(1),
            memory = self.tanh(src_concept_outputs),
            mask = src_concept_mask
        )
        prior_concept_attn = prior_concept_attn.squeeze(1)
        
        # commonsense
        prior_cs_enc, prior_cs_attn = self.cs_prior_attn(
            query = prior_query.unsqueeze(1), # enc_outputs[:,0,:].unsqueeze(1),
            memory = self.tanh(commonsense_outputs),
            mask = commonsense_mask.eq(0)
        )
        prior_cs_attn = prior_cs_attn.squeeze(1)
        
        # knowledge selection
        target_post_batch = batch["target_post_batch"]
        mask_target = target_post_batch.data.eq(config.PAD_idx).unsqueeze(1)
        target_emb = self.embedding(target_post_batch) + self.embedding(batch["target_post_mask"])
        target_cs_outputs = self.cognition_encoder(target_emb, mask_target)[:, 0, :]
        target_concept_outputs = self.encoder(target_emb, mask_target)[:, 0, :]
        # merge strategy
        
        # concept selection
        posterior_concept_enc, posterior_concept_attn = self.concept_posterior_attn(
            query = target_concept_outputs.unsqueeze(1),
            memory = self.tanh(src_concept_outputs),
            mask = src_concept_mask
        )
        posterior_concept_attn = posterior_concept_attn.squeeze(1)
        
        # commonsense selection
        posterior_cs_enc, posterior_cs_attn = self.cs_posterior_attn(
            query = target_cs_outputs.unsqueeze(1),
            memory = self.tanh(commonsense_outputs),
            mask = commonsense_mask.eq(0)
        )
        posterior_cs_attn = posterior_cs_attn.squeeze(1)
        
        # pretrain bow loss
        bow_logits = self.bow_output_layer(torch.cat((posterior_concept_enc, posterior_cs_enc), dim=-1)) # bsz, 1, vocab_size
        bow_logits = bow_logits.repeat(1, dec_batch.size(1), 1)
        bow_loss = self.criterion_bow(
            bow_logits.contiguous().view(-1, bow_logits.size(-1)), 
            dec_batch.contiguous().view(-1)
        )
        
        if config.pretrain and train:
            bow_loss.backward()
            self.optimizer.step()
            return bow_loss.item()
        
        # distribution alignment
        concept_kl_loss = self.criterion_kl(torch.log(prior_concept_attn+1e-20),
                                            posterior_concept_attn.detach())
        cs_kl_loss = self.criterion_kl(torch.log(prior_cs_attn+1e-20),
                                       posterior_cs_attn.detach())
        kl_loss = cs_kl_loss + concept_kl_loss
        
        cs_enc = prior_cs_enc.squeeze(1)
        concept_enc = prior_concept_enc.squeeze(1)
        
        cs_fake_enc = torch.cat((cs_enc[-1].unsqueeze(0), cs_enc[:-1]), dim=0)
        concept_fake_enc = torch.cat((concept_enc[-1].unsqueeze(0), concept_enc[:-1]), dim=0)
        # mim_loss = self.infomax(cs_enc, cs_fake_enc, concept_enc, concept_fake_enc)
        coarse_mim_loss = self.infomax_score(cs_enc, cs_fake_enc, concept_enc, concept_fake_enc)
        
        # Fine-grained MIM
        bsz, react_uttr_num, _ = batch["react_batch"].size()
        assert uttr_num == react_uttr_num
        react_batch = batch["react_batch"].view(bsz*react_uttr_num, -1)
        react_batch_mask = react_batch.data.eq(config.PAD_idx).unsqueeze(1)
        react_emb = self.embedding(react_batch)
        react_batch_outputs = self.react_encoder(react_emb, react_batch_mask)
        # react_batch_enc, _ = self.react_selfattn(react_batch_outputs, react_batch_mask.squeeze(1))
        react_batch_enc = torch.mean(react_batch_outputs, dim=1)
        react_batch_enc = react_batch_enc.view(bsz, react_uttr_num, -1)
        react_batch_enc = self.react_ctx_encoder(torch.cat((react_batch_enc.unsqueeze(2).repeat(1, 1, enc_outputs.size(1), 1),
                                                            enc_outputs.unsqueeze(1).repeat(1, react_uttr_num, 1, 1)), 
                                                            dim=-1).view(bsz*react_uttr_num, enc_outputs.size(1), -1), 
                                                            src_mask.unsqueeze(1).repeat(1, react_uttr_num, 1, 1).view(bsz*react_uttr_num, -1,enc_outputs.size(1)))
        react_batch_enc = react_batch_enc[:,0,:].view(bsz, react_uttr_num, -1)
        react_batch_enc = self.react_linear(react_batch_enc)
        bsz, _, emb = react_batch_enc.size()
        uttr_emotion = react_batch_enc[:,0].unsqueeze(1).repeat(1, batch["max_uttr_cs_num"], 1)
        split_intent_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_intent_num"], 1).view(bsz, -1, emb)
        split_need_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_need_num"], 1).view(bsz, -1, emb)
        split_want_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_want_num"], 1).view(bsz, -1, emb)
        split_effect_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_effect_num"], 1).view(bsz, -1, emb)
        react_enc = torch.cat((uttr_emotion, 
                               split_intent_emotion,
                               split_need_emotion,
                               split_want_emotion,
                               split_effect_emotion), dim=1)
        assert react_enc.size(1) == cs_num
        react_fake_enc = torch.cat((react_enc[-1].unsqueeze(0), react_enc[:-1]), dim=0)
        commonsense_fake_outputs = torch.cat((commonsense_outputs[-1].unsqueeze(0), commonsense_outputs[:-1]), dim=0)
        fine_mim_loss = self.fine_grained_infomax_score(commonsense_outputs, commonsense_fake_outputs, react_enc, react_fake_enc, commonsense_mask)
        
        mim_loss = config.coarse_weight * coarse_mim_loss + config.fine_weight * fine_mim_loss
        
        # uttr_split_react_mask = batch["uttr_split_react_mask"]
        # fine_mask = torch.cat((torch.ones((bsz, 1)).long().to(config.device), uttr_split_react_mask), dim=1)
        # assert fine_mask.size(1) == react_batch_enc.size(1)
        # fine_emotion, _ = self.fine_emotion_selfattn(react_batch_enc, fine_mask.eq(0))
        
        fine_emotion = react_batch_enc[:, 0]
        emotion_emb = self.emotion_norm(torch.cat((concept_enc, fine_emotion), dim=-1))
        emo_gate = F.sigmoid(self.emotion_gate(emotion_emb))
        emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion
        
        # emotion prediction
        if self.dataset == "ED":
            emotion_logits = self.emotion_linear(emotion_enc)
            emotion_loss = self.criterion_ce(emotion_logits, batch["program_label"])
            pred_emotion = np.argmax(emotion_logits.detach().cpu().numpy(), axis=1)
            emotion_acc = accuracy_score(batch["program_label"].detach().cpu().numpy(), pred_emotion)
        
        # Merge Context, Cognition-Affection-Strategy Signals
        if self.dataset == "ESConv":
            ctx_enc_outputs = self.ctx_merge_lin(torch.cat((
                enc_outputs, 
                cs_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                concept_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                strategy_enc_outputs.unsqueeze(1).repeat(1, enc_outputs.size(1), 1)
            ), dim=2))
            
            # predict next strategy
            strategy_logits = self.strategy_linear(ctx_enc_outputs[:,0,:])
            strategy_label = batch["strategy_label"]
            str_loss = self.criterion_ce(strategy_logits, strategy_label)
            
            pred_strategy = np.argmax(strategy_logits.detach().cpu().numpy(), axis=1)
            strategy_acc = accuracy_score(batch["strategy_label"].detach().cpu().numpy(), pred_strategy)
        else:
            ctx_enc_outputs = self.ctx_merge_lin(torch.cat((
                enc_outputs, 
                cs_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                emotion_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
            ), dim=2))
        
        # Decoder
        sos_token = (
            torch.LongTensor([config.SOS_idx] * enc_batch.size(0))
            .unsqueeze(1)
            .to(config.device)
        )
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), dim=1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        
        # batch_size * seq_len * 300 (GloVe)
        dec_emb = self.embedding(dec_batch_shift)
        pre_logit, attn_dist = self.decoder(
            dec_emb, 
            ctx_enc_outputs, 
            (src_mask, mask_trg),
            cs_enc_outputs=commonsense_outputs,
            cs_enc_mask=commonsense_mask.eq(0).unsqueeze(1),
            concept_enc_outputs=src_concept_outputs,
            concept_enc_mask=src_concept_mask.unsqueeze(1),
        )
        
        ## compute output dist
        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab if config.pointer_gen else None,
            extra_zeros,
            attn_dist_db=None,
        )
        
        ctx_loss = self.criterion_ppl(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )
        
        # reference CEM
        _, preds = logit.max(dim=-1)
        preds = self.clean_preds(preds)
        self.update_frequency(preds)
        self.criterion.weight = self.calc_weight()
        not_pad = dec_batch.ne(config.PAD_idx)
        target_tokens = not_pad.long().sum().item()
        div_loss = self.criterion(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )
        div_loss /= target_tokens
        
        if self.dataset == "ED":
            loss = bow_loss + kl_loss + mim_loss + ctx_loss + 1.5 * div_loss + emotion_loss
        else:
            loss = bow_loss + kl_loss + mim_loss + ctx_loss + 1.5 * div_loss + str_loss
            
        if train:
            loss.backward()
            self.optimizer.step()
        
        return (
            bow_loss.item(),
            kl_loss.item(),
            mim_loss.item(),
            ctx_loss.item(),
            math.exp(min(ctx_loss.item(), 100)),
            str_loss.item() if self.dataset=="ESConv" else 0,
            strategy_acc.item() if self.dataset=="ESConv" else 0,
            emotion_loss.item() if self.dataset=="ED" else 0,
            emotion_acc.item() if self.dataset=="ED" else 0
        )
    
    def decoder_greedy(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        enc_vad_batch = batch["context_vad"]
        
        # Encode Context
        src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_emb = self.embedding(batch["mask_input"])
        src_emb = self.embedding(enc_batch) + mask_emb
        enc_outputs = self.encoder(src_emb, src_mask)  # batch_size * seq_len * 300
        
        # Affection: Encode Concept
        concept_input = batch["concept_batch"]
        concept_mask = concept_input.data.eq(config.PAD_idx).unsqueeze(1)
        concept_vad_batch = batch["concept_vad_batch"]
        mask_concept = batch["mask_concept"]
        concept_adj_mask = batch["concept_adjacency_mask_batch"]
        # mask_concept = concept_input.data.eq(config.PAD_idx).unsqueeze(1)  # real mask
        # concept_mask = self.embedding(mask_concept)  # KG_idx embedding
        concept_emb = self.embedding(concept_input) + self.embedding(mask_concept)  # KG_idx embedding
        src_concept_input_emb = torch.cat((src_emb, concept_emb), dim=1)
        src_concept_outputs = self.concept_graph_encoder(src_concept_input_emb, 
                                                         src_concept_input_emb,
                                                         src_concept_input_emb,
                                                         concept_adj_mask)
        src_concept_mask = torch.cat((enc_batch, concept_input), dim=1).data.eq(config.PAD_idx)
        src_concept_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1)
        src_concept_vad = torch.softmax(src_concept_vad, dim=-1).unsqueeze(2).repeat(1, 1, config.emb_dim)
        src_concept_outputs = self.vad_layernorm(src_concept_vad * src_concept_outputs)
        
        # Cognition: Encode Commonsense
        bsz, uttr_num, uttr_length = batch["uttr_batch_concat"].size()
        uttr_batch_concat = batch["uttr_batch_concat"].view(bsz*uttr_num, -1)
        uttr_batch_mask = uttr_batch_concat.data.eq(config.PAD_idx).unsqueeze(1)
        
        bsz, cs_num, cs_length = batch["cs_batch"].size()
        cs_batch = batch["cs_batch"].view(bsz*cs_num, -1)
        cs_batch_mask = cs_batch.data.eq(config.PAD_idx).unsqueeze(1)
        cs_mask = batch["cs_mask"] # bsz, cs_num
        
        cs_adj_mask = batch["cs_adjacency_mask_batch"]
        
        uttr_batch_emb = self.embedding(uttr_batch_concat)
        uttr_batch_outputs = self.cognition_encoder(uttr_batch_emb, uttr_batch_mask)[:,0].view(bsz, uttr_num, -1)
        
        cs_batch_emb = self.embedding(cs_batch)
        cs_batch_outputs = self.cognition_encoder(cs_batch_emb, cs_batch_mask)[:,0].view(bsz, cs_num, -1)
        uttr_cs_outputs = torch.cat((uttr_batch_outputs, cs_batch_outputs), dim=1)
        
        assert uttr_cs_outputs.size(1) == cs_adj_mask.size(1)
        relation_emb = self.relation_embedding(cs_adj_mask)
        uttr_cs_graph_outputs = self.cs_graph_encoder(uttr_cs_outputs,
                                                  uttr_cs_outputs,
                                                  uttr_cs_outputs,
                                                  cs_adj_mask,
                                                  relation_emb)
        
        commonsense_outputs = uttr_cs_graph_outputs[:,-cs_batch_outputs.size(1):,:]
        commonsense_mask = cs_mask
        if self.dataset == "ESConv":
            # Strategy: Encode Strategy Sequence
            strategy_seqs = batch["strategy_seqs_batch"]
            mask_strategy = strategy_seqs.data.eq(config.PAD_idx).unsqueeze(1)
            strategy_seqs_emb = self.strategy_embedding(strategy_seqs)
            strategy_seqs_emb = self.add_position_embedding(strategy_seqs, strategy_seqs_emb)
            strategy_enc_outputs = self.strategy_encoder(strategy_seqs_emb, mask_strategy)
            strategy_enc_outputs = strategy_enc_outputs[:,0,:]
            
            prior_query = self.tanh(self.prior_query_linear(torch.cat((enc_outputs[:,0,:], strategy_enc_outputs), dim=-1)))
        else:
            prior_query = self.tanh(self.prior_query_linear(enc_outputs[:,0,:]))
        
        # concept
        prior_concept_enc, prior_concept_attn = self.concept_prior_attn(
            query = prior_query.unsqueeze(1), # enc_outputs[:,0,:].unsqueeze(1),
            memory = self.tanh(src_concept_outputs),
            mask = src_concept_mask
        )
        prior_concept_attn = prior_concept_attn.squeeze(1)
        
        # commonsense
        prior_cs_enc, prior_cs_attn = self.cs_prior_attn(
            query = prior_query.unsqueeze(1), # enc_outputs[:,0,:].unsqueeze(1),
            memory = self.tanh(commonsense_outputs),
            mask = commonsense_mask.eq(0)
        )
        prior_cs_attn = prior_cs_attn.squeeze(1)
        cs_enc = prior_cs_enc.squeeze(1)
        concept_enc = prior_concept_enc.squeeze(1)
        
        # Fine-grained MIM
        bsz, react_uttr_num, _ = batch["react_batch"].size()
        assert uttr_num == react_uttr_num
        react_batch = batch["react_batch"].view(bsz*react_uttr_num, -1)
        react_batch_mask = react_batch.data.eq(config.PAD_idx).unsqueeze(1)
        react_emb = self.embedding(react_batch)
        react_batch_outputs = self.react_encoder(react_emb, react_batch_mask)
        # react_batch_enc, _ = self.react_selfattn(react_batch_outputs, react_batch_mask.squeeze(1))
        react_batch_enc = torch.mean(react_batch_outputs, dim=1)
        react_batch_enc = react_batch_enc.view(bsz, react_uttr_num, -1)
        react_batch_enc = self.react_ctx_encoder(torch.cat((react_batch_enc.unsqueeze(2).repeat(1, 1, enc_outputs.size(1), 1),
                                                        enc_outputs.unsqueeze(1).repeat(1, react_uttr_num, 1, 1)), 
                                                        dim=-1).view(bsz*react_uttr_num, enc_outputs.size(1), -1), 
                                                        src_mask.unsqueeze(1).repeat(1, react_uttr_num, 1, 1).view(bsz*react_uttr_num, -1,enc_outputs.size(1)))
        react_batch_enc = react_batch_enc[:,0,:].view(bsz, react_uttr_num, -1)
        react_batch_enc = self.react_linear(react_batch_enc)
        bsz, _, emb = react_batch_enc.size()
        uttr_emotion = react_batch_enc[:,0].unsqueeze(1).repeat(1, batch["max_uttr_cs_num"], 1)
        split_intent_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_intent_num"], 1).view(bsz, -1, emb)
        split_need_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_need_num"], 1).view(bsz, -1, emb)
        split_want_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_want_num"], 1).view(bsz, -1, emb)
        split_effect_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_effect_num"], 1).view(bsz, -1, emb)
        react_enc = torch.cat((uttr_emotion, 
                               split_intent_emotion,
                               split_need_emotion,
                               split_want_emotion,
                               split_effect_emotion), dim=1)
        assert react_enc.size(1) == cs_num
   
        # uttr_split_react_mask = batch["uttr_split_react_mask"]
        # fine_mask = torch.cat((torch.ones((bsz, 1)).long().to(config.device), uttr_split_react_mask), dim=1)
        # assert fine_mask.size(1) == react_batch_enc.size(1)
        # fine_emotion, _ = self.fine_emotion_selfattn(react_batch_enc, fine_mask.eq(0))
        
        fine_emotion = react_batch_enc[:, 0]
        emotion_emb = self.emotion_norm(torch.cat((concept_enc, fine_emotion), dim=-1))
        emo_gate = F.sigmoid(self.emotion_gate(emotion_emb))
        emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion
        
        # Merge Context, Cognition-Affection-Strategy Signals
        if self.dataset == "ESConv":
            ctx_enc_outputs = self.ctx_merge_lin(torch.cat((
                enc_outputs, 
                cs_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                concept_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                strategy_enc_outputs.unsqueeze(1).repeat(1, enc_outputs.size(1), 1)
            ), dim=2))
        else:
            ctx_enc_outputs = self.ctx_merge_lin(torch.cat((
                enc_outputs, 
                cs_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                emotion_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
            ), dim=2))

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            ys_embed = self.embedding(ys)
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    ys_embed, ctx_enc_outputs, (src_mask, mask_trg),
                    cs_enc_outputs=commonsense_outputs,
                    cs_enc_mask=commonsense_mask.eq(0).unsqueeze(1),
                    concept_enc_outputs=src_concept_outputs,
                    concept_enc_mask=src_concept_mask.unsqueeze(1),
                )

            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, _ = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), ctx_output, (src_mask, mask_trg)
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            filtered_logit = top_k_top_p_filtering(
                logit[0, -1] / 0.7, top_k=0, top_p=0.9, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logit, dim=-1)

            next_word = torch.multinomial(probs, 1).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            # _, next_word = torch.max(logit[:, -1], dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

        # Cognition: Encode Commonsense
        # bsz = batch["uttr_batch"].size(0)
        # uttr_batch = batch["uttr_batch"]
        # uttr_mask = uttr_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # bsz, uttr_split_num, _ = batch["uttr_split_batch"].size()
        # uttr_split_batch = batch["uttr_split_batch"].view(bsz*uttr_split_num, -1)
        # uttr_split_mask = uttr_split_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # uttr_batch_emb = self.embedding(uttr_batch)
        # uttr_batch_outputs = self.cognition_encoder(uttr_batch_emb, uttr_mask)[:,0].unsqueeze(1)
        # uttr_split_batch_emb = self.embedding(uttr_split_batch)
        # uttr_split_batch_ouptuts = self.cognition_encoder(uttr_split_batch_emb, uttr_split_mask)[:,0].view(bsz, uttr_split_num, -1)
        
        # bsz, cs_num, _ = batch["x_intent_batch"].size()
        # cs_batch = torch.cat((batch["x_intent_batch"].view(bsz*cs_num, -1),
        #                       batch["x_need_batch"].view(bsz*cs_num, -1),
        #                       batch["x_want_batch"].view(bsz*cs_num, -1),
        #                       batch["x_effect_batch"].view(bsz*cs_num, -1),
        #                       ), dim=0)
        # cs_batch_mask = cs_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # cs_mask = torch.cat((batch["x_intent_mask"],
        #                      batch["x_need_mask"],
        #                      batch["x_want_mask"],
        #                      batch["x_effect_mask"]), dim=1)
        
        # bsz, split_num, cs_split_num, _ = batch["x_intent_split_batch"].size()
        # cs_split_batch = torch.cat((batch["x_intent_split_batch"].view(bsz*split_num*cs_split_num, -1),
        #                       batch["x_need_split_batch"].view(bsz*split_num*cs_split_num, -1),
        #                       batch["x_want_split_batch"].view(bsz*split_num*cs_split_num, -1),
        #                       batch["x_effect_split_batch"].view(bsz*split_num*cs_split_num, -1)), dim=0)
        # cs_split_batch_mask = cs_split_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # cs_split_mask = torch.cat((batch["x_intent_split_mask"].view(bsz, -1),
        #                            batch["x_need_split_mask"].view(bsz, -1),
        #                            batch["x_want_split_mask"].view(bsz, -1),
        #                            batch["x_effect_split_mask"].view(bsz, -1)))
        
        # cs_batch_emb = self.embedding(cs_batch)
        # cs_batch_outputs = self.cognition_encoder(cs_batch_emb, cs_batch_mask)
        # cs_split_batch_emb = self.embedding(cs_split_batch)
        # cs_split_batch_outputs = self.cognition_encoder(cs_split_batch_emb, cs_split_batch_mask)
        # cs_outputs = torch.cat((cs_batch_outputs[:,0].view(bsz, cs_num, -1), 
        #                         cs_split_batch_outputs[:,0].view(bsz, split_num*cs_split_num, -1)), dim=1)
        # src_cs_emb = torch.cat((uttr_batch_outputs, 
                                # uttr_split_batch_ouptuts,
                                # cs_outputs), dim=1)
        
        # cs_enc = torch.bmm(prior_cs_attn*commonsense_mask, commonsense_outputs)
        # src_concept_vad = torch.softmax(src_concept_vad, dim=-1)
        # concept_enc = torch.bmm(prior_concept_attn*src_concept_vad, src_concept_outputs)