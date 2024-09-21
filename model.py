import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import librosa
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
import os

# in line 52, use torch.nn.functional.relu cause unknown error, so we write it by ourselves
def my_relu(x):
    return torch.maximum(x, torch.zeros_like(x))
def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    """
    Computes an attention mask of the form `(batch, seq_len)` with an attention for each element in the batch that
    stops at the corresponding element in `seq_lens`.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, *)`):
            The sequences to mask, where `*` is any number of sequence-specific dimensions including none.
        seq_lens (`torch.Tensor` of shape `(batch)`:
            Each element represents the length of the sequence at the same index in `hidden_states`

    Returns:
        `torch.FloatTensor`: The float attention mask of shape `(batch, seq_len)`
    """
    batch_size, mask_seq_len = hidden_states.shape[:2]

    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    mask = mask.masked_fill(bool_mask, 0)

    return mask

# FeedForward模組，用於Length Adapter
class ConformerFeedForward(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(p=dropout, inplace=True)
        self.intermediate_dense = nn.Linear(in_features=1024, out_features=4096, bias=True)
        # self.intermediate_act_fn = torch.nn.ReLU(inplace=False)
        self.output_dense = nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.output_dropout = nn.Dropout(p=dropout, inplace=True)
    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = my_relu(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states
################################################################################################
# self-attention 模組，用於Length Adapter
# 此slef-atteintion 模組有使用相對位置編碼
# copy from SeamlessM4T model
class ConformerSelfAttention(nn.Module):
    def __init__(self, use_position_embeddings=True):
        super().__init__()
        self.hidden_size = 1024
        self.speech_encoder_attention_heads = 16
        self.position_embeddings_type = "relative_key"
        # 最多往左看8個數
        self.right_max_position_embeddings = 8
        # 最多往右看64個數
        self.left_max_position_embeddings = 64

        
        self.head_size = self.hidden_size // self.speech_encoder_attention_heads
        self.num_heads = self.speech_encoder_attention_heads
        self.position_embeddings_type = self.position_embeddings_type if use_position_embeddings else None

        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(p=0.0, inplace=False)

        if self.position_embeddings_type == "relative_key":
            self.left_max_position_embeddings = self.left_max_position_embeddings
            self.right_max_position_embeddings = self.right_max_position_embeddings
            num_positions = self.left_max_position_embeddings + self.right_max_position_embeddings + 1
            self.distance_embedding = nn.Embedding(num_positions, self.head_size)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # self-attention mechanism
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        if self.position_embeddings_type == "relative_key":
            query_length, key_length = query.shape[2], key.shape[2]

            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_r - position_ids_l
            distance = torch.clamp(distance, -self.left_max_position_embeddings, self.right_max_position_embeddings)

            positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
            positional_embedding = positional_embedding.to(dtype=query.dtype)  # fp16 compatibility

            relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)
            attn_weights = attn_weights + (relative_position_attn_weights / math.sqrt(self.head_size))

        # apply attention_mask if necessary
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # => (batch, head, time1, time2)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # => (batch, head, time1, d_k)
        attn_output = torch.matmul(attn_weights, value)

        # => (batch, time1, hidden_size)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        attn_output = self.linear_out(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
        

# TextualAdapter 模組，單層self-attention with relative position encoding，與上面的ConformerSelfAttention相同，但dimension不同
# wav-bert 2.0 的 dim = 1024, Length Adapter 的 dim = 1024, 但 TextualAdapter 的 dim = 512
class TextualAdapter(nn.Module):
    def __init__(self, use_position_embeddings=True):
        super().__init__()
        self.hidden_size = 512
        self.speech_encoder_attention_heads = 8
        self.position_embeddings_type = "relative_key"
        self.right_max_position_embeddings = 8
        self.left_max_position_embeddings = 64
        
        
        
        self.head_size = self.hidden_size // self.speech_encoder_attention_heads
        self.num_heads = self.speech_encoder_attention_heads
        self.position_embeddings_type = self.position_embeddings_type if use_position_embeddings else None

        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(p=0.0, inplace=False)

        if self.position_embeddings_type == "relative_key":
            self.left_max_position_embeddings = self.left_max_position_embeddings
            self.right_max_position_embeddings = self.right_max_position_embeddings
            num_positions = self.left_max_position_embeddings + self.right_max_position_embeddings + 1
            self.distance_embedding = nn.Embedding(num_positions, self.head_size)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # self-attention mechanism
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        if self.position_embeddings_type == "relative_key":
            query_length, key_length = query.shape[2], key.shape[2]

            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_r - position_ids_l
            distance = torch.clamp(distance, -self.left_max_position_embeddings, self.right_max_position_embeddings)

            positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
            positional_embedding = positional_embedding.to(dtype=query.dtype)  # fp16 compatibility

            relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)
            attn_weights = attn_weights + (relative_position_attn_weights / math.sqrt(self.head_size))

        # apply attention_mask if necessary
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # => (batch, head, time1, time2)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # => (batch, head, time1, d_k)
        attn_output = torch.matmul(attn_weights, value)

        # => (batch, time1, hidden_size)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        attn_output = self.linear_out(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output

# =======================================================================================================
# Length Adapter模組的forward，用於縮短speech encoder output的長度
class ConformerAdapterLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 1024
        self.kernel_size = 2
        self.stride = 2

        # 1. residual convolution
        self.residual_layer_norm = nn.LayerNorm(self.embed_dim)
        # 縮短長度約1/2，長度計算方法可參考:https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.residual_conv = nn.Conv1d(
            self.embed_dim,
            2 * self.embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.activation = nn.GLU(dim=1)

        # 2. Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.self_attn_conv = nn.Conv1d(
            self.embed_dim,
            2 * self.embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.self_attn = ConformerSelfAttention()
        self.self_attn_dropout = nn.Dropout(p=0.1, inplace=False)

        # 3. Feed-forward
        self.ffn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = ConformerFeedForward(dropout=0.1)

    # 因為輸出會改變輸入的 seq_len ，後續的計算會需要用到 attention mask，因此需要對 attention mask 做修改
    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        pad = self.kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1

        return seq_lens.floor()

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        residual = self.residual_layer_norm(hidden_states)

        # Apply pooling to the residual to match the sequence length of the
        # multi-head attention output.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        residual = residual.transpose(1, 2)

        hidden_states = self.self_attn_layer_norm(hidden_states)
        # Apply pooling before feeding to the multihead-attention layer.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.transpose(1, 2)

        attention_mask_4d = None
        if attention_mask is not None:
            # 取得縮短後的 seq_len
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                hidden_states.device
            )
            # 根據縮短的 seq_len 重新計算 attention mask
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            attention_mask_4d = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        # The rest of the computation is identical to a vanilla Transformer
        # encoder layer.
        hidden_states, attn_weigths = self.self_attn(
            hidden_states,
            attention_mask=attention_mask_4d,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states) + residual

        return hidden_states , attention_mask

# Length Adapter 模組本身，因為 num_adapter = 2 所以上面的 ConformerAdapterLayer 會跑兩次
class ConformerAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_adapter = 2

        self.layers = nn.ModuleList(
            [ConformerAdapterLayer() for i in range(self.num_adapter)]
        )

    def forward(self, hidden_states, attention_mask):
        # down project hidden_states if necessary

        for layer in self.layers:
            hidden_states, attention_mask = layer(hidden_states, attention_mask)

        return hidden_states, attention_mask
        
################################################################################################



# Speech Encoder 模組，包含w2v-bert 2.0以及 Length Adapter
class SpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0") # from huggingface
        self.intermediate_ffn = ConformerFeedForward(dropout=0.0)
        self.adapter = ConformerAdapter()
        self.inner_layer_norm = nn.LayerNorm(1024)
        # 因為 decoder 的 dim = 512, 所以這邊經過 Lengh Adapter 後先將 dim 降成 512, 之後進入 TextualAdapter 後還是 dim = 512, 銜接到 decoder 的 dim
        self.downsample = nn.Linear(1024, 512)
        ##################
        
    def forward(
        self,
        inputs,
        attention_mask,
        output_attentions = False
    ):
        hidden_states = self.encoder(input_features=inputs, attention_mask=attention_mask)

        expanded_hidden_states = self.intermediate_ffn(hidden_states.last_hidden_state)
        hidden_states = hidden_states.last_hidden_state + 0.5 * expanded_hidden_states

        if self.adapter is not None:
            hidden_states, attention_mask_adapter = self.adapter(hidden_states, attention_mask=attention_mask)

        hidden_states = self.inner_layer_norm(hidden_states)
        hidden_states = self.downsample(hidden_states)

        # 因為 Textual Adapter 是 self-attention，所以需要用到 attention mask
        return hidden_states, attention_mask_adapter

################################################################################################
def get_text_encoder():
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    return model.base_model.encoder
def get_text_decoder():
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    return model.base_model.decoder
def get_embedding():
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    embeddings = model.get_input_embeddings()
    return embeddings.weight
def get_bias():
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    return model.final_logits_bias

def get_lm_head():
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    linear = torch.nn.Linear(512, 65001)
    embeddings = model.get_input_embeddings()
    linear.weight = torch.nn.Parameter(embeddings.weight)
    linear.bias = torch.nn.Parameter(model.final_logits_bias)
    return linear
    
class SpeechToTextTranslationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = get_text_encoder()
        self.text_decoder = get_text_decoder()
        # lm_head 為 nn.Linear, in dim = 512, out dim = 65001
        self.lm_head = get_lm_head()
        ######### Textual Adapter Module #########
        self.textual_adapter = TextualAdapter()
        self.final_projection = nn.Linear(65001, 512)
        ##########################################
        self.speech_encoder = SpeechEncoder()

    def forward(self, text_input_ids, text_attention_mask, speech_input, speech_attention_mask, decoder_input_ids, decoder_attention_mask):
        # input ids 經過 text encoder
        text_encoder_output = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state
        # input audio 經過 speech encoder 以及 Textual Adapter
        speech_encoder_output, speech_encoder_logit, speech_encoder_attention_mask = self.forward_speech(speech_input=speech_input, attention_mask=speech_attention_mask)
        # CTC loss 需要知道 batch 中每一筆資料的 seq_len
        # 計算 batch 中每一筆資料的 seq_len
        speech_encoder_len = None
        if(speech_encoder_attention_mask is not None):
            speech_encoder_len = torch.sum(speech_encoder_attention_mask, 1)
        # text decoder 根據 text encoder 輸出 target language text
        text_decoder_output = self.text_decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=text_encoder_output, encoder_attention_mask=text_attention_mask).last_hidden_state
        # text decoder 根據 speech encoder 輸出 target language text
        speech_decoder_output = self.text_decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=speech_encoder_output, encoder_attention_mask=speech_encoder_attention_mask).last_hidden_state
        # 個別經過 Linear(512, 65001) 映射到 tokenizer 的文字數量
        text_logits = self.lm_head(text_decoder_output)
        speech_logits = self.lm_head(speech_decoder_output)
        # text_logits, speech_logits : text decoder 輸出  # speech_encoder_logit, speech_encoder_len : 為了計算 CTC loss
        return text_logits, speech_logits, speech_encoder_logit, speech_encoder_len
        
    def forward_text(self, text_input_ids, attention_mask=None):
        return self.text_encoder(input_ids=text_input_ids, attention_mask=attention_mask).last_hidden_state

    def forward_speech(self, speech_input, attention_mask=None):
        hidden_states, attention_mask_adapter = self.speech_encoder(inputs=speech_input, attention_mask=attention_mask)
        # 將 speech domain 映射到 text domain，為了計算 CTC loss，結果要回傳算 CTC loss
        speech_encoder_logit = self.lm_head(hidden_states)
        # 將 text domain 映射回 model dimension
        hidden_states = self.final_projection(speech_encoder_logit)
        
        # prepare 4d attention mask for textual adapter
        # 因為 Textual Adapter 為 self-attention，所以需要準備 attention mask
        attention_mask_4d = None
        if attention_mask_adapter is not None:
            attention_mask_4d = _prepare_4d_attention_mask(
                attention_mask_adapter,
                hidden_states.dtype,
            )
        output = self.textual_adapter(
            hidden_states,
            attention_mask=attention_mask_4d,
            output_attentions=False,)
        # attention_mask_adapter是為了給後續的 text decoder 計算而回傳
        return output, speech_encoder_logit, attention_mask_adapter

    def inference(self, encoder_hidden_state, attention_mask_adapter=None, tokenizer=None, device=None):
        if attention_mask_adapter is not None:
            attention_mask_adapter = attention_mask_adapter.to(device)
        encoder_hidden_state = encoder_hidden_state.to(device)
        seq_len = 1
        eos_ids = tokenizer(tokenizer.eos_token, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        decoder_input_ids = tokenizer("<pad>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        while(seq_len<512):
            decoder_output_vectors = self.text_decoder(decoder_input_ids, encoder_hidden_states=encoder_hidden_state, encoder_attention_mask=attention_mask_adapter).last_hidden_state
            lm_logits = self.lm_head(decoder_output_vectors)
            next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
            if(next_decoder_input_ids==eos_ids):
                break
            else:
                seq_len = seq_len+1
            
        return tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True), decoder_input_ids

    def inference_text(self, encoder_hidden_state, attention_mask=None, tokenizer=None, device=None):
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        encoder_hidden_state = encoder_hidden_state.to(device)
        seq_len = 1
        eos_ids = tokenizer(tokenizer.eos_token, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        decoder_input_ids = tokenizer("<pad>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        while(seq_len<512):
            decoder_output_vectors = self.text_decoder(decoder_input_ids, encoder_hidden_states=encoder_hidden_state, encoder_attention_mask=attention_mask).last_hidden_state
            lm_logits = self.lm_head(decoder_output_vectors)
            next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
            if(next_decoder_input_ids==eos_ids):
                break
            else:
                seq_len = seq_len+1
            
        return tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True), decoder_input_ids

    
        
    