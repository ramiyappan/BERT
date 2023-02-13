from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this attention is applied after calculating the attention score following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiply query and key 
    # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number
    
    """ <<--- STUDENT IMPLEMENTATION --->> """
    # get dimension of key
    d_k = self.attention_head_size # key.size(-1) == attention_head_size

    # calculate (query * key) / sqrt(d_k)
    S = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(d_k)

    # apply attention_mask before normalizing, which adds 0 to non-padding tokens and -inf to mask future tokens.
    S += attention_mask

    # normalize the scores
    attn = self.dropout(F.softmax(S, dim=-1))

    # multiply the attention scores to the value and get back V'
    scores = torch.matmul(attn, value) 

    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
    scores = scores.permute(0,2,1,3)
    scores = scores.reshape(key.size(0), key.size(2), -1)

    return scores
    #raise NotImplementedError
    """ <<--- END OF STUDENT IMPLEMENTATION --->> """

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # calculate the multi-head attention 
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # self attention
    self.self_attention = BertSelfAttention(config)
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # layer out
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    input: the input
    output: the input that requires the sublayer to transform
    dense_layer, dropput: the sublayer
    ln_layer: layer norm that takes input+sublayer(output)
    """

    # todo
    """ <<--- STUDENT IMPLEMENTATION --->> """
    return ln_layer(input+dropout(dense_layer(output)))
    """ <<--- END OF STUDENT IMPLEMENTATION --->> """

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
    each block consists of 
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the output of BertSelfAttention and the input of BertSelfAttention
    3. a feed forward layer
    4. a add-norm that takes the output of feed forward layer and the input of feed forward layer
    """

    # todo
    """ <<--- STUDENT IMPLEMENTATION --->> """
    # multi-head attention w/ self.self_attention
    multi_head_layer = self.self_attention(hidden_states, attention_mask)
    
    # add-norm layer
    # takes i/o of BertSelfAttention
    norm_layer = self.add_norm(hidden_states, multi_head_layer, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
    
    # feed forward
    ff_layer = self.interm_af(self.interm_dense(norm_layer))
    
    # another add-norm layer
    # takes i/o of feed forward layer
    norm_layer = self.add_norm(norm_layer, ff_layer, self.out_dense, self.out_dropout, self.out_layer_norm)
    
    return norm_layer
    #raise NotImplementedError
    """ <<--- END OF STUDENT IMPLEMENTATION --->> """


class BertModel(BertPreTrainedModel):
  """
  the bert model returns the final embeddings for each token in a sentence
  it consists
  1. embedding (used in self.embed)
  2. a stack of n bert layers (used in self.encode)
  3. a linear transformation layer for [CLS] token (used in self.forward, as given)
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # bert encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # get word embedding from self.word_embedding
    # todo
    """ <<--- STUDENT IMPLEMENTATION --->> """
    inputs_embeds = self.word_embedding(input_ids)

    # get position index and position embedding from self.pos_embedding
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)
    """ <<--- END OF STUDENT IMPLEMENTATION --->> """

    # get token type ids, since we are not consider token type, just a placeholder
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # add three embeddings together
    embeds = inputs_embeds + tk_type_embeds + pos_embeds

    # layer norm and dropout
    embeds = self.embed_layer_norm(embeds)
    embeds = self.embed_dropout(embeds)

    return embeds

    raise NotImplementedError

  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # get the extended attention mask for self attention
    # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
    # non-padding tokens with 0 and padding tokens with a large negative number 
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
    hidden_layer = []

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # feed the encoding from the last bert_layer to the next
      hidden_states = layer_module(hidden_states, extended_attention_mask)
      hidden_layer.append(hidden_states)

    return hidden_states, hidden_layer

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output, hidden_layer = self.encode(embedding_output, attention_mask=attention_mask)

    # get cls token hidden state
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk, 'hidden_layer': hidden_layer}
