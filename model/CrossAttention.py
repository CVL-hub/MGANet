import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
import math
class CrossAttention(nn.Module):
   def __init__(self, embedding_dim):
       """_summary_

       Args:
           num_iterations: 迭代的次数，文章中默认为3
           num_slots (_type_): 槽的数量，也就是文章中原型的数量
           slot_size (_type_): 槽的大小，也就是channel的大小
           mlp_hidden_size (_type_): GRU的hidden_layer_size
           epsilon (_type_, optional): 小偏移, 防止除零
       Returns:
           _type_: _description_
       """
       super(CrossAttention, self).__init__()

       self.project_k = nn.Linear(embedding_dim, embedding_dim)
       self.project_q = nn.Linear(embedding_dim, embedding_dim)
       self.project_v = nn.Linear(embedding_dim, embedding_dim)

       self.out_proj = nn.Linear(embedding_dim, embedding_dim)

      
   def forward(self, q: Tensor, k: Tensor, v: Tensor):
      b, c, h, w = q.shape
      x = q

      q = q.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
      k = k.reshape(b, -1, c)
      v = v.reshape(b, -1, c)

      q = self.project_q(q)
      k = self.project_k(k)
      v = self.project_v(v)

      attn = q @ k.permute(0, 2, 1)
      attn = attn / math.sqrt(c)
      attn = torch.softmax(attn, dim=-1)

      out = attn @ v
      out = self.out_proj(out)

      return out.permute(0, 2, 1).contiguous().view(b, -1, h, w) + x
     


    