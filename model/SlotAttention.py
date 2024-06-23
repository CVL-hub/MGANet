import numpy as np
import torch.nn as nn
import torch

class SlotAttention(nn.Module):
   def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size, resolution,
               epsilon=1e-8):
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
       super(SlotAttention, self).__init__()
       self.resolution = resolution
       self.num_iterations = num_iterations
       self.num_slots = num_slots
       self.slot_size = slot_size
       self.mlp_hidden_size = mlp_hidden_size
       self.epsilon = epsilon
       
       self.project_k = nn.Linear(slot_size, slot_size, bias=False)
       self.project_q = nn.Linear(slot_size, slot_size, bias=False)
       self.project_v = nn.Linear(slot_size, slot_size, bias=False)

       self.mlp1 = nn.Sequential(
          nn.Linear(self.slot_size, self.slot_size),
          nn.ReLU(inplace=True),
          nn.Linear(self.slot_size, self.slot_size))
       
       self.mlp2 = nn.Sequential(
          nn.Linear(self.slot_size, self.mlp_hidden_size),
          nn.ReLU(inplace=True),
          nn.Linear(self.mlp_hidden_size, self.slot_size))
       
       self.gru = nn.GRUCell(self.slot_size, self.slot_size)
   def forward(self, inputs, pro):
      # inputs has shape [B,S,H,W]
      # pro has shape [B, n, S, 1 ,1]
      b, s, h, w = inputs.shape
      pro = torch.reshape(pro, shape=(b, -1, s))
      pro = self.mlp1(pro)
      k = self.project_k(pro)
      v = self.project_k(pro)
      slots = inputs.view(b, -1, s).contiguous()

      for _ in range(self.num_iterations):
         slots_prev = slots
         q = self.project_q(slots)
         q = q * (self.slot_size ** -0.5)
         attn_logits = torch.einsum('bid,bjd->bij',k, q) # b x n x k
         attn = nn.functional.softmax(attn_logits, -1)
         attn = attn + self.epsilon

         attn = attn / torch.sum(attn, 1, keepdim=True)

         updates = torch.einsum('bij,bid->bjd', attn, v)

         slots = self.gru(
            updates.reshape(-1, s),
            slots_prev.reshape(-1, s)
         )
         slots = slots.reshape(b, -1, s)
         slots = slots + self.mlp2(slots)
      return slots.view(b, s, h, w).contiguous()


    