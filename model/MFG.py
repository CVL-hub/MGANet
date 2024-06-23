import numpy as np
import torch.nn as nn
import torch

def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0)
  grid = grid.astype(np.float32)
  return np.concatenate([grid, 1.0 - grid], axis=-1)

def spatial_broadcast(slots, resolution):
  """Broadcast slot features to a 2D grid and collapse slot dimension."""
  # `slots` has shape: [batch_size, num_slots, slot_size].
  slots = torch.reshape(slots, (-1, slots.shape[-1]))[:, None, None, :] #batch_size*num_slots, 1, 1,slot_size 
  grid = slots.repeat(1, resolution[0], resolution[1], 1)
  # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
  return grid

class MultiFeatureGrouping(nn.Module):
   def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size, resolution):
      super(MultiFeatureGrouping, self).__init__()
      
      self.resolution = resolution # [W, H]
      self.num_slots = num_slots
      self.slot_size = slot_size
      
      self.slot_attention1 = SlotAttention(num_iterations, num_slots[0], slot_size, mlp_hidden_size, resolution)
      self.slot_attention2 = SlotAttention(num_iterations, num_slots[1], slot_size, mlp_hidden_size, resolution)
      self.conv = nn.Conv2d(2 * slot_size, slot_size, kernel_size=1, bias=True)
   def forward(self, image):
      x = image
      slots1 = self.slot_attention1(image) # 
      slots2 = self.slot_attention2(slots1 + x)
      alpha_g = nn.Sigmoid()(self.conv(torch.cat([slots1, slots2], dim=1)))
      return x + alpha_g * slots1 + (1 - alpha_g)*slots2

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

       self.norm_inputs = nn.LayerNorm(slot_size)
       self.norm_slots = nn.LayerNorm(slot_size)
       self.norm_mlp = nn.LayerNorm(slot_size)

       self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_size))
       #nn.init.xavier_uniform_(self.slots_mu)
       self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_size))
       nn.init.xavier_uniform_(self.slots_log_sigma)
       
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
       self.encoder_pos = SoftPositionEmbedding(slot_size, self.resolution)
       self.decoder_pos = SoftPositionEmbedding(slot_size, self.resolution)
       self.layer_norm = nn.LayerNorm(slot_size)
       self.conv = nn.Conv2d(slot_size, slot_size // num_slots, kernel_size=1, bias=False)
   def forward(self, inputs):
      # inputs has shape [B,S,H,W]
      b, s, h, w, device, dtype = *inputs.shape, inputs.device, inputs.dtype
      inputs = torch.reshape(inputs, shape=(b, w, h, s))
      inputs = self.encoder_pos(inputs)
      inputs = torch.reshape(inputs, shape=(b, w*h, s))
      inputs = self.mlp1(self.layer_norm(inputs))

      inputs = self.norm_inputs(inputs)
      k = self.project_k(inputs)
      v = self.project_k(inputs)
      mu = self.slots_mu.expand(b, self.num_slots, self.slot_size)
      sigma = self.slots_log_sigma.exp().expand(b, self.num_slots, self.slot_size)
      slots = mu + sigma * torch.randn(b, self.num_slots, self.slot_size, device=device, dtype=dtype)

      for _ in range(self.num_iterations):
         slots_prev = slots
         slots = self.norm_slots(slots)

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
         slots = slots + self.mlp2(self.norm_mlp(slots))
      slots_broder = spatial_broadcast(slots, self.resolution) # [B*num_slots, w, h, slot_size]
      slots_pos = self.decoder_pos(slots_broder) #[B*num_slots, w, h, slot_size] 解码位置编码考虑
      slots_pos = torch.reshape(slots_pos, (-1, s, h, w))
      slots_down = self.conv(slots_pos) # [B*num_slots, s//num_slots, h, w]
      list1 = []
      for i in range(self.num_slots):
         y = slots_down[b*i:b*(i+1)]
         list1.append(y)
      slots_final1 = torch.cat(list1, dim = 1)
      return slots_final1 # B X C X H X W 

class SoftPositionEmbedding(nn.Module):
    def __init__(self, hidden_size, resolution):
        """_summary_

        Args:
            hidden_size: 隐藏层的大小，即通道的数量
            resolution: 图像的宽和高
        """
        super(SoftPositionEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.resolution = resolution
        self.dense = nn.Sequential(
           nn.Linear(4, 64, bias=True),
           nn.ReLU(),
           nn.Linear(64, hidden_size)
        )
        self.grid = build_grid(self.resolution)

    def forward(self, inputs):
       device = inputs.device
       dtype = inputs.dtype
       return inputs + self.dense(torch.from_numpy(self.grid).to(device))    

    