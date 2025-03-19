import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

import config
model_config = config.model_config

class Net(nn.Module):
    def __init__(self , model_raw_large, model_raw_small):
        super(Net, self).__init__()
        self.use_layer_large = model_raw_large
        self.use_layer_small = model_raw_small
        
        self.fc_large_1 = nn.Linear(1000, 256)
        self.fc_large_2 = nn.Linear(256, 64)
        
        self.fc_small_1 = nn.Linear(1000, 256)
        self.fc_small_2 = nn.Linear(256, 64)
        
        self.fc1 = nn.Linear(64*5, 64)
        self.fc2 = nn.Linear(64, 2)
        
        self.dropout = nn.Dropout(model_config['dropout'])
        self.transform = transforms.Resize(model_config['patch_size'])
        
    def forward(self, x):
        x_t, x_b = torch.chunk(x, 2, axis=2)
        x_tl, x_tr = torch.chunk(x_t, 2, axis=3)
        x_bl, x_br = torch.chunk(x_b, 2, axis=3)
        x = self.transform(x)

        x = self.use_layer_large(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_large_1(x))
        x = self.dropout(x)
        x = F.relu(self.fc_large_2(x))
        x = self.dropout(x)
        
        x_tl = self.use_layer_small(x_tl)
        x_tl = x_tl.view(x_tl.size(0), -1)
        x_tl = F.relu(self.fc_small_1(x_tl))
        x_tl = self.dropout(x_tl)
        x_tl = F.relu(self.fc_small_2(x_tl))
        x_tl = self.dropout(x_tl)
        
        x_tr = self.use_layer_small(x_tr)
        x_tr = x_tr.view(x_tr.size(0), -1)
        x_tr = F.relu(self.fc_small_1(x_tr))
        x_tr = self.dropout(x_tr)
        x_tr = F.relu(self.fc_small_2(x_tr))
        x_tr = self.dropout(x_tr)
        
        x_bl = self.use_layer_small(x_bl)
        x_bl = x_bl.view(x_bl.size(0), -1)
        x_bl = F.relu(self.fc_small_1(x_bl))
        x_bl = self.dropout(x_bl)
        x_bl = F.relu(self.fc_small_2(x_bl))
        x_bl = self.dropout(x_bl)
        
        x_br = self.use_layer_small(x_br)
        x_br = x_br.view(x_br.size(0), -1)
        x_br = F.relu(self.fc_small_1(x_br))
        x_br = self.dropout(x_br)
        x_br = F.relu(self.fc_small_2(x_br))
        x_br = self.dropout(x_br)
        
        embed = torch.cat((x, x_tl, x_tr, x_bl, x_br), 1)
        del x_tl, x_tr, x_bl, x_br, x
        x = F.relu(self.fc1(embed))
        x = self.dropout(x)
        x = self.fc2(x)
        return x, embed