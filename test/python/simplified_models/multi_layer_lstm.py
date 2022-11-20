import torch.nn as nn 
import torch.nn.functional as F
import torch

class lstm_cell(nn.Module):
    def __init__(self, hidden_dim):
        super(lstm_cell, self).__init__()
        
        self.wi = nn.Linear(hidden_dim * 2, hidden_dim)
        self.wf = nn.Linear(hidden_dim * 2, hidden_dim)
        self.wg = nn.Linear(hidden_dim * 2, hidden_dim)
        self.wo = nn.Linear(hidden_dim * 2, hidden_dim)
        self.h0 = torch.randn(8,hidden_dim)
        self.c0 = torch.randn(8,hidden_dim)
    def init_state(self):
        return self.h0, self.c0
    def forward(self, h, c, x):
        x = torch.concat([h,x], axis = -1)
        i = self.wi(x)
        f = self.wf(x)
        g = self.wg(x)
        o = self.wo(x)
        
        c = f * c + i * g 
        h = o * c 
        return [h, c], h
    
class multi_layer_lstm(nn.Module):
    def __init__(self, n_layer, hidden_dim):
        super(multi_layer_lstm, self).__init__()
        self.n_layer = n_layer 
        self.cells = nn.ModuleList(
            [lstm_cell(hidden_dim) for _ in range(n_layer)]
        )
    
    def forward(self, xs):
        with torch.no_grad():
            states = [cell.init_state() for cell in self.cells]
            for l, cell in enumerate(self.cells):
                for idx in range(len(xs)):
                    states[l], xs[idx] = cell(*states[l], xs[idx])
        return states[-1][0]
        
seqlen = 20
hdim = 512
n_layer = 4
model = multi_layer_lstm(n_layer, hdim)

model.eval()

print(model)

xs = [torch.randn(8,hdim) for _ in range(seqlen)]
model(xs)

torch.onnx.export(model, {'xs': xs}, 'multi_layer_lstm.onnx', export_params = True, 
                  input_names = ['xs'], output_names = ['output'])                                