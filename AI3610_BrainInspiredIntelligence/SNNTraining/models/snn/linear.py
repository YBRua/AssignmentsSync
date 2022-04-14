import torch
import torch.nn as nn
import snntorch as snn


class SNNLinear(nn.Module):
    def __init__(
            self,
            input_dims: int,
            output_dims: int,
            beta: float,
            spike_grad,
            steps):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, output_dims)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.steps = steps

    def forward(self, x, rate_encoding=False):
        spk_rec = []
        mem_rec = []

        mem1 = self.lif1.init_leaky()

        if rate_encoding:
            for step in range(self.steps):
                xt = x[step]
                spk1, mem1 = self.lif1(self.fc1(xt), mem1)
                spk_rec.append(spk1)
                mem_rec.append(mem1)
        else:
            for step in range(self.steps):
                spk1, mem1 = self.lif1(self.fc1(x), mem1)
                spk_rec.append(spk1)
                mem_rec.append(mem1)

        return torch.stack(spk_rec), torch.stack(mem_rec)
