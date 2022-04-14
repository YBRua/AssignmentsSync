import torch
import torch.nn as nn
import snntorch as snn


class SNNConvNet(nn.Module):
    def __init__(
            self,
            beta: float,
            spike_grad,
            steps):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc1 = nn.Linear(16 * 4 * 4, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.steps = steps

    def forward(self, x, rate_encoding=False):
        spk_rec = []
        mem_rec = []

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        if rate_encoding:
            for step in range(self.steps):
                xt = x[step]
                spk1, mem1 = self.lif1(self.pool1(self.conv1(xt)), mem1)
                spk2, mem2 = self.lif2(self.pool2(self.conv2(spk1)), mem2)
                flt = spk2.view(-1, 16 * 4 * 4)  # B, 16*5*5
                spk3, mem3 = self.lif3(self.fc1(flt), mem3)
                spk_rec.append(spk3)
                mem_rec.append(mem3)

        else:
            for step in range(self.steps):
                spk1, mem1 = self.lif1(self.pool1(self.conv1(x)), mem1)
                spk2, mem2 = self.lif2(self.pool2(self.conv2(spk1)), mem2)
                flt = spk2.view(-1, 16 * 4 * 4)  # B, 16*5*5
                spk3, mem3 = self.lif3(self.fc1(flt), mem3)
                spk_rec.append(spk3)
                mem_rec.append(mem3)

        return torch.stack(spk_rec), torch.stack(mem_rec)
