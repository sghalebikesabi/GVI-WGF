import torch
import math


class RBF(torch.nn.Module):
    def __init__(self, sigma):
        super(RBF, self).__init__()

        self.sigma = sigma
        self.timer = 0

    def median(self, tensor):
        tensor = tensor.flatten().sort()[0]
        length = tensor.shape[0]

        if length % 2 == 0:
            szh = length // 2
            kth = [szh - 1, szh]
        else:
            kth = [(length - 1) // 2]
        return tensor[kth].mean()

    def forward(self, X, Y):

        # XX = torch.einsum("zyij,zyij->zyi", X, X)
        # XY = torch.einsum("zyij,xwij->zyi", X, Y)
        # YY = torch.einsum("zyij,zyij->zyi", Y, Y)

        # dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
        dnorm2 = (X - Y).pow(2).mean(-1).mean(-1)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma == "rolling_median":
            sigma = self.median(dnorm2.detach()) / (
                2 * torch.tensor(math.log(X.size(0) + 1))
            )
        elif self.sigma == "median":
            sigma = self.median(dnorm2.detach()) / (
                2 * torch.tensor(math.log(X.size(0) + 1))
            )
            if self.timer < 5:
                self.timer += 1
            else:
                self.sigma = sigma
        elif self.sigma == "half_median":
            sigma = self.median(dnorm2.detach()) / 2 / (
                2 * torch.tensor(math.log(X.size(0) + 1))
            )
            if self.timer < 5:
                self.timer += 1
            else:
                self.sigma = sigma
        elif self.sigma == "twice_median":
            sigma = self.median(dnorm2.detach()) * 2 / (
                2 * torch.tensor(math.log(X.size(0) + 1))
            )
            if self.timer < 5:
                self.timer += 1
            else:
                self.sigma = sigma
        else:
            sigma = self.sigma

        gamma = 1.0 / (2 * sigma)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY.sum() / K_XY.size(1)
