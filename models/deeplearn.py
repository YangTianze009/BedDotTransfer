import torch as tc
import numpy as np


class MaxMeanPooling(tc.nn.Module):
    def __init__(self):
        tc.nn.Module.__init__(self)

    def forward(self, x):
        assert len(x.shape) == 3
        m = tc.where(tc.abs(x).sum(axis=-1) > 0, 1.0, 0.0)
        x_avg = x.sum(axis=1) / m.sum(axis=-1).unsqueeze(1)
        x_max = (x * m.unsqueeze(-1)).max(axis=1).values
        return tc.cat([x_avg, x_max], -1)


class SENet(tc.nn.Module):
    def __init__(self, input_dim, deduction, dropout):
        tc.nn.Module.__init__(self)
        self._gate = tc.nn.Sequential(
            tc.nn.Dropout(dropout),
            tc.nn.Linear(input_dim, input_dim // deduction),
            tc.nn.BatchNorm1d(input_dim // deduction),
            tc.nn.LeakyReLU(),
            tc.nn.Linear(input_dim // deduction, input_dim),
            tc.nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.squeeze()
        return x * self._gate(x)


class PeakClassifier(tc.nn.Module):

    version = "classifier"

    def __init__(self, kernel_size, hidden_size, deduction, dropout, device="cpu"):
        tc.nn.Module.__init__(self)
        input_size = kernel_size**2 + 1
        self._cnn = tc.nn.Sequential(
            tc.nn.Conv1d(1, hidden_size, kernel_size, padding=0), tc.nn.Dropout(0.15)
        )
        self._classifier = tc.nn.Sequential(
            MaxMeanPooling(),
            SENet(hidden_size * 2, deduction, dropout),
            tc.nn.Linear(hidden_size * 2, 2),
        )
        self._lossfn = tc.nn.CrossEntropyLoss()
        self.to(device)
        self.device = device

    def forward(self, x, *args, **kwrds):
        x = x.to(self.device).diff()
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        if len(x.shape) > 2 and x.shape[1] != 1:
            x = x.permute(0, 2, 1)
        assert len(x.shape) == 3 and x.shape[1] == 1
        z = self._cnn(x).transpose(2, 1)
        return tc.softmax(self._classifier(z), -1)

    def compute_loss(self, real, pred):
        return self._lossfn(pred.to(self.device), real.to(self.device))


class PeakDetector(tc.nn.Module):

    version = "detector"

    def __init__(self, hidden_size, deduction, dropout, device="cpu"):
        tc.nn.Module.__init__(self)
        self._rnn = tc.nn.GRU(1, hidden_size, bidirectional=True, batch_first=True)
        self._classifier = tc.nn.Sequential(
            tc.nn.Dropout(dropout),
            tc.nn.Linear(hidden_size * 2, 2),
        )
        self._lossfn = FocalLoss(10.0, [0.1, 0.9])
        # self._lossfn = tc.nn.CrossEntropyLoss()
        self.device = device
        self.to(device)

    def forward(self, x):
        assert len(x.shape) == 2
        x = x.diff().unsqueeze(-1).to(self.device)
        h = self._rnn(x.to(tc.float32))[0]
        bs, ts, dim = h.shape
        y = self._classifier(h.reshape(-1, dim))
        y = tc.softmax(y, -1).reshape(bs, ts, 2)
        return tc.cat([y[:, :1, :], y], axis=1)

    def compute_loss(self, real, pred):
        return self._lossfn(
            pred.reshape(-1, 2).to(self.device), real.flatten().to(self.device)
        )


class FocalLoss(tc.nn.Module):
    def __init__(self, gamma=0.5, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = tc.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = tc.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = tc.nn.functional.log_softmax(input, -1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class ECGClassifier(tc.nn.Module):

    version = "20230328"

    def __init__(self, kernel_size, hidden_size, deduction, dropout, device="cpu"):
        tc.nn.Module.__init__(self)
        self._peak_detector = PeakDetector(4, 4, 0.1, device)
        self._peak_classifier = PeakClassifier(
            kernel_size, hidden_size, deduction, dropout, device
        )
        self._lossfn = FocalLoss(2.0)
        self._padding = 30
        self.to(device)
        self.device = device

    def forward(self, x):
        assert len(x.shape) == 2
        p = self._peak_detector(x)
        x = p / x.shape[1]
        # x = tc.tensor(np.array(p)) / x.shape[1]
        return self._peak_classifier(x)

    def compute_loss(self, real, pred):
        return self._lossfn(pred.to(self.device), real.to(self.device))
