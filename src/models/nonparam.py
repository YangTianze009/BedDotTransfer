import torch as tc
import numpy as np
from scipy.signal import find_peaks_cwt, find_peaks


class PeakClassifier:

    version = "classifier"

    def __init__(self, threshold=0.001, device="cpu"):
        self._threshold = threshold
        self.device = device

    def forward(self, x, *args, **kwrds):
        x = x.to(self.device)
        x = x.diff()  # distances between peaks
        mask = tc.where(x > 0.0, 1, 0.0).detach()
        avg = (x.sum(axis=-1) / mask.sum(axis=-1)).unsqueeze(-1)
        var = (((x - avg) ** 2) * mask).sum(axis=-1) / mask.sum(axis=-1)
        logit = tc.sqrt(var) - self._threshold
        return tc.softmax(tc.stack([-logit, logit]).T, -1)

    def compute_loss(self, real, pred):
        return self._lossfn(pred.to(self.device), real.to(self.device))


class PeakDetector:

    version = "detector"

    def __init__(self, distance=30, device="cpu"):
        self._distance = distance
        self.device = device

    def forward(self, X):

        X = X.cpu().numpy()
        predicts = []
        for x in X:
            peaks = find_peaks(-x, distance=self._distance)[0].tolist()
            pred = np.zeros((len(x), 2))
            pred[:, 0] = 1.0
            for peak in peaks:
                pred[peak, 0] = 0.0
                pred[peak, 1] = 1.0
            predicts.append(pred)
        return tc.tensor(np.array(predicts))

    def compute_loss(self, real, pred):
        return self._lossfn(
            pred.reshape(-1, 2).to(self.device), real.flatten().to(self.device)
        )


class ECGClassifier:

    def __init__(self, distance=30, threshold=0.001, device="cpu"):
        tc.nn.Module.__init__(self)
        self._peak_detector = PeakDetector(distance)
        self._peak_classifier = PeakClassifier(threshold)
        self._padding = 30
        self.device = device

    def forward(self, x):
        assert len(x.shape) == 2
        p = []
        for sample in x:
            peaks = find_peaks(-sample, distance=30)[0][: self._padding].tolist()
            peaks.extend([0] * (self._padding - len(peaks)))
            p.append(peaks)
        x = tc.tensor(np.array(p)) / x.shape[1]
        return self._peak_classifier(x)

    def compute_loss(self, real, pred):
        return self._lossfn(pred.to(self.device), real.to(self.device))
