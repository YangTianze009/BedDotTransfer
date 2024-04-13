import os
import pickle
import random
import numpy as np
import torch as tc
import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from models.deeplearn import PeakDetector
from evaluators import peak_evaluate as evaluate


CUDA = 1
SEED = 42

MODEL_CONFIG = {
    "hidden_size": 128,
    "deduction": 8,
    "dropout": 0.15,
    "device": "cuda:%s" % CUDA if tc.cuda.is_available() else "cpu",
}


TRAIN_CONFIG = {
    "use_amp": False,
    "learn_rate": 3e-3,
    "batch_size": 64,
    "workers": 0,
    "num_epochs": 30,
    "decay_rate": 0.1,
    "decay_tol": 2,
    "early_stop": 2,
    "weight_decay": 0.0,
    "optimizer": "AdamW",
    "max_norm": 1.0,
    "frozen_train_size": 30,
    "log_frequency": 1000,
}


def plotting_signal(s, y, p, name="none"):
    assert len(s) == len(p) == len(y)
    # between = np.arange(s.min(), s.max()+1)
    plt.plot(range(len(s)), s, color="black")
    if y:
        between = np.arange(0, s.max() + 1)
        for idx, val in enumerate(y):
            if val == 1:
                plt.plot([idx] * len(between), between, color="red", alpha=0.6)
    if p:
        between = np.arange(s.min() - 1, 0)
        for idx, val in enumerate(p):
            if val == 1:
                plt.plot([idx] * len(between), between, color="blue", alpha=0.6)
    # plt.title("Noise Level=%s" % name)
    plt.savefig("figures/%s.pdf" % name)
    plt.clf()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tc.manual_seed(seed)
    tc.cuda.manual_seed_all(seed)


class DataReader(tc.utils.data.Dataset):
    def __init__(self, fpath):
        tc.utils.data.Dataset.__init__(self)
        self._data = np.load(fpath)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        x = self._data[idx][:1000]
        x = (x - x.mean()) / x.std()
        y = [0] * len(x)
        for idx in find_peaks(x, height=0.45, prominence=0.3, distance=25)[0].tolist():
            y[idx] = 1
        # plotting_signal(x, y)
        return (tc.tensor(x, dtype=tc.float32), tc.tensor(y, dtype=tc.int64))


def fit(
    datas,
    model,
    noise,
    optimizer,
    learn_rate,
    batch_size,
    num_epochs,
    max_norm,
    log_frequency,
    frozen_train_size,
    decay_rate,
    decay_tol,
    early_stop,
    weight_decay,
    use_amp,
    workers,
):
    optimizer = getattr(tc.optim, optimizer)(
        model.parameters(), lr=learn_rate, weight_decay=weight_decay
    )
    scheduler = tc.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)
    progress = tqdm.tqdm(
        total=len(datas[0]) // batch_size * num_epochs, desc="Training:"
    )
    scaler = tc.cuda.amp.GradScaler(enabled=use_amp)

    train = tc.utils.data.DataLoader(
        datas[0],
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    valid = tc.utils.data.DataLoader(
        datas[1], batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test1 = tc.utils.data.DataLoader(
        datas[2], batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test2 = tc.utils.data.DataLoader(
        datas[3], batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    os.makedirs("outputs/", exist_ok=True)
    idx, epoch, total_tol, current_tol = 0, 0, 0, 0
    frozen_train = []
    best_model, best_score = "outputs/seed%d_%s.pth" % (SEED, model.version), -float(
        "inf"
    )

    for epoch in range(epoch + 1, num_epochs + 1):
        model.train()

        for batch in train:
            if idx <= frozen_train_size:
                frozen_train.append(batch)
            if batch[0].shape[0] != batch_size:
                continue
            optimizer.zero_grad()
            loss = model.compute_loss(batch[1], model(batch[0]))
            loss.backward()
            tc.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            progress.update(1)
            idx += 1

        ##            if idx == frozen_train_size or idx % log_frequency == 0:
        ##                scores = evaluate(model, frozen_train)
        ##                print("Train Step=%d | %s" % (idx, "|".join("%s: %.4f" % _ for _ in scores.items())))

        scores = evaluate(model, valid)
        # p = tc.argmax(model(batch[0]), -1)[0].cpu().detach().numpy()
        # plotting_signal(batch[0][0].cpu().numpy(), p.tolist())
        print(
            "Valid Epoch=%d | %s"
            % (epoch, "|".join("%s: %.4f" % _ for _ in scores.items()))
        )
        if scores["F1"] > best_score:
            current_tol = 0
            best_score = scores["F1"]
            tc.save(model.state_dict(), best_model)
            with open(best_model + ".trainer", "wb") as f:
                pickle.dump(
                    {
                        "epoch": epoch,
                        "index": idx,
                        "best_model": best_model,
                        "best_score": best_score,
                        "total_tol": total_tol,
                    },
                    f,
                )
        else:
            current_tol += 1
            if current_tol == decay_tol:
                print("Reducing learning rate by %.4f" % decay_rate)
                scheduler.step()
                model.load_state_dict(tc.load(best_model))
                current_tol = 0
                total_tol += 1
            if total_tol == early_stop + 1:
                print("Early stop at epoch %s with F1=%.4f" % (epoch, scores["F1"]))
                break
    progress.close()

    print("Reload model:" + best_model)
    model.load_state_dict(tc.load(best_model))
    scores = evaluate(model, test1)
    model.eval()
    for x, y in test1:
        p = tc.argmax(model(x), -1)[0].cpu().detach().tolist()
        plotting_signal(x[0].cpu().numpy(), y.cpu()[0].tolist(), p, "Stable" + noise)
        break
    print(
        "Test Noise=%s | Stable | Epoch=%d | %s"
        % (noise, epoch, "|".join("%s: %.4f" % _ for _ in scores.items()))
    )

    scores = evaluate(model, test2)
    model.eval()
    for x, y in test2:
        p = tc.argmax(model(x), -1)[0].cpu().detach().tolist()
        plotting_signal(
            x[0].cpu().numpy(), y.cpu()[0].tolist(), p, "Real_BSG_Test" + noise
        )
        break
    print(
        "Test Noise=%s | Real_BSG_Test | Epoch=%d | %s"
        % (noise, epoch, "|".join("%s: %.4f" % _ for _ in scores.items()))
    )
    return model


if __name__ == "__main__":
    window = "0.040"
    for noisy in ["00", "02", "04", "06", "08", "10"]:
        set_seed(SEED)
        # noisy = "04"
        print(
            f"Noise Level: {noisy} | Window Size: {window} | find_peak() ARGS: height=0.45, prominence=0.3, distance=25"
        )
        train = DataReader(
            f"datasets/stable_noise{noisy}/envelope_data/{window.replace('.','_')}/extracted_envelope_data_10k.npy"
        )
        valid = DataReader(
            f"datasets/stable_noise{noisy}/envelope_data/{window.replace('.','_')}/extracted_envelope_data_2k.npy"
        )
        test1 = DataReader(
            f"datasets/stable_noise{noisy}/envelope_data/{window.replace('.','_')}/extracted_envelope_data_5k.npy"
        )
        test2 = DataReader(
            f"datasets/real_BSG_data/envelope_data/{window.replace('.','_')}/envelope_day_data_test.npy"
        )
        model = PeakDetector(**MODEL_CONFIG)
        fit((train, valid, test1, test2), model, noisy, **TRAIN_CONFIG)
