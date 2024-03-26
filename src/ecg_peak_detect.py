import os
import pickle

import tqdm
import torch as tc
import numpy as np

from datautils import BaseECGDataset
from models.deeplearn import PeakDetector
from evaluators import peak_evaluate as evaluate


CUDA = 0
SEED = 42

MODEL_CONFIG = {
    "hidden_size": 128,
    "deduction": 8,
    "dropout": 0.15,
    "device": "cuda:%s" % CUDA if tc.cuda.is_available() else "cpu",
    }


TRAIN_CONFIG = {
    "use_amp": True,
    "learn_rate": 3e-3,
    "batch_size": 64,
    "workers": 2,
    "num_epochs": 10,
    "decay_rate": 0.1,
    "decay_tol": 2,
    "early_stop": 2,
    "weight_decay": 1e-2,
    "optimizer": "AdamW",
    "max_norm": 1.0,
    "frozen_train_size": 30,
    "continue_ckpt": False,
    "log_frequency": 656
    }


class ECGSignal(BaseECGDataset):
    def __init__(self, *args, **kwrds):
        BaseECGDataset.__init__(self, *args, **kwrds)

    def __getitem__(self, idx):
        data = self._read_data(idx)
        peaks = [0] * len(data["data"])
        for pos in data["pos"]:
            peaks[pos] = 1.0
        return (tc.tensor(data["data"]).float(),
                tc.tensor(peaks).long())   # Signal --> Peak



def fit(datas, model, optimizer, learn_rate, batch_size, num_epochs, max_norm, log_frequency, frozen_train_size, decay_rate, decay_tol, early_stop, weight_decay, use_amp, workers, continue_ckpt):
    optimizer = getattr(tc.optim, optimizer)(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = tc.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)
    progress = tqdm.tqdm(total=len(datas[0]) // batch_size * num_epochs, desc="Training:")
    scaler = tc.cuda.amp.GradScaler(enabled=use_amp)
    
    train = tc.utils.data.DataLoader(datas[0], batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    valid = tc.utils.data.DataLoader(datas[1], batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test = tc.utils.data.DataLoader(datas[2], batch_size=batch_size, shuffle=False,  num_workers=2, pin_memory=True)

    os.makedirs("outputs/", exist_ok=True)
    epoch, total_tol, current_tol = 0, 0, 0
    frozen_train = []
    best_model, best_score = "outputs/seed%d_%s.pth" % (SEED, model.version), -float("inf")
    idx = 0
    if continue_ckpt:
        if os.path.isfile(best_model + ".trainer"):
            with open(best_model + ".trainer", "rb") as f:
                info = pickle.load(f)
                assert info["best_model"] == best_model
            model.load_state_dict(tc.load(best_model))
            epoch, idx = info["epoch"], info["index"]
            best_score, total_tol = info["best_score"], info["total_tol"]
            print("Reloaded Model Success!")
        else:
            print("Ignore reloaded model: %s" % best_model)
        
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

            if idx == 1 or idx % log_frequency == 0:
                scores = evaluate(model, frozen_train)
                print("Train Step=%d | %s" % (idx, "|".join("%s: %.4f" % _ for _ in scores.items())))
                
        scores = evaluate(model, valid)
        print("Valid Epoch=%d | %s" % (epoch, "|".join("%s: %.4f" % _ for _ in scores.items())))
        if scores["F1"] > best_score:
            current_tol = 0
            best_score = scores["F1"]
            tc.save(model.state_dict(), best_model)
            with open(best_model + ".trainer", "wb") as f:
                pickle.dump({"epoch": epoch, "index": idx,
                          "best_model": best_model,
                          "best_score": best_score,
                          "total_tol": total_tol},
                          f)
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
    scores = evaluate(model, test)
    print("Test Epoch=%d | %s" % (epoch, "|".join("%s: %.4f" % _ for _ in scores.items())))
    return model




if __name__ == "__main__":
    root = "datasets/mit-bih/"
    datas = [ECGSignal(root + "/" + subset, subset, train=0.85, valid=0.05) for subset in ["train", "valid", "test"]]
    model = PeakDetector(**MODEL_CONFIG)
    model = fit(datas, model, **TRAIN_CONFIG)
        
