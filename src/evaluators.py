import torch as tc
import numpy as np
import tqdm
from sklearn import metrics


def label_evaluate(model, dataset, desc="Label Evaluate:"):
    model.eval()
    reals, preds = [], []
    with tc.no_grad():
        for x, y in tqdm.tqdm(dataset, desc=desc):
            assert isinstance(x, tc.Tensor) and isinstance(y, tc.Tensor)
            assert len(y) == len(x), "evaluting the segment classification results."
            p = model(x)
            if len(p.shape) == 2:
                p = tc.argmax(p, -1)
            assert len(y) == len(p)
            reals.extend(y.cpu().tolist())
            preds.extend(p.cpu().tolist())
    model.train()
    return {"ACC": metrics.accuracy_score(reals, preds),
            "F1": metrics.f1_score(reals, preds)}


def peak_evaluate(model, dataset, window=2, desc="Peak Evaluate:"):
    model.eval()
    reals, preds = [], []
    with tc.no_grad():
        for x, y in tqdm.tqdm(dataset, desc=desc):
            assert isinstance(x, tc.Tensor) and isinstance(y, tc.Tensor)
            assert x.shape == y.shape, "evaluating peak detection results"
            s = y.shape[1]
            y = y.cpu().tolist()
            p = model(x)
            if len(p.shape) == 3:
                p = tc.argmax(p, -1)
            p = p.cpu().tolist()
            for r, (ry, rp) in enumerate(zip(y, p)):
                assert len(rp) == len(ry) == s
                for i, val in enumerate(ry):
                    if val == 1:
                        left, right = max(0, i - window), min(s, 1 + i + window)
                        label = min(1.0, sum(rp[left:right]))
                        rp[left:right] = [0] * (right - left)
                        rp[i] = label
                reals.extend(ry)
                preds.extend(rp)
    model.train()
    return {"ACC": metrics.accuracy_score(reals, preds),
            "F1": metrics.f1_score(reals, preds)}


                
            
                        
                        
                
            
            
            
            
            
            
            
