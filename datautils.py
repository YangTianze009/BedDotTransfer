import os
import bisect
import pickle
import random
import shutil
import collections

import wfdb
import torch as tc
import numpy as np
import wfdb.processing


class BaseECGDataset(tc.utils.data.Dataset):
    def __init__(
        self, folder, subset="full", padding=10, train=0.7, valid=0.1, seed=42
    ):
        tc.utils.data.Dataset.__init__(self)
        self._root = folder
        self._padding = 10
        self._files = self._load_files(subset, train, valid, seed)

    def _load_files(self, subset, train, valid, seed):
        assert subset in {"full", "train", "valid", "test"}
        assert 0.0 <= train + valid <= 1.0
        files = [_ for _ in os.listdir(self._root) if _.endswith(".pkl")]
        files.sort()
        if subset == "full" or self._root.endswith(subset):
            return files

        random.seed(seed)
        random.shuffle(files)

        train = int(train * len(files))
        valid = int(valid * len(files)) + train
        if subset == "train":
            return files[: train + 1]
        if subset == "valid":
            return files[train + 1 : valid + 1]
        return files[valid + 1 :]

    def _read_data(self, idx):
        return pickle.load(open(self._root + "/" + self._files[idx], "rb"))

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        return self._read_data(idx)


def read(file_name, target_fs=100, origin_fs=360):
    file_name = file_name.rsplit(".", 1)[0]
    record = wfdb.rdrecord(file_name).p_signal[:, 0]
    atr_ann = wfdb.rdann(file_name, "atr")
    try:
        qrs_ann = wfdb.rdann(file_name, "qrsc")
    except:
        qrs_ann = wfdb.rdann(file_name, "qrs")
    record, qrs_ann = wfdb.processing.resample_singlechan(
        record, qrs_ann, fs=origin_fs, fs_target=target_fs
    )
    ann_sample = list(map(lambda x: int(x * target_fs / origin_fs), atr_ann.sample))
    aux_note = list(
        map(lambda x: 1 if x == "(AFIB" or x == "(AFL" else 0, atr_ann.aux_note)
    )
    return {
        "record": record,
        "sample": ann_sample,
        "annote": aux_note,
        "peaks": qrs_ann.sample,
    }


def normalize(record):
    return (record - record.mean()) / record.std()


def truncate(seq, idx, sigma=0.4):
    return int(sigma * seq[idx - 1] + (1.0 - sigma) * seq[idx])


def segment(record, sample, annote, peaks, idx=10, freq=100, duration=10.0):
    seg_size = int(freq * duration)
    head = truncate(peaks, idx)
    tail = truncate(peaks, -idx)
    seg_head, seg_tail = head, head + seg_size
    num_segs, num_lbls = 0, 0
    lbl_count = collections.Counter()
    while seg_tail < tail or idx == len(peaks) - 1:
        volumns = []
        positions = []
        labels = []
        bboxs = []
        old_idx = idx
        for idx, peak in enumerate(peaks[idx:], idx):
            if not seg_head < peak < seg_tail:
                break
            win_head = max(truncate(peaks, idx), seg_head)
            win_tail = min(truncate(peaks, idx + 1), seg_tail)
            if peak - win_head > freq:
                win_head = peak - 100
            if win_tail - peak > 200:
                win_tail = peak + 200
            win_head = (win_head - seg_head) / seg_size
            win_tail = 1.0 + (win_tail - seg_tail) / seg_size
            bboxs.append((win_head, win_tail))
            labels.append(annote[bisect.bisect_left(sample, peak) - 1])
            positions.append(peak - seg_head)

            volumns.append(record[peak])

        if len(positions) > 1:
            num_segs += 1
            num_lbls += len(labels)
            assert all(map(lambda x: 0.0 <= x <= seg_size, positions))
            seg_label = 1 if sum(labels) * 2 >= len(labels) else 0
            lbl_count[seg_label] += 1
            yield {
                "label": seg_label,
                "data": normalize(record[seg_head : seg_tail + 1]),
                "pos": positions,
                "vols": volumns,
                "bboxs": bboxs,
                "labels": labels,
            }

        if old_idx == idx:
            break
        seg_head = truncate(peaks, idx)
        seg_tail = seg_head + seg_size

    print(
        "Totally detect %s segments with %s labels." % (num_segs, num_lbls), lbl_count
    )


def split(folder, train=0.7, valid=0.1, seed=42):
    assert 0 <= train + valid <= 1.0
    assert os.path.exists(folder + "/full")
    random.seed(seed)
    files = [_ for _ in os.listdir(folder + "/full") if _.endswith(".pkl")]
    files.sort()
    random.shuffle(files)
    train = int(train * len(files))
    valid = train + int(valid * len(files))
    for name in ["train", "valid", "test"]:
        os.makedirs(folder + "/%s" % name, exist_ok=True)
    for file in files[: train + 1]:
        shutil.copy(folder + "/full/" + file, folder + "/train/" + file)
    for file in files[train + 1 : valid + 1]:
        shutil.copy(folder + "/full/" + file, folder + "/valid/" + file)
    for file in files[valid + 1 :]:
        shutil.copy(folder + "/full/" + file, folder + "/test/" + file)


def create_dataset_on_disk(root, records, subset="full"):
    label_counts = collections.Counter()
    for file in records:
        data = read(root + "/" + file)
        fpath = root + "/%s" % subset
        os.makedirs(fpath, exist_ok=True)
        for idx, sample in enumerate(segment(**data)):
            label_counts[sample["label"]] += 1
            pickle.dump(
                sample, open(fpath + r"/u%s_i%s.pkl" % (file, idx), "wb"), protocol=2
            )
    print("%s Size: %s | Label Distribution: %s" % (subset, records, label_counts))


if __name__ == "__main__":
    root = r"datasets\mit-bih"
    records = [
        "04015",
        "04043",
        "04048",
        "04126",
        "04746",
        "04908",
        "04936",
        "05091",
        "05121",
        "05261",
        "06426",
        "06453",
        "06995",
        "07162",
        "07859",
        "07910",
        "08215",
        "08219",
        "08378",
        "08405",
        "08434",
        "08455",
    ]
    train_size = int(0.8 * len(records))
    valid_size = int(0.9 * len(records))
    random.seed(2023)
    random.shuffle(records)
    create_dataset_on_disk(root, records[:train_size], "train")
    create_dataset_on_disk(root, records[train_size:valid_size], "valid")
    create_dataset_on_disk(root, records[valid_size:], "test")
