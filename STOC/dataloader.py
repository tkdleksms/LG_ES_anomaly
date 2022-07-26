import torch
import pandas as pd
import matplotlib.pyplot as plt

from .dataset import BuildDataset


def get_dataloader(args):
    """
    Return dataloader
    Parameters:
        root_path (str): path of the data
        data_name (str): name of the data
        batch_size (int): batch size
        window_size (int): window size for time series condition
        slide_size (int): moving window size
        normal (bool): normalization
    Returns:
        train/dev/tst dataloader (torch.utils.data.DataLoader):
            output shape - time series: (batch size, window size, num_features)
    """
    # arguments only to use
    root_path = args.root_path
    data_name = args.data_name
    test_ratio = args.test_ratio
    valid_ratio = args.valid_ratio
    normal = args.normal
    batch_size = args.batch_size
    num_features = args.num_features
    make_plot = args.make_plot

    data = pd.read_csv(root_path + data_name)
    test_len = int(test_ratio * data.shape[0])
    train_len = int((1 - valid_ratio - test_ratio) * data.shape[0])

    full = data.copy()
    trn = data.iloc[:train_len, :].copy()
    dev = data.iloc[train_len:-test_len, :].copy()
    tst = data.iloc[
        -test_len:,
    ].copy()

    full_len = len(full)
    trn_len = len(trn)
    dev_len = len(dev)
    tst_len = len(tst)

    # 데이터 정규화
    if normal:
        print("normalizing...")
        trn_max = max(trn.loc[:, "value"])
        trn_min = min(trn.loc[:, "value"])

        full.loc[:, "value"] = (data.loc[:, "value"] - trn_min) / (trn_max - trn_min)
        trn.loc[:, "value"] = (trn.loc[:, "value"] - trn_min) / (trn_max - trn_min)
        dev.loc[:, "value"] = (dev.loc[:, "value"] - trn_min) / (trn_max - trn_min)
        tst.loc[:, "value"] = (tst.loc[:, "value"] - trn_min) / (trn_max - trn_min)

        # normalized_data = pd.concat([trn, dev, tst])
        print("save normalized data")
        full.to_csv(
            f"{root_path}/normalized_{data_name[:-4]}_val_{valid_ratio}_tst_{test_ratio}.csv"
        )

    if make_plot:
        make_matplt(data, data_name, train_len, test_len, None)

    # 시계열 데이터를 window_size만큼 자르고, slide_size씩 이동하여 dataset 구축
    full_dataset = BuildDataset(full, args)
    trn_dataset = BuildDataset(trn, args)
    dev_dataset = BuildDataset(dev, args)
    tst_dataset = BuildDataset(tst, args)

    # dataloader 구축
    full_dataloader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=args.shuffle,
        num_workers=4,
        drop_last=False,
    )
    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size=batch_size,
        shuffle=args.shuffle,
        num_workers=4,
        drop_last=False,
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=args.shuffle,
        num_workers=4,
        drop_last=False,
    )
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset,
        batch_size=batch_size,
        shuffle=args.shuffle,
        num_workers=4,
        drop_last=False,
    )

    return (
        full_len,
        trn_len,
        dev_len,
        tst_len,
        full_dataloader,
        trn_dataloader,
        dev_dataloader,
        tst_dataloader,
    )


def make_matplt(df: pd.DataFrame, data_name, trn_len, tst_len, save_path=None):
    fig, ax = plt.subplots(figsize=(20, 5))

    ax.set_title(data_name)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.plot(df.timestamp, df.value)

    ax.axvline(
        x=df.timestamp[trn_len], label="train {}".format(df.timestamp[trn_len]), c="k"
    )
    ax.axvline(
        x=df.timestamp[len(df) - tst_len],
        label="test {}".format(df.timestamp[len(df) - tst_len]),
        c="k",
    )
    ax.set_xticks([df.timestamp[trn_len], df.timestamp[len(df) - tst_len]])
    ax.set_xticklabels(["train", "test"], rotation=0, color="k")

    for i, label in enumerate(df.is_anomaly):
        if label == 1:
            ax.axvline(x=df.timestamp[i], linewidth=0.5, color="#d62728")

    if save_path != None:
        fig.savefig(f"{save_path}/splied_data.png")
