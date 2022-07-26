import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils import print_log
from .mean_std import obj_stats_384


def plt_show(args, img, ori, epoch):
    mean, std = obj_stats_384(args.obj)
    img = img[0, :, :, :].detach().cpu().numpy()
    ori = ori[0, :, :, :].detach().cpu().numpy()
    if img.dtype != "uint8":
        img_numpy = denormalization(img, mean, std)
        ori_numpy = denormalization(ori, mean, std)

    fig, plots = plt.subplots(1, 2)

    fig.set_figwidth(9)
    fig.set_tight_layout(True)
    plots = plots.reshape(-1)
    plots[0].imshow(img_numpy)
    plots[1].imshow(ori_numpy)

    plots[0].set_title("recons")
    plots[1].set_title("original")


def denormalization(x, mean, std):
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.0).astype(np.uint8)
    return x


class EarlyStop:
    """Used to early stop the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, verbose=True, delta=0, save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, log):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, log)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print_log(
                (f"EarlyStopping counter: {self.counter} out of {self.patience}"), log
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, log)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, optimizer, log):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print_log(
                (
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
                ),
                log,
            )

        torch.save(model.state_dict(), self.save_name)
        self.val_loss_min = val_loss
