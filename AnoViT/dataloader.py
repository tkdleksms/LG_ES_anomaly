import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np

from .dataset import MVTecDataset
from .mean_std import obj_stats_384


def get_dataloader(args):
    obj_mean, obj_std = obj_stats_384(args.obj)
    trainT = T.Compose(
        [
            T.Resize(args.image_size, Image.ANTIALIAS),
            T.ToTensor(),
            T.Normalize(mean=obj_mean, std=obj_std),
        ]
    )

    validT = T.Compose(
        [
            T.Resize(args.image_size, Image.ANTIALIAS),
            T.ToTensor(),
            T.Normalize(mean=obj_mean, std=obj_std),
        ]
    )

    testT = T.Compose(
        [
            T.Resize(args.image_size, Image.ANTIALIAS),
            T.ToTensor(),
            T.Normalize(mean=obj_mean, std=obj_std),
        ]
    )

    train_dataset = MVTecDataset(
        args,
        args.dataset_path,
        class_name=args.obj,
        is_train=True,
        resize=args.image_size,
        transform_x=trainT,
    )
    valid_dataset = MVTecDataset(
        args,
        args.dataset_path,
        class_name=args.obj,
        is_train=True,
        resize=args.image_size,
        transform_x=validT,
    )
    img_nums = len(train_dataset)

    indices = list(range(img_nums))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    split = int(np.floor(args.val_ratio * img_nums))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    test_dataset = MVTecDataset(
        args,
        args.dataset_path,
        class_name=args.obj,
        is_train=False,
        resize=args.image_size,
        transform_x=testT,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, sampler=valid_sampler
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
