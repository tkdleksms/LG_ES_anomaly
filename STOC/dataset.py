import torch
from torch.utils.data import Dataset


class BuildDataset(Dataset):
    """
    A class to make a dataset of Yahoo S5 dataset
    Yahoo dataset has columns of timestamp, value, and is_anomaly which is the anomaly label

    Args:
        # args에 root_path, data_name, window_size, slide_size 받아서 output으로 뱉기

        args (object): including settings for loading data and preprocessing
            self.data_name = 'real_1.csv' # dataset name
            self.root_path = '../yahoo_S5/A1Benchmark'
            self.window_size = 60

    """

    def __init__(self, data, args: object):

        self.window_size = args.window_size
        self.slide_size = args.slide_size
        self.forecast_step = args.forecast_step

        self.data = data.iloc[:, 1].values
        self.label = data.iloc[:, -1].values

        self.start_point = range(
            0, len(self.data) - self.window_size - self.forecast_step
        )

        # for forecasting
        self.forecast = args.forecast
        self.forecast_step = args.forecast_step

    def __len__(self):
        return len(self.start_point)

    def __getitem__(self, idx):
        return torch.FloatTensor(
            self.data[self.start_point[idx] : self.start_point[idx] + self.window_size]
        ), torch.FloatTensor(
            self.data[
                self.start_point[idx]
                + self.forecast_step : self.start_point[idx]
                + self.forecast_step
                + self.window_size
            ]
        )
