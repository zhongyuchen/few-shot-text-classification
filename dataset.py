import torch
import torch.utils.data
from utils import batch_padding


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, pad_idx):
        assert len(data_dict['data']) == len(data_dict['target'])
        self.data = self._padding(data_dict['data'], pad_idx)
        self.target = torch.tensor(data_dict['target'])
        self.len = len(data_dict['target'])

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.len

    def _padding(self, data, pad_idx):
        return batch_padding(data, pad_idx)
