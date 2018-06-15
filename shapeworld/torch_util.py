import sys
import numpy as np
import torch.utils.data


class ShapeWorldDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, mode=None, include_model=False):
        self.dataset = dataset
        self.mode = mode
        self.include_model = include_model
        self.index = 0

    def __getitem__(self, index):
        if index == 0:
            self.index = 0
        else:
            self.index += 1
        assert index == self.index
        generated = self.dataset.generate(n=1, mode=self.mode, include_model=self.include_model, alternatives=False)
        for value_name in generated:
            if self.dataset.values[value_name] == 'world':
                generated[value_name] = np.transpose(generated[value_name], axes=(0, 3, 1, 2))
        return {value_name: value[0] for value_name, value in generated.items()}

    def __len__(self):
        return sys.maxsize

    def __add__(self, other):
        raise NotImplementedError
