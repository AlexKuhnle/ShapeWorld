import sys
import numpy as np
import torch.utils.data


class ShapeWorldDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, mode=None, include_model=False, epoch=False, is_channels_first=True):
        super(ShapeWorldDataset, self).__init__()
        self.dataset = dataset
        self.mode = mode
        self.include_model = include_model
        self.epoch = epoch
        self.is_channels_first = is_channels_first
        self.initialize_iterator()
        self.index = -1

    def initialize_iterator(self):
        if self.epoch:
            self.iterator = self.dataset.epoch(n=1, mode=self.mode, include_model=self.include_model, alternatives=False)
        else:
            self.iterator = self.dataset.iterate(n=1, mode=self.mode, include_model=self.include_model, alternatives=False)

    def __getitem__(self, index):
        self.index += 1
        assert index == self.index, 'random shuffling invalid'
        try:
            generated = next(self.iterator)
            for value_name in generated:
                if self.is_channels_first and (self.dataset.values[value_name] == 'world' or (value_name.endswith('_features') and self.dataset.values[value_name[:-9]] == 'world')):
                    generated[value_name] = np.transpose(generated[value_name], axes=(0, 3, 1, 2))
            return {value_name: value[0] for value_name, value in generated.items()}
        except StopIteration:
            self.initialize_iterator()
            self.index = -1
            return None

    def __len__(self):
        return sys.maxsize

    def __add__(self, other):
        raise NotImplementedError


class ShapeWorldDataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1):
        assert isinstance(dataset, ShapeWorldDataset)
        super(ShapeWorldDataLoader, self).__init__(dataset=dataset, batch_size=batch_size)

    def __iter__(self):
        self.sample_iter = iter(self.batch_sampler)
        while True:
            indices = next(self.sample_iter)
            batch = list()
            for i in indices:
                instance = self.dataset[i]
                if instance is None:
                    break
                batch.append(instance)
            yield self.collate_fn(batch)
            if instance is None:
                break

    def __len__(self):
        return sys.maxsize
