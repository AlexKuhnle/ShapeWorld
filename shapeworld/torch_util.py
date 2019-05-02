import sys
import numpy as np
import torch.utils.data


class ShapeWorldDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, mode=None, include_model=False, epoch=False, is_channels_first=True, preprocessing=None):
        super(ShapeWorldDataset, self).__init__()
        self.dataset = dataset
        self.mode = mode
        self.include_model = include_model
        self.epoch = epoch
        if self.epoch:
            self.dataset.random_sampling = False
        self.is_channels_first = is_channels_first
        self.preprocessing = dict() if preprocessing is None else preprocessing
        self.initialize_iterator()
        self.index = -1

    def initialize_iterator(self):
        if self.epoch:
            self.iterator = self.dataset.epoch(n=1, mode=self.mode, include_model=self.include_model, alternatives=False)
        else:
            self.iterator = self.dataset.iterate(n=1, mode=self.mode, include_model=self.include_model, alternatives=False)

    def __getitem__(self, index):
        self.index += 1
        assert index == self.index, 'random shuffling invalid: ' + str((index, self.index, self, self.mode))
        try:
            generated = next(self.iterator)
            for value_name, value in generated.items():
                if self.is_channels_first and (self.dataset.values[value_name] == 'world' or value_name.endswith('_features')):
                    generated[value_name] = np.transpose(value[0], axes=(2, 0, 1))
                else:
                    generated[value_name] = value[0]
            for value_name, preprocessing in self.preprocessing.items():
                generated[value_name] = preprocessing(generated[value_name])
            return {value_name: value for value_name, value in generated.items()}
        except StopIteration:
            self.initialize_iterator()
            self.index = -1
            return None

    def __len__(self):
        return sys.maxsize

    def __add__(self, other):
        raise NotImplementedError


class ShapeWorldDataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, num_workers=0):
        assert isinstance(dataset, ShapeWorldDataset)
        super(ShapeWorldDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

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
