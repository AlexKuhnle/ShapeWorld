from shapeworld.dataset import Dataset
from shapeworld.datasets import nlvr_util


class NLVRDataset(Dataset):

    dataset_type = 'nlvr_agreement'
    dataset_name = 'nlvr'
    dataset_values = {'world1': 'world', 'world2': 'world', 'world3': 'world', 'world_model1': 'model', 'world_model2': 'model', 'world_model3': 'model', 'description': 'text', 'description_model': 'model', 'description_length': 'int', 'agreement': 'float'}

    def __init__(self, directory):
        world_size = tuple(next(nlvr_util.images_iter(directory=directory, mode='train'))[1][0].shape[:2])
        self.description_size = 0
        words = set()
        for _, _, description, _ in nlvr_util.descriptions_iter(directory=directory, mode='train'):
            description = description.split()
            self.description_size = max(self.description_size, len(description))
            words.update(description)
        words = sorted(words)
        super(NLVRDataset, self).__init__(world_size=world_size, vectors=dict(description=self.description_size), words=words)
        self.nlvr = {mode: nlvr_util.nlvr(directory=directory, mode=mode) for mode in ('train', 'validation', 'test')}

    def generate(self, n, mode=None, noise_range=None, include_model=False, alternatives=False):
        assert noise_range is None or noise_range == 0.0
        batch = self.zero_batch(n, include_model=include_model, alternatives=alternatives)
        unknown = self.words['UNKNOWN']
        for i in range(n):
            try:
                worlds, world_models, description, agreement = next(self.nlvr[mode])
            except StopIteration:
                if i > 0:
                    return {key: value[:i] for key, value in batch.items()}
                else:
                    return None
            batch['world1'][i], batch['world2'][i], batch['world3'][i] = worlds
            if include_model:
                batch['world_model1'][i], batch['world_model2'][i], batch['world_model3'][i] = world_models
            description = description.split()
            assert len(description) <= self.description_size
            for w, word in enumerate(description):
                batch['description'][i][w] = self.words.get(word, unknown)
            batch['description_length'][i] = len(description)
            batch['agreement'][i] = agreement
        return batch


dataset = NLVRDataset
NLVRDataset.default_config = dict()
