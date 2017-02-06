from shapeworld.dataset import Dataset, MixerCaptionAgreementDataset


class MixerDataset(MixerCaptionAgreementDataset):

    def __init__(self, dataset_configs, distribution, train_distribution, validation_distribution, test_distribution, **kwargs):
        datasets = [Dataset.from_config(config=config) for config in dataset_configs]
        super().__init__(
            datasets=datasets,
            distribution=distribution,
            train_distribution=train_distribution,
            validation_distribution=validation_distribution,
            test_distribution=test_distribution)


dataset = MixerDataset
