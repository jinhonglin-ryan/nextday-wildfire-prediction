import torch
from tfrecord.torch.dataset import MultiTFRecordDataset
from torch.utils.data import DataLoader
import os
from os import path as osp
import deepdish as dd
import numpy as np

from torch.utils.data import IterableDataset
from torchvision import transforms

clipping_dict = {
    'elevation': (-10, 6500),
    'chili': (0, 255),
    'impervious': (0, 100),
    'water': (0, 100),
    'population': (0, 810694),
    'fuel1': (-1e6, 1e6),
    'fuel2': (-1e6, 1e6),
    'fuel3': (-1e6, 1e6),
    'NDVI': (-2000, 10000),
    'pdsi': (-15, 15),
    'pr': (0, 690),
    'erc': (0, 132),
    'bi': (0, 215),
    'avg_sph': (0, 0.02),
    'tmp_day': (-43, 43),
    'tmp_75': (-43, 43),
    'gust_med': (0, 60),
    'wind_avg': (0, 43),
    'wind_75': (0, 43),
    'wdir_wind': (-np.pi, np.pi),
    'wdir_gust': (-np.pi, np.pi),
    'viirs_PrevFireMask': (0, 1),
}


class ndwsDataset(IterableDataset):
    def __init__(self,
                 dataset_directory='ndws_western_dataset',
                 mode='train',
                 infinite=False,
                 ):
        self.FEATURES = ['elevation', 'chili', 'impervious', 'water',
                         'population', 'fuel1', 'fuel2', 'fuel3', 'NDVI',
                         'pdsi', 'pr', 'erc', 'bi', 'avg_sph', 'tmp_day',
                         'tmp_75', 'gust_med', 'wind_avg', 'wind_75', 'wdir_wind', 'wdir_gust']
        self.FIRE_MASKS = ['viirs_PrevFireMask', 'viirs_FireMask']

        self.dataset_directory = dataset_directory

        tfrecord_pattern = dataset_directory + '/cleaned_' + mode + '_ndws_conus_western_{}.tfrecord'
        tfindex_pattern = dataset_directory + '/cleaned_' + mode + '_ndws_conus_western_{}.tfindex'

        all_files = os.listdir(dataset_directory)
        splits = {}
        total_sizes = 0
        for fname in all_files:
            if 'tfrecord' in fname and mode in fname:
                dotloc = fname.find('.')
                number = fname[dotloc - 3:dotloc]
                filesize = osp.getsize(osp.join(dataset_directory, fname))
                total_sizes += filesize
                splits[number] = filesize
        for x in splits:
            splits[x] = splits[x] / total_sizes
        # The purpose of the above block is to sample tfrecords according to their size

        self.splits = splits
        description = {x: 'float' for x in self.FEATURES + self.FIRE_MASKS}

        self.dataset = MultiTFRecordDataset(
            tfrecord_pattern,
            tfindex_pattern,
            self.splits,
            description,
            infinite=infinite
        )

    def __iter__(self):
        return iter(self.dataset)


class ndwsDataLoader:
    """
    A custom DataLoader that enables preprocessing of batches from an IterableDataset.

    Allows for dropping features and loads data_processing in (B,C,H,W) format.
    Also allows moving batches to device

    Args:
        dataset: The source IterableDataset
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data_processing loading
        device: Device to load tensors to ('cpu' or 'cuda')
        pin_memory: If True, pin memory for faster GPU transfer
        drop_last: If True, drop the last non-full batch
    """

    def __init__(
            self,
            dataset,
            batch_size=1,
            drop_features=[],
            img_dim=64,
            rescale=True,
            stats_file=None,
            crop_augmentation=True,
            random_rotation=True,
            random_flip=True,
    ):
        self.dataset = dataset
        self.hw = img_dim
        self.rescale = rescale
        self.crop_augmentation = crop_augmentation
        self.random_rotation = random_rotation
        self.random_flip = random_flip

        self.drop_features = drop_features

        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
        )

        sample_datapoint = next(iter(dataset))
        self.base_features = list(sample_datapoint.keys())
        self.target = 'viirs_FireMask'
        self.features = []
        for feature in self.base_features:
            if feature in self.drop_features or feature == self.target:
                continue
            self.features.append(feature)

        self.feat_dict = {x: i for i, x in enumerate(self.features)}

        # aggregate train dataset statistics for normalization
        self.stats = dd.io.load(stats_file)

    def preprocess_batch(self, batch):
        """Preprocess a single batch of data_processing.

        Drops specified features and setups up data_processing

        Args:
            batch: A batch of data_processing from the dataset. Could be a tensor,
                  tuple/list of tensors, or dict of tensors.

        Returns:
            The preprocessed batch
        """
        # batch is in the form of a dict
        # target = batch[self.target].view(-1,self.hw,self.hw)

        res = []
        for feature in self.features:
            reshaped = batch[feature].view(-1, self.hw, self.hw).unsqueeze(1)
            reshaped = torch.clamp(reshaped, min=clipping_dict[feature][0], max=clipping_dict[feature][1])
            if self.rescale and feature not in ['viirs_PrevFireMask', 'viirs_FireMask', 'fuel1', 'fuel2', 'fuel3']:
                reshaped = (reshaped - self.stats[feature]['mean']) / np.sqrt(self.stats[feature]['var'])

            res.append(reshaped)
        target = batch[self.target].view(-1, self.hw, self.hw).unsqueeze(1)
        res.append(target)

        data = torch.cat(res, axis=1)

        # perform data_processing augmentation
        if self.crop_augmentation:
            crop_transform = transforms.RandomCrop(self.hw // 2)
            data = crop_transform(data)
        else:
            data = transforms.CenterCrop(self.hw // 2)(data)
        if self.random_rotation:
            rotation_transform = transforms.RandomChoice([
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.RandomRotation(degrees=(180, 180)),
                transforms.RandomRotation(degrees=(270, 270)),
                transforms.RandomRotation(degrees=(0, 0))
            ])
            data = rotation_transform(data)
        if self.random_flip:
            flip_transform = transforms.RandomHorizontalFlip()
            data = flip_transform(data)

        target = data[:, -1, ...]
        data = data[:, :-1, ...]

        return data, target

    def __iter__(self):
        """Iterate over preprocessed batches."""
        for batch in self.dataloader:
            yield self.preprocess_batch(batch)


def load(config):
    modes = ['train', 'test', 'eval']

    data_loaders = {}

    for mode in modes:

        data = ndwsDataset(mode=mode, dataset_directory=config['dataset_directory'])

        batch_size = config['batch_size']
        crop_augment = config['crop_augment']
        if mode == 'test':
            batch_size = 2 * batch_size
            crop_augment = False

        stats_file = osp.join(config['stats_directory'], f'ndws_western_{mode}_stats.h5')
        loader = ndwsDataLoader(
            dataset=data,
            batch_size=batch_size,
            drop_features=config['features_to_drop'],
            rescale=config['rescale'],
            stats_file=stats_file,
            crop_augmentation=crop_augment,
            random_rotation=config['rot_augment'],
            random_flip=config['flip_augment'],
        )

        data_loaders[mode] = loader

    return data_loaders




if __name__ == '__main__':
    config = {
        'dataset_directory': '/Users/jinhonglin/Desktop/nextday-wildfire-prediction/modified_dataset/ndws_western_dataset',
        'stats_directory': '/Users/jinhonglin/Desktop/nextday-wildfire-prediction/modified_dataset/data_statistics',
        'batch_size': 16,
        'features_to_drop': [],
        'rescale': False,
        'crop_augment': False,
        'rot_augment': False,
        'flip_augment': False,
    }

    data_loaders = load(config)

    train_loader = data_loaders['train']

    for data, target in train_loader:
        print("x：", data.shape)  # (B, C, H, W)
        print("y：", target.shape)  # (B, H, W)
        print(data)
        break
