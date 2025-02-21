import os
import tensorflow as tf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# 1. TFRecord Parsing Setup
# ------------------------------
# Define the TFRecord feature description
feature_description = {
    'tmp_day': tf.io.VarLenFeature(tf.float32),
    'population': tf.io.VarLenFeature(tf.float32),
    'wind_75': tf.io.VarLenFeature(tf.float32),
    'wdir_gust': tf.io.VarLenFeature(tf.float32),
    'wdir_wind': tf.io.VarLenFeature(tf.float32),
    'fuel3': tf.io.VarLenFeature(tf.float32),
    'viirs_PrevFireMask': tf.io.VarLenFeature(tf.float32),
    'elevation': tf.io.VarLenFeature(tf.float32),
    'avg_sph': tf.io.VarLenFeature(tf.float32),
    'chili': tf.io.VarLenFeature(tf.float32),
    'NDVI': tf.io.VarLenFeature(tf.float32),
    'fuel1': tf.io.VarLenFeature(tf.float32),
    'fuel2': tf.io.VarLenFeature(tf.float32),
    'viirs_FireMask': tf.io.VarLenFeature(tf.float32),  # target variable
    'tmp_75': tf.io.VarLenFeature(tf.float32),
    'water': tf.io.VarLenFeature(tf.float32),
    'impervious': tf.io.VarLenFeature(tf.float32),
    'pr': tf.io.VarLenFeature(tf.float32),
    'pdsi': tf.io.VarLenFeature(tf.float32),
    'gust_med': tf.io.VarLenFeature(tf.float32),
    'wind_avg': tf.io.VarLenFeature(tf.float32),
    'bi': tf.io.VarLenFeature(tf.float32),
    'erc': tf.io.VarLenFeature(tf.float32),
}


def _parse_tfrecord(example_proto):
    """Parse a single TFRecord example."""
    return tf.io.parse_single_example(example_proto, feature_description)


# ------------------------------
# 2. Define the PyTorch Dataset
# ------------------------------
class WildfireDataset(Dataset):
    def __init__(self, tfrecord_folder):
        """
        Reads TFRecord files from tfrecord_folder, parses each record,
        and stores samples in self.samples as a list of dictionaries.
        """
        self.samples = []
        tfrecord_files = [os.path.join(tfrecord_folder, f)
                          for f in os.listdir(tfrecord_folder) if f.endswith(".tfrecord")]
        for tfrecord_file in tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            parsed_dataset = raw_dataset.map(_parse_tfrecord).apply(tf.data.experimental.ignore_errors())

            for record in parsed_dataset:
                sample = {}
                # Extract target (fire mask)
                fire_mask_sparse = record.get("viirs_FireMask", None)
                if fire_mask_sparse is not None:
                    sample["fire_mask"] = tf.sparse.to_dense(fire_mask_sparse, default_value=0.0).numpy()
                else:
                    sample["fire_mask"] = np.array([0.0])

                # Extract each feature (full array, not just scalars)
                for key in feature_description.keys():
                    if key == "viirs_FireMask":
                        continue  # already extracted as target
                    feature_sparse = record.get(key, None)
                    if feature_sparse is not None:
                        sample[key] = tf.sparse.to_dense(feature_sparse, default_value=0.0).numpy()
                    else:
                        sample[key] = np.array([0.0])
                self.samples.append(sample)

        if len(self.samples) == 0:
            raise RuntimeError("No samples loaded. Check your TFRecord folder.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return a dictionary with each feature converted to a torch.Tensor."""
        sample = self.samples[idx]
        sample_torch = {}
        for key, value in sample.items():
            sample_torch[key] = torch.tensor(value, dtype=torch.float32)
        return sample_torch


# ------------------------------
# 3. Save Parsed Data to Disk
# ------------------------------
def save_parsed_data(dataset, output_dir):
    """
    Save the parsed dataset to disk.

    The function organizes the data by feature key and saves each
    as a separate .npy file. All samples for a given feature are
    stacked along a new axis (i.e., sample index).
    """
    os.makedirs(output_dir, exist_ok=True)

    if len(dataset.samples) == 0:
        print("No samples to save!")
        return

    # Get all keys from the first sample (assumes all samples have the same keys)
    keys = dataset.samples[0].keys()
    data_by_key = {key: [] for key in keys}

    # Organize data by feature key
    for sample in dataset.samples:
        for key in keys:
            data_by_key[key].append(sample[key])

    # Save each feature's data as a .npy file
    for key, data_list in data_by_key.items():
        try:
            data_array = np.stack(data_list)
        except Exception as e:
            print(f"Could not stack data for key {key} due to: {e}. Saving as object array.")
            data_array = np.array(data_list, dtype=object)
        file_path = os.path.join(output_dir, f"{key}.npy")
        np.save(file_path, data_array)
        print(f"Saved {key} data to {file_path}")


# ------------------------------
# 4. Main Execution: Parse, Save, and Create DataLoader
# ------------------------------
if __name__ == '__main__':
    # Folder where your TFRecord files are stored
    tfrecord_folder = "./"  # adjust as needed

    # Folder where you want to store the parsed data
    output_dir = "./parsed_output"

    # Create the PyTorch-friendly dataset by parsing TFRecords
    dataset = WildfireDataset(tfrecord_folder)

    # Save the parsed data to disk (each feature will be saved separately)
    save_parsed_data(dataset, output_dir)

    # Example: Create a DataLoader to use the dataset in PyTorch
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Iterate over one batch and print the shape of each feature tensor
    for batch in dataloader:
        print("Batch keys and shapes:")
        for key, tensor in batch.items():
            print(f"{key}: {tensor.shape}")
        break
