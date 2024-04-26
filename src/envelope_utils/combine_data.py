import os
import numpy as np
from sklearn.utils import shuffle

# Define the root directory where your data is stored
root_dir = "datasets"

# Define the noise levels
noise_levels = ["00", "02", "04", "06", "08", "10"]

# Iterate through each noise level
for noise_level in noise_levels:
    # Create lists to store data for each split
    train_data = []
    test_data = []
    val_data = []

    # Iterate through each folder within the noise level directory
    for folder in ["0_200", "0_250", "0_280", "0_320", "0_350"]:
        # Load data files for each split
        train_file = os.path.join(
            root_dir,
            f"stable_noise{noise_level}",
            "envelope_data",
            folder,
            "extracted_envelope_data_10k.npy",
        )
        test_file = os.path.join(
            root_dir,
            f"stable_noise{noise_level}",
            "envelope_data",
            folder,
            "extracted_envelope_data_5k.npy",
        )
        val_file = os.path.join(
            root_dir,
            f"stable_noise{noise_level}",
            "envelope_data",
            folder,
            "extracted_envelope_data_2k.npy",
        )

        # Load data from files
        train_data.append(np.load(train_file))
        test_data.append(np.load(test_file))
        val_data.append(np.load(val_file))

    # Concatenate data from all folders
    train_data = np.concatenate(train_data, axis=0)
    test_data = np.concatenate(test_data, axis=0)
    val_data = np.concatenate(val_data, axis=0)

    # Shuffle the data
    train_data = shuffle(train_data, random_state=42)
    test_data = shuffle(test_data, random_state=42)
    val_data = shuffle(val_data, random_state=42)

    # Save the combined and shuffled data
    np.save(
        os.path.join(
            root_dir,
            f"stable_noise{noise_level}",
            "envelope_data",
            f"final_train_data_noise{noise_level}.npy",
        ),
        train_data,
    )
    np.save(
        os.path.join(
            root_dir,
            f"stable_noise{noise_level}",
            "envelope_data",
            f"final_test_data_noise{noise_level}.npy",
        ),
        test_data,
    )
    np.save(
        os.path.join(
            root_dir,
            f"stable_noise{noise_level}",
            "envelope_data",
            f"final_val_data_noise{noise_level}.npy",
        ),
        val_data,
    )
