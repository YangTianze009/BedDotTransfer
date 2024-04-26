import os
import numpy as np
from sklearn.utils import shuffle

# Define the root directory where your data is stored
root_dir = "datasets"

# Define the noise levels
noise_levels = ["00", "02", "04", "06", "08", "10"]

# Create lists to store combined data
all_train_data = []
all_test_data = []
all_val_data = []

# Iterate through each noise level
for noise_level in noise_levels:
    # Load the final datasets for each noise level
    train_file = os.path.join(
        root_dir,
        f"stable_noise{noise_level}",
        "envelope_data",
        f"final_train_data_noise{noise_level}.npy",
    )
    test_file = os.path.join(
        root_dir,
        f"stable_noise{noise_level}",
        "envelope_data",
        f"final_test_data_noise{noise_level}.npy",
    )
    val_file = os.path.join(
        root_dir,
        f"stable_noise{noise_level}",
        "envelope_data",
        f"final_val_data_noise{noise_level}.npy",
    )

    # Load data from files
    train_data = np.load(train_file)
    test_data = np.load(test_file)
    val_data = np.load(val_file)

    # Append data to the combined lists
    all_train_data.append(train_data)
    all_test_data.append(test_data)
    all_val_data.append(val_data)

# Concatenate all data for each split
combined_train_data = np.concatenate(all_train_data, axis=0)
combined_test_data = np.concatenate(all_test_data, axis=0)
combined_val_data = np.concatenate(all_val_data, axis=0)

# Shuffle the combined data
combined_train_data = shuffle(combined_train_data, random_state=42)
combined_test_data = shuffle(combined_test_data, random_state=42)
combined_val_data = shuffle(combined_val_data, random_state=42)

# Save the combined and shuffled data
np.save(
    os.path.join(root_dir, "stabled_enveloped_data", "combined_train_data.npy"),
    combined_train_data,
)
np.save(
    os.path.join(root_dir, "stabled_enveloped_data", "combined_test_data.npy"),
    combined_test_data,
)
np.save(
    os.path.join(root_dir, "stabled_enveloped_data", "combined_val_data.npy"),
    combined_val_data,
)
