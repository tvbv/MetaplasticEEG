import os
import pickle
import numpy as np
from multiprocessing import Pool
from datetime import datetime


DATA_DIRECTORY = "/.../Data/pckl" # Change it to where the pckl are stored
SAVE_DIRECTORY = "/.../Data/TUH_preprocessed" # Change it to were you want to save the processed data


SLICE_LENGHT = 12


# Define a function to load each file and process it
def load_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    label_mapping = {"seiz": 1, "bckg": 0}
    x = data["signals"].transpose()
    y = label_mapping[data["label"]]  # map the label to an integer
    return x, y


for DATA_TO_PROCESS in ["eval", "dev", "train"]:
    print(DATA_TO_PROCESS)

    folder_path = DATA_DIRECTORY + f"/task-binary_datatype-{DATA_TO_PROCESS}/"
    output_x_file_path = SAVE_DIRECTORY + f"/{DATA_TO_PROCESS}_x_not_mmap.npy"
    output_y_file_path = SAVE_DIRECTORY + f"/{DATA_TO_PROCESS}_y_not_mmap.npy"

    # Get the list of files and their paths
    file_list = os.listdir(folder_path)
    file_list = [os.path.join(folder_path, filename) for filename in file_list]

    # Load the first file to get the shape of each individual data
    with open(file_list[0], "rb") as f:
        first_file_data = pickle.load(f)
    signal_shape = np.shape(first_file_data["signals"].transpose())

    # Get the number of files
    n_files = len(file_list)

    # Define the shape of the final arrays
    final_x_shape = (n_files,) + signal_shape
    final_y_shape = (n_files,)

    # Create arrays with the right size and data type
    x_array = np.zeros(final_x_shape, dtype=np.float16)
    y_array = np.zeros(final_y_shape, dtype=np.uint8)

    # Apply the load_file function to all files in parallel using Pool.map
    with Pool() as p:
        results = p.map(load_file, file_list)

    # Fill the arrays with the data from the pickle files
    for i, (x, y) in enumerate(results):
        x_array[i, :, :] = x
        y_array[i] = y

    # Save the arrays to disk
    np.save(output_x_file_path, x_array)
    np.save(output_y_file_path, y_array)

    # Delete the arrays to free up memory
    del x_array, y_array

print("done")
