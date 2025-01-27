import matplotlib
# matplotlib.use("tkagg")

import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.lines as mlines

def csv_columns_to_numpy(file_path):
    """
    Reads a CSV file using DictReader,
    groups data by field (column),
    and returns a dictionary of NumPy arrays keyed by field name.
    """
    with open(file_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Create a dictionary to store columns; initialize each key with an empty list
        columns = {field: [] for field in reader.fieldnames}
        
        # Populate the lists for each column
        for row in reader:
            for field in reader.fieldnames:
                columns[field].append(row[field])
                
    # Convert each list in the dictionary to a NumPy array
    # Optionally, convert numeric fields to float or int if appropriate
    for field in columns:
        # Example: try converting to float if possible, otherwise keep as string
        try:
            columns[field] = np.array(columns[field], dtype=np.float64)
        except ValueError:
            # If conversion fails (e.g., for "name" or "city"), keep them as strings
            columns[field] = np.array(columns[field], dtype=str)
    
    return columns


def generatePlots(conv1d100,conv1d300,dt_proj100,dt_proj300):

    plt.figure()
    vpMamba = plt.boxplot([conv1d100,conv1d300],showmeans=True)
    plt.xticks([1, 2], ['100 Samples', '300 Samples'])
    plt.ylabel('Weight Activation')
    plt.grid()
    plt.title("Weight Activations for E321 conv1d Mamba Approximation")

    plt.figure()
    vpLSTM = plt.boxplot([dt_proj100,dt_proj300],showmeans=True)
    plt.xticks([1, 2], ['100 Samples', '300 Samples'])
    plt.ylabel('Weight Activation')
    plt.grid()
    plt.title("Weight Activations for E321 dt_proj Mamba Approximation")


if __name__ == "__main__":

    fileExt = "Samples.csv"

    fileName = "superWeight"

    numSamples = 100

    filepath = fileName + str(numSamples) + fileExt
    data_dict = csv_columns_to_numpy(filepath)
    # Each key in data_dict corresponds to a field name, and the value is a NumPy array.
    in_proj100 = data_dict["in_proj"]
    conv1d100 = data_dict["conv1d"]
    x_proj100 = data_dict["x_proj"]
    dt_proj100 = data_dict["dt_proj"]
    out_proj100 = data_dict["out_proj"]
    

    fileExt = "Samples.csv"

    fileName = "superWeight"

    numSamples = 300

    filepath = fileName + str(numSamples) + fileExt
    data_dict = csv_columns_to_numpy(filepath)
    # Each key in data_dict corresponds to a field name, and the value is a NumPy array.
    in_proj300 = data_dict["in_proj"]
    conv1d300 = data_dict["conv1d"]
    x_proj300 = data_dict["x_proj"]
    dt_proj300 = data_dict["dt_proj"]
    out_proj300 = data_dict["out_proj"]


    generatePlots(conv1d100,conv1d300,dt_proj100,dt_proj300)


    plt.show()