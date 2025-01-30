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


def generateEulerMRPPlots(layersEuler, layersMRP):
    layer_names = ["in_proj", "conv1d", "x_proj", "dt_proj", "out_proj"]
    
    for i, layer_name in enumerate(layer_names):
        plt.figure()
        vpMamba = plt.boxplot([layersEuler[i], layersMRP[i]], showmeans=True)
        plt.xticks([1, 2], ['Euler Representation', 'MRP Representation'])
        plt.ylabel('Weight Activation')
        plt.grid()
        plt.title(f"Weight Activations for E321 + MRP {layer_name} Mamba Approximation")
def generate4d6dPlots(layersEuler, layersMRP):
    layer_names = ["in_proj", "conv1d", "x_proj", "dt_proj", "out_proj"]
    
    for i, layer_name in enumerate(layer_names):
        plt.figure()
        vpMamba = plt.boxplot([layersEuler[i], layersMRP[i]], showmeans=True)
        plt.xticks([1, 2], ['4d', '6d'])
        plt.ylabel('Weight Activation')
        plt.grid()
        plt.title(f"Weight Activations for Short Period CR3BP {layer_name} Mamba Approximation")

def generate4d6dResPlots(layersEuler, layersMRP):
    layer_names = ["in_proj", "conv1d", "x_proj", "dt_proj", "out_proj"]
    
    for i, layer_name in enumerate(layer_names):
        plt.figure()
        vpMamba = plt.boxplot([layersEuler[i], layersMRP[i]], showmeans=True)
        plt.xticks([1, 2], ['4d', '6d'])
        plt.ylabel('Weight Activation')
        plt.grid()
        plt.title(f"Weight Activations for Resonant 4:3 CR3BP {layer_name} Mamba Approximation")


def returnLayerActivationArray(filepath):
    data_dict = csv_columns_to_numpy(filepath)
    # Each key in data_dict corresponds to a field name, and the value is a NumPy array.
    in_proj = data_dict["in_proj"]
    conv1d = data_dict["conv1d"]
    x_proj = data_dict["x_proj"]
    dt_proj = data_dict["dt_proj"]
    out_proj = data_dict["out_proj"]

    return [in_proj,conv1d,x_proj,dt_proj,out_proj]

if __name__ == "__main__":

    fileExt = "Samples.csv"
    fileName = "superWeight"

    filepath = fileName + "Euler" + fileExt
    activationsEuler = returnLayerActivationArray(filepath)    

    filepath = fileName + "MRP" + fileExt
    activationsMRP = returnLayerActivationArray(filepath)    

    generateEulerMRPPlots(activationsEuler,activationsMRP)

    filepath = fileName + "CR3BPSP4d" + fileExt
    activationsCR3BPSP4d = returnLayerActivationArray(filepath)
    
    filepath = fileName + "CR3BPSP6d" + fileExt
    activationsCR3BPSP6d = returnLayerActivationArray(filepath)

    generate4d6dPlots(activationsCR3BPSP4d,activationsCR3BPSP6d)

    filepath = fileName + "CR3BP4d" + fileExt
    activationsCR3BP4d = returnLayerActivationArray(filepath)
    
    filepath = fileName + "CR3BP6d" + fileExt
    activationsCR3BP6d = returnLayerActivationArray(filepath)

    generate4d6dResPlots(activationsCR3BP4d,activationsCR3BP6d)


    plt.show()