import matplotlib.pyplot as plt
import csv
import numpy as np

fieldnames = ['Mamba Train','LSTM Train','Mamba Test','LSTM Test']

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


def generateViolinPlots():
    plt.figure()
    vpMamba = plt.violinplot([mambaTrain,lstmTrain],showmeans=True)
    plt.xticks([1, 2], ['Mamba', 'LSTM'])
    plt.ylabel('Time (sec)')
    plt.grid()
    plt.title("Network Training Times")
    vpMamba['cmeans'].set_color('black') 
    vpMamba['cmeans'].set_linestyle('--')

    for i, body in enumerate(vpMamba['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor(edge_colors[i])
        body.set_alpha(0.7)  # Set transparency


    plt.figure()
    vpLSTM = plt.violinplot([mambaTest,lstmTest],showmeans=True)
    plt.xticks([1, 2], ['Mamba', 'LSTM'])
    plt.ylabel('Time (sec)')
    plt.grid()
    plt.title("Network Testing Timescl")
    vpLSTM['cmeans'].set_color('black') 
    vpLSTM['cmeans'].set_linestyle('--')
    for i, body in enumerate(vpLSTM['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor(edge_colors[i])
        body.set_alpha(0.7)  # Set transparency

def generateBoxPlots():
    plt.figure()
    plt.boxplot([mambaTrain,lstmTrain],showmeans=True)
    plt.xticks([1, 2], ['Mamba', 'LSTM'])
    plt.ylabel('Time (sec)')
    plt.grid()
    plt.title("Network Training Times")

    plt.figure()
    plt.boxplot([mambaTest,lstmTest],showmeans=True)
    plt.xticks([1, 2], ['Mamba', 'LSTM'])
    plt.ylabel('Time (sec)')
    plt.grid()
    plt.title("Network Testing Times")


if __name__ == "__main__":

    filepath = "p2bp.csv"
    filepath = "cr3bp.csv"

    data_dict = csv_columns_to_numpy(filepath)
    
    colors = ['lightblue', 'lightcoral']
    edge_colors = ['blue', 'red']


    # Each key in data_dict corresponds to a field name, and the value is a NumPy array.
    mambaTrain = data_dict["Mamba Train"]
    lstmTrain = data_dict["LSTM Train"]
    mambaTest = data_dict["Mamba Test"]
    lstmTest = data_dict["LSTM Test"]
    
    generateBoxPlots()
    generateViolinPlots()

    plt.show()