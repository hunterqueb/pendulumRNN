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


def generateTimeViolinPlots():
    colors = ['lightblue', 'lightcoral']
    edge_colors = ['blue', 'red']

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
    plt.title("Network Testing Times")
    vpLSTM['cmeans'].set_color('black') 
    vpLSTM['cmeans'].set_linestyle('--')
    for i, body in enumerate(vpLSTM['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor(edge_colors[i])
        body.set_alpha(0.7)  # Set transparency

def generateRMSEViolinPlots(keepLSTM = True):
    # colors = ['lightblue', 'lightcoral']
    # edge_colors = ['blue', 'red']

    plt.figure()
    vpMamba = plt.violinplot(RSMEMambaPos,showmeans=True)
    plt.xticks([1, 2, 3], ['x', 'y','z'])
    plt.ylabel('Distance (km)')
    plt.grid()
    plt.title("Root Mean Square Error of Position Dimensions")
    vpMamba['cmeans'].set_color('black') 
    vpMamba['cmeans'].set_linestyle('--')

    for i, body in enumerate(vpMamba['bodies']):
        # body.set_facecolor(colors[i])
        # body.set_edgecolor(edge_colors[i])
        body.set_alpha(0.7)  # Set transparency

    if keepLSTM:
        vpLSTM = plt.violinplot(RSMELSTMPos,showmeans=True)
        plt.xticks([1, 2, 3], ['x', 'y','z'])
        vpLSTM['cmeans'].set_color('black') 
        vpLSTM['cmeans'].set_linestyle('--')
        for i, body in enumerate(vpLSTM['bodies']):
            # body.set_facecolor(colors[i])
            # body.set_edgecolor(edge_colors[i])
            body.set_alpha(0.7)  # Set transparency

        mambaLine = mlines.Line2D([], [], color='b', label='Mamba')
        LSTMLine = mlines.Line2D([], [], color='orange', label='LSTM')
        meanLine = mlines.Line2D([], [], color='black', label='Mean',linestyle='dashed')
        
        plt.legend(handles=[mambaLine,LSTMLine,meanLine])

    plt.figure()
    vpMamba = plt.violinplot(RSMEMambaVel,showmeans=True)
    plt.xticks([1, 2, 3], ['vx', 'vy','vz'])
    plt.ylabel('Speed (km/s)')
    plt.grid()
    plt.title("Root Mean Square Error of Velocity Dimensions")
    vpMamba['cmeans'].set_color('black') 
    vpMamba['cmeans'].set_linestyle('--')

    for i, body in enumerate(vpMamba['bodies']):
        # body.set_facecolor(colors[i])
        # body.set_edgecolor(edge_colors[i])
        body.set_alpha(0.7)  # Set transparency

    if keepLSTM:
        vpLSTM = plt.violinplot(RSMELSTMPVel,showmeans=True)
        vpLSTM['cmeans'].set_color('black') 
        vpLSTM['cmeans'].set_linestyle('--')
        for i, body in enumerate(vpLSTM['bodies']):
            # body.set_facecolor(colors[i])
            # body.set_edgecolor(edge_colors[i])
            body.set_alpha(0.7)  # Set transparency
        
        plt.legend(handles=[mambaLine,LSTMLine,meanLine])


    return

def generateRMSEBoxPlots(RMSEMambaPos,RMSEMambaVel,RMSELSTMPos,RMSELSTMVel,keepLSTM = True,percentError = False):
    # colors = ['lightblue', 'lightcoral']
    # edge_colors = ['blue', 'red']
    if percentError:
        unitsLabelPos = '% Error'
        unitsLabelVel = '% Error'

        RMSEMambaPos = [RMSEMambaPos[0]*100,RMSEMambaPos[1]*100,RMSEMambaPos[2]*100]
        RMSEMambaVel = [RMSEMambaVel[0]*100,RMSEMambaVel[1]*100,RMSEMambaVel[2]*100]
        if keepLSTM:
            RMSELSTMPos = [RMSELSTMPos[0]*100,RMSELSTMPos[1]*100,RMSELSTMPos[2]*100]
            RMSELSTMVel = [RMSELSTMVel[0]*100,RMSELSTMVel[1]*100,RMSELSTMVel[2]*100]
    else:
        unitsLabelPos = "km"
        unitsLabelVel = "km/s"
    plt.figure()
    vpMamba = plt.boxplot(RMSEMambaPos,showmeans=True,sym='')
    plt.xticks([1, 2, 3], ['x', 'y','z'])
    plt.ylabel('Distance ({})'.format(unitsLabelPos))
    plt.grid()
    plt.title("Root Mean Square Error of Position Dimensions")
    # vpMamba['cmeans'].set_color('black') 
    # vpMamba['cmeans'].set_linestyle('--')

    plt.figure()
    vpMamba = plt.boxplot(RMSEMambaVel,showmeans=True,sym='')
    plt.xticks([1, 2, 3], ['vx', 'vy','vz'])
    plt.ylabel('Speed ({})'.format(unitsLabelVel))
    plt.grid()
    plt.title("Root Mean Square Error of Velocity Dimensions")
    # vpMamba['cmeans'].set_color('black') 
    # vpMamba['cmeans'].set_linestyle('--')

    if keepLSTM:
        plt.figure()

        vpLSTM = plt.boxplot(RMSELSTMPos,showmeans=True,sym='')
        plt.xticks([1, 2, 3], ['x', 'y','z'])
        plt.ylabel('Distance ({})'.format(unitsLabelPos))
        plt.grid()
        plt.title("Root Mean Square Error of Position Dimensions")

        
        plt.figure()
        vpLSTM = plt.boxplot(RMSELSTMVel,showmeans=True,sym='')
        plt.xticks([1, 2, 3], ['vx', 'vy','vz'])
        plt.ylabel('Speed ({})'.format(unitsLabelVel))
        plt.grid()
        plt.title("Root Mean Square Error of Velocity Dimensions")

    return

def generateRMSEBoxPlotsDiff(keepLSTM = True):
    return

def generateTimeBoxPlots():
    plt.figure()
    plt.boxplot([mambaTrain,lstmTrain],showmeans=True,sym='')
    plt.xticks([1, 2], ['Mamba', 'LSTM'])
    plt.ylabel('Time (sec)')
    plt.grid()
    plt.title("Network Training Times")

    plt.figure()
    plt.boxplot([mambaTest,lstmTest],showmeans=True,sym='')
    plt.xticks([1, 2], ['Mamba', 'LSTM'])
    plt.ylabel('Time (sec)')
    plt.grid()
    plt.title("Network Testing Times")

def generateP2BPEnergyPlots():
    fileName = "p2bpEnergy"
    fileExt = '.npy'
    energy = np.load(fileName+fileExt)
    energyMamba = np.load(fileName+"Mamba"+fileExt)
    energyLSTM = np.load(fileName+"LSTM"+fileExt)
    t = np.linspace(0,10,len(energy[:,0]))

    energyMambaMean = energyMamba.mean(axis=1)
    energyMambaMin = energyMamba.min(axis=1)
    energyMambaMax = energyMamba.max(axis=1)
    energyMambaErr = np.stack((energyMambaMean-energyMambaMin, energyMambaMax-energyMambaMean))

    plt.figure()
    plt.plot(t,energy[:,0], 'k-')
    plt.fill_between(t,energyMambaMin, energyMambaMax)

    # plt.fill_between(t,energy[:,0]-energyLSTM[:,0], energy[:,0]+energyLSTM[:,0])

if __name__ == "__main__":

    fileExt = ".csv"

    import argparse
    parser = argparse.ArgumentParser(description="Generate Box Plots for Journal Paper")
    parser.add_argument('--file', type=str, default='cr3bp', help='Orbit Type (p2bp/cr3bp)')
    parser.add_argument('--suffix', type=str, default='', help='File Suffix (""/"Short")')
    parser.add_argument('--percent', dest='percent', action="store_true", help='Disable Percent Error for RMSE')
    parser.set_defaults(percent=False)
    parser.add_argument('--desktop',dest='desktop', action="store_true", help='Box Plots for Desktop Runtime Results')
    parser.set_defaults(desktop=False)
    parser.add_argument('--path', type=str, default='data/journalPaper/', help='Path to the CSV files. Default is "data/journalPaper"')
    args = parser.parse_args()

    fileName = args.file
    suffix = args.suffix
    percentError = args.percent
    desktop = args.desktop
    path = args.path

    # fileName = "p2bp"
    # fileName = "cr3bp"

    # suffix = ''
    # suffix = 'Short'
    fileName = path + fileName
    fileName = fileName + suffix

    if desktop:
        fileName = fileName + "Time_desktop"
    else:
        filepath = fileName + "Time" + fileExt
    data_dict = csv_columns_to_numpy(filepath)
    # Each key in data_dict corresponds to a field name, and the value is a NumPy array.
    mambaTrain = data_dict["Mamba Train"]
    lstmTrain = data_dict["LSTM Train"]
    mambaTest = data_dict["Mamba Test"]
    lstmTest = data_dict["LSTM Test"]
    
    generateTimeBoxPlots()
    # generateTimeViolinPlots()

    # ==========================================================================================

    if percentError:
        filepath = fileName + "RMSPEMamba" + fileExt
    else:
        filepath = fileName + "RMSEMamba" + fileExt
    data_dict = csv_columns_to_numpy(filepath)

    RMSEMambaPos = [data_dict["x"],data_dict["y"],data_dict["z"]]
    RMSEMambaVel = [data_dict["vx"],data_dict["vy"],data_dict["vz"]]

    if percentError:
        filepath = fileName + "RMSPELSTM" + fileExt
    else:
        filepath = fileName + "RMSELSTM" + fileExt
    data_dict = csv_columns_to_numpy(filepath)

    RMSELSTMPos = [data_dict["x"],data_dict["y"],data_dict["z"]]
    RMSELSTMVel = [data_dict["vx"],data_dict["vy"],data_dict["vz"]]


    # generateRMSEViolinPlots(keepLSTM=True)
    # generateRMSEViolinPlots(keepLSTM=False)

    generateRMSEBoxPlots(RMSEMambaPos,RMSEMambaVel,RMSELSTMPos,RMSELSTMVel,keepLSTM=True,percentError=percentError)


    # generateP2BPEnergyPlots()

    plt.show()