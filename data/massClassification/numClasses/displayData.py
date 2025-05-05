import os
import re
import pandas as pd

def parse_log_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    # Extract metadata
    metadata = {}
    metadata_patterns = {
        'MaxMass': (r'Maximum mass for classification\s*:\s*([\d.]+)', float),
        'NumClasses': (r'Number of equispaced classes\s*:\s*(\d+)', int),
        'NumSystems': (r'Number of random systems\s*:\s*(\d+)', int),
        'UseLSTM': (r'Use LSTM comparison\s*:\s*(\w+)', lambda x: x == 'True'),
        'UseTransformer': (r'Use Transformer comparison\s*:\s*(\w+)', lambda x: x == 'True'),
        'NumLayers': (r'Number of Layers\s*:\s*(\d+)', int),
        'Damping': (r'Damping of System\s*:\s*([\d.]+)', float),
        'TotalParams_LSTM': (r'Entering LSTM Training Loop[\s\S]+?Total parameters:\s*(\d+)', int),
        'TotalParams_Mamba': (r'Entering Mamba Training Loop[\s\S]+?Total parameters:\s*(\d+)', int),
        'MemoryMB_LSTM': (r'Entering LSTM Training Loop[\s\S]+?Total memory \(MB\):\s*([\d.]+)', float),
        'MemoryMB_Mamba': (r'Entering Mamba Training Loop[\s\S]+?Total memory \(MB\):\s*([\d.]+)', float),
        'Elapsed_LSTM': (r'Entering LSTM Training Loop[\s\S]+?Elapsed time is\s*([\d.]+)', float),
        'Elapsed_Mamba': (r'Entering Mamba Training Loop[\s\S]+?Elapsed time is\s*([\d.]+)', float)
    }
    for key, (pattern, cast_func) in metadata_patterns.items():
        match = re.search(pattern, content)
        if match:
            try:
                metadata[key] = cast_func(match.group(1))
            except ValueError:
                metadata[key] = None
        else:
            metadata[key] = None

    # Extract validation accuracies
    val_acc_lstm = re.findall(r'Entering LSTM Training Loop[\s\S]+?Epoch \[\d+/\d+\], Training Loss: [\d.]+\s*Validation Loss: [\d.]+, Validation Accuracy: ([\d.]+)%', content)
    val_acc_mamba = re.findall(r'Entering Mamba Training Loop[\s\S]+?Epoch \[\d+/\d+\], Training Loss: [\d.]+\s*Validation Loss: [\d.]+, Validation Accuracy: ([\d.]+)%', content)

    if val_acc_lstm:
        val_acc_lstm = list(map(float, val_acc_lstm))
        metadata['MaxValAcc_LSTM'] = max(val_acc_lstm)
        metadata['EpochValAcc_LSTM'] = val_acc_lstm.index(max(val_acc_lstm)) + 1
    else:
        metadata['MaxValAcc_LSTM'] = None
        metadata['EpochValAcc_LSTM'] = None

    if val_acc_mamba:
        val_acc_mamba = list(map(float, val_acc_mamba))
        metadata['MaxValAcc_Mamba'] = max(val_acc_mamba)
        metadata['EpochValAcc_Mamba'] = val_acc_mamba.index(max(val_acc_mamba)) + 1
    else:
        metadata['MaxValAcc_Mamba'] = None
        metadata['EpochValAcc_Mamba'] = None

    metadata['File'] = os.path.basename(filepath)
    return metadata

def parse_all_logs(folder):
    all_data = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt') or filename.endswith('.log') or filename.endswith('.out'):
            filepath = os.path.join(folder, filename)
            parsed = parse_log_file(filepath)
            all_data.append(parsed)

    df = pd.DataFrame(all_data)
    return df

if __name__ == '__main__':
    folder = 'data/massClassification/dampingMassClass'  # replace with your folder path
    df = parse_all_logs(folder)
    output_csv = folder + '/aggregated_log_data.csv'
    df.to_csv(output_csv, index=False)
    print(f"Saved aggregated data to {output_csv}")
