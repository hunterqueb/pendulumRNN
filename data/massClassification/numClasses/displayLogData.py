import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_training_log(file_path):
    with open(file_path, "r") as file:
        log_data = file.read()

    model_sections = re.split(r"Entering (.+?) Training Loop", log_data)[1:]
    parsed_data = []

    for i in range(0, len(model_sections), 2):
        model_name = model_sections[i].strip()
        model_log = model_sections[i + 1]

        val_accs = re.findall(r"Validation Accuracy: ([\d.]+)%", model_log)
        max_val_acc = max(map(float, val_accs)) if val_accs else 0.0

        params_match = re.search(r"Total parameters: (\d+)", model_log)
        params = int(params_match.group(1)) if params_match else 0

        memory_match = re.search(r"Total memory \(MB\): ([\d.]+)", model_log)
        memory_mb = float(memory_match.group(1)) if memory_match else 0.0

        loss_vals = re.findall(r"Loss: ([\d.]+)", model_log)
        final_loss = float(loss_vals[-1]) if loss_vals else 0.0

        epochs_trained = len(val_accs)

        time_match = re.search(r"Elapsed time is ([\d.]+) seconds", model_log)
        training_time = float(time_match.group(1)) if time_match else 0.0

        early_stopping = "Yes" if "Early stopping" in model_log else "No"

        lr_events = len(re.findall(r"reducing learning rate", model_log))

        parsed_data.append({
            "Model": model_name,
            "Max Validation Accuracy": f"{max_val_acc:.2f}%",
            "Epochs Trained": epochs_trained,
            "Total Params": params,
            "Memory (MB)": round(memory_mb, 4),
            "Final Loss": final_loss,
            "Training Time (s)": round(training_time, 2),
            "Early Stopping": early_stopping,
            "LR Reduction Events": lr_events,
            "Numeric Accuracy": max_val_acc  # for plotting
        })

    return pd.DataFrame(parsed_data)

def plot_validation_accuracy(df, output_file):
    plt.figure(figsize=(8, 5))
    plt.bar(df["Model"], df["Numeric Accuracy"], color="skyblue")
    plt.ylim(0, 100)
    plt.ylabel("Max Validation Accuracy (%)")
    plt.title("Max Validation Accuracy by Model")
    for i, val in enumerate(df["Numeric Accuracy"]):
        plt.text(i, val + 1, f"{val:.2f}%", ha='center')
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":

    import argparse
    # get input string from command line and use as input_log
    parser = argparse.ArgumentParser(description="Parse training log and generate summary to CSV table and plot of validation accuracy.")
    parser.add_argument("input_log", type=str, help="Input log file name without extension")
    args = parser.parse_args()
    input_log = args.input_log
    
    folder = "parsedData/"
    logExt = ".log"
    output_csv = "parsed_summary"
    outputFile = folder + output_csv + "_" + input_log + ".csv"
    outputPlot = folder + "validation_accuracy_" + input_log +".png"
    df = parse_training_log(input_log + logExt)

    # Show table in console
    print("\nModel Summary:")
    print(df.drop(columns="Numeric Accuracy").to_string(index=False))

    # Save CSV
    df.drop(columns="Numeric Accuracy").to_csv(outputFile, index=False)
    print(f"Saved summary to {outputFile}")

    # Generate and save plot
    plot_validation_accuracy(df,outputPlot)
