import re
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt

OUTPUT_FOLDER = "parsed_data"

def parse_training_log_summary(file_path):
    """
    Parses a single log file for summary metrics.
    """
    with open(file_path, "r") as file:
        log_data = file.read()

    model_sections = re.split(r"Entering (.+?) Training Loop", log_data)[1:]
    parsed_data = []

    for i in range(0, len(model_sections), 2):
        model_name = model_sections[i].strip()
        model_log = model_sections[i + 1]

        val_accs = re.findall(r"Validation Accuracy:\s*([\d.]+)%", model_log)
        max_val_acc = max(map(float, val_accs)) if val_accs else 0.0

        final_loss_matches = re.findall(r"Validation Loss:\s*([\d.]+)", model_log)
        final_loss = float(final_loss_matches[-1]) if final_loss_matches else 0.0

        params_match = re.search(r"Total parameters:\s*(\d+)", model_log)
        params = int(params_match.group(1)) if params_match else 0

        memory_match = re.search(r"Total memory \(MB\):\s*([\d.]+)", model_log)
        memory_mb = float(memory_match.group(1)) if memory_match else 0.0

        time_matches = re.findall(r"Elapsed time is\s*([\d.]+)\s*seconds", model_log)
        training_time = float(time_matches[-1]) if time_matches else 0.0

        early_stopping = "Yes" if "Early stopping" in model_log else "No"
        lr_events = len(re.findall(r"reducing learning rate", model_log))

        parsed_data.append({
            "Model": model_name,
            "Max Validation Accuracy": f"{max_val_acc:.2f}%",
            "Epochs Trained": len(val_accs),
            "Final Validation Loss": round(final_loss, 4),
            "Total Parameters": params,
            "Memory (MB)": round(memory_mb, 4),
            "Training Time (s)": round(training_time, 2),
            "Early Stopping": early_stopping,
            "LR Reduction Events": lr_events,
            "Numeric Accuracy": max_val_acc
        })

    df = pd.DataFrame(parsed_data)
    return df


def parse_training_log_all_epochs(file_path):
    """
    Parses a single log file for epoch-level metrics.
    """
    with open(file_path, "r") as file:
        log_data = file.read()

    model_sections = re.split(r"Entering (.+?) Training Loop", log_data)[1:]

    rows = []

    for i in range(0, len(model_sections), 2):
        model_name = model_sections[i].strip()
        model_log = model_sections[i + 1]

        current_epoch = None

        for line in model_log.splitlines():
            # Check for epoch line
            epoch_match = re.match(r"Epoch\s*\[(\d+)/\d+\]", line.strip())
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            # Check for validation accuracy line
            valacc_match = re.search(r"Validation Accuracy:\s*([\d.]+)%", line)
            if valacc_match and current_epoch is not None:
                val_acc = float(valacc_match.group(1))
                rows.append({
                    "Model": model_name,
                    "Epoch": current_epoch,
                    "Validation Accuracy": val_acc
                })
                current_epoch = None

    df = pd.DataFrame(rows)
    return df


def plot_validation_accuracy_over_epochs(df, output_file):
    """
    Plots validation accuracy over epochs for each model.
    """
    if df.empty:
        print(f"No epoch-level data to plot for {output_file}")
        return

    plt.figure(figsize=(10, 6))
    for model_name, group in df.groupby("Model"):
        plt.plot(
            group["Epoch"],
            group["Validation Accuracy"],
            marker='o',
            label=model_name
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved plot to {output_file}")


def find_all_log_files(root_folder):
    log_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".log"):
                full_path = os.path.join(dirpath, file)
                log_files.append(full_path)
    return log_files


def process_log_file(file_path, root_folder):
    """
    Process one log file and save outputs under parsed_data/ with matching subfolders.
    """
    print(f"\nProcessing: {file_path}")

    summary_df = parse_training_log_summary(file_path)
    epochs_df = parse_training_log_all_epochs(file_path)

    # Compute relative path from root_folder
    relative_path = os.path.relpath(file_path, start=root_folder)
    relative_dir = os.path.dirname(relative_path)

    # Compute output folder under parsed_data
    output_dir = os.path.join(OUTPUT_FOLDER, relative_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_basename = os.path.splitext(os.path.basename(file_path))[0]

    summary_csv = os.path.join(output_dir, f"summary_{log_basename}.csv")
    epochs_csv = os.path.join(output_dir, f"val_acc_epochs_{log_basename}.csv")
    plot_png = os.path.join(output_dir, f"val_acc_plot_{log_basename}.png")

    # Save summary
    summary_df.drop(columns="Numeric Accuracy").to_csv(summary_csv, index=False)
    print(summary_df.drop(columns="Numeric Accuracy").to_string(index=False))
    print(f"Saved summary CSV: {summary_csv}")

    # Save epoch-level CSV
    epochs_df.to_csv(epochs_csv, index=False)
    print(f"Saved epoch-level CSV: {epochs_csv}")

    # Plot
    plot_validation_accuracy_over_epochs(epochs_df, plot_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process all .log files under a root folder."
    )
    parser.add_argument(
        "root_folder",
        type=str,
        help="Path to the root folder to search for log files."
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    all_logs = find_all_log_files(args.root_folder)
    print(f"Found {len(all_logs)} log files.")

    for log_file in all_logs:
        process_log_file(log_file, args.root_folder)
