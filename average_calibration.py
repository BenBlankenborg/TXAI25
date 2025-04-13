import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob

def average_calibration_plot(pickle_pattern="ensemble_predictions_run_*.pkl", num_bins=10):
    all_run_curves = []
    binned_uncertainties = None

    file_list = sorted(glob.glob(pickle_pattern))

    for fname in file_list:
        with open(fname, "rb") as f:
            predictions_list, ground_truths = pickle.load(f)

        uncertainties = []
        errors = []

        for preds, gt in zip(predictions_list, ground_truths):
            mu = np.mean(preds)
            sigma = np.std(preds, ddof=1)
            uncertainties.append(sigma)
            errors.append(abs(mu - gt))

        uncertainties = np.array(uncertainties)
        errors = np.array(errors)

        # Establish bin edges only once
        if binned_uncertainties is None:
            global_min = np.min(uncertainties)
            global_max = np.max(uncertainties)
            bin_edges = np.linspace(global_min, global_max, num_bins + 1)
            binned_uncertainties = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        binned_errors = []
        for i in range(num_bins):
            mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
            if np.any(mask):
                binned_errors.append(np.mean(errors[mask]))
            else:
                binned_errors.append(np.nan)  # Handle empty bin

        all_run_curves.append(binned_errors)

    # Convert to array for stats
    all_run_curves = np.array(all_run_curves)
    binned_errors_mean = np.nanmean(all_run_curves, axis=0)
    binned_errors_std = np.nanstd(all_run_curves, axis=0)

    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(binned_uncertainties, binned_errors_mean, 'o-', color='blue', label="Average Calibration")
    plt.fill_between(
        binned_uncertainties,
        binned_errors_mean - binned_errors_std,
        binned_errors_mean + binned_errors_std,
        color='blue',
        alpha=0.2,
        label="±1 Std Dev"
    )
    plt.plot([0, np.nanmax(binned_uncertainties)], [0, np.nanmax(binned_uncertainties)], 'k--', label="Ideal")
    plt.xlabel("Predicted Std Dev (Uncertainty)")
    plt.ylabel("Actual Absolute Error")
    plt.title("Average Calibration Plot (Across Runs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Average_CalibrationPlotMCD_ensemble.png")
    plt.close()

# Run the function
average_calibration_plot()



