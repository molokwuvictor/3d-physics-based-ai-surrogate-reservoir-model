import re
import numpy as np
import matplotlib.pyplot as plt

# User parameter: max boxplots to show
max_boxplots = 50  # change this to your preferred max number

# Load the file
with open("tensor_log.txt", "r") as f:
    text = f.read()

# Extract all quoted value blocks (each block = one row)
value_blocks = re.findall(r'values:\s*"([^"]+)"', text)

all_rows = []
for block in value_blocks:
    try:
        float_vals = [float(x.strip()) for x in block.split(",") if x.strip()]
        if float_vals:
            all_rows.append(float_vals)
    except ValueError:
        continue

print(f"Extracted {len(all_rows)} rows.")

num_rows = len(all_rows)

# Sampling rows uniformly if needed
if num_rows > max_boxplots:
    indices = np.linspace(0, num_rows - 1, max_boxplots, dtype=int)
    sampled_rows = [all_rows[i] for i in indices]
    sampled_indices = indices
else:
    sampled_rows = all_rows
    sampled_indices = range(num_rows)

def moving_average(data, window_size=5):
    """Compute moving average with window_size, using 'valid' mode."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Assume 'sampled_rows' and 'sampled_indices' are prepared as before

plt.figure(figsize=(max(10, len(sampled_rows)*0.2), 6), dpi=300)
plt.rcParams["font.family"] = "Times New Roman"
title_fontsize = 14
label_fontsize = 12

# Boxplot of sampled rows
plt.boxplot(sampled_rows, positions=range(len(sampled_rows)), patch_artist=True, showmeans=True)

# Calculate mean of each row for moving average line
row_means = [np.mean(row) for row in sampled_rows]

# Compute moving average (adjust window size as needed, here 5 or smaller if too few points)
window_size = min(5, len(row_means))
mov_avg = moving_average(row_means, window_size=window_size)

# Because moving average reduces array length, adjust x positions accordingly
x_mov_avg = np.arange(window_size - 1, len(row_means))

plt.plot(x_mov_avg, mov_avg, color='red', linewidth=2, label='Moving Average of Row Means')

# Force y-axis to start at 0
plt.ylim(bottom=0)

plt.title(f"Box plots of grid (average) time steps from batches of training data, sampled after every {len(sampled_rows)} training steps,\n including the moving average of all batch means, computed with a window size of {window_size}", fontsize=title_fontsize)
plt.xlabel("Training step", fontsize=label_fontsize)
plt.ylabel("Average time step (days)", fontsize=label_fontsize)
plt.xticks(ticks=range(len(sampled_rows)), labels=sampled_indices, rotation=90, fontsize=label_fontsize)
plt.yticks(fontsize=label_fontsize)
plt.legend(fontsize=label_fontsize)
plt.grid(False)
#plt.grid(False, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# OPTIONAL: Plot summary statistics (min, mean, max) for all rows (unchanged)
mins = [min(row) for row in all_rows]
means = [sum(row)/len(row) for row in all_rows]
maxs = [max(row) for row in all_rows]

plt.figure(figsize=(12, 5), dpi=300)
plt.rcParams["font.family"] = "Times New Roman"

plt.plot(mins, label="Min", alpha=0.7)
plt.plot(means, label="Mean", alpha=0.7)
plt.plot(maxs, label="Max", alpha=0.7)
plt.title("Summary Statistics Over All Rows", fontsize=title_fontsize)
# Force y-axis to start at 0
plt.xlim(left=0)

plt.xlabel("Training steps", fontsize=label_fontsize)
plt.ylabel("Average time step (days)", fontsize=label_fontsize)
plt.legend()
plt.grid(False)
#plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(fontsize=label_fontsize)
plt.yticks(fontsize=label_fontsize)
plt.tight_layout()
plt.show()
