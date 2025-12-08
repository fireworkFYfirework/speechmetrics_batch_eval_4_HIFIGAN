from speechmetrics import load
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# config
use_pesq = False  # 

# load metrics
metric_list = ['mosnet', 'stoi', 'sisdr']
if use_pesq:
    metric_list.append('pesq')
metrics = load(metric_list, window=None)

# folders
generated_dir = 'audio/clean_testset_wav'
ground_truth_dir = 'audio/noisy_testset_wav'

# get matching files
gen_files = sorted([f for f in os.listdir(generated_dir) if f.endswith('.wav')])
results = {metric: [] for metric in metric_list}
filenames = []

print("evaluating files...")
for filename in gen_files:
    gen_path = os.path.join(generated_dir, filename)
    gt_path = os.path.join(ground_truth_dir, filename)
    
    if os.path.exists(gt_path):
        scores = metrics(gen_path, gt_path)
        filenames.append(filename)
        for metric_name, score in scores.items():
            # extract scalar
            if hasattr(score, 'item'):
                score = score.item()
            elif isinstance(score, (list, np.ndarray)):
                score = float(np.mean(score))
            results[metric_name].append(score)

print(f"evaluated {len(filenames)} files")

# create dataframe
df = pd.DataFrame(results, index=filenames)
print("\nsummary statistics:")
print(df.describe())

# plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# boxplot
ax = axes[0, 0]
df.boxplot(ax=ax)
ax.set_title('metric distributions')
ax.set_ylabel('score')

# mean scores
ax = axes[0, 1]
means = df.mean()
ax.bar(means.index, means.values)
ax.set_title('average scores')
ax.set_ylabel('mean score')
ax.set_ylim([0, max(means.values) * 1.2])

# score trends
ax = axes[1, 0]
for col in df.columns:
    ax.plot(df[col], label=col, alpha=0.7)
ax.set_title('score per file')
ax.set_xlabel('file index')
ax.set_ylabel('score')
ax.legend()

# histogram
ax = axes[1, 1]
df.hist(bins=20, ax=ax, alpha=0.7)
ax.set_title('score histograms')
ax.set_xlabel('score')
ax.set_ylabel('frequency')

plt.tight_layout()
plt.savefig('evaluation_report.png', dpi=300)
print("\nsaved: evaluation_report.png")

# save csv
df.to_csv('evaluation_results.csv')
print("saved: evaluation_results.csv")

plt.show()