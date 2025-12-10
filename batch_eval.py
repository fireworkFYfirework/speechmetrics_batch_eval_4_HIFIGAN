from eval import evaluate_audio_metrics
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_matching_audio_files(generated_dir, ground_truth_dir):
    generated_files = sorted([f for f in os.listdir(generated_dir) if f.endswith('.wav')])
    matching_files = []
    
    for filename in generated_files:
        generated_path = os.path.join(generated_dir, filename)
        ground_truth_path = os.path.join(ground_truth_dir, filename)
        
        if os.path.exists(ground_truth_path):
            matching_files.append({
                'filename': filename,
                'generated_path': generated_path,
                'ground_truth_path': ground_truth_path
            })
    
    return matching_files


def evaluate_batch_files(matching_files):
    results = []
    filenames = []
    
    print("evaluating files...")
    for file_info in matching_files:
        filename = file_info['filename']
        generated_path = file_info['generated_path']
        ground_truth_path = file_info['ground_truth_path']
        
        scores = evaluate_audio_metrics(generated_path, ground_truth_path)
        filenames.append(filename)
        results.append(scores)
    
    print(f"evaluated {len(filenames)} files")
    
    return filenames, results


def create_results_dataframe(filenames, results):
    dataframe = pd.DataFrame(results, index=filenames)
    return dataframe


def print_summary_statistics(dataframe):
    print("\nsummary statistics:")
    print("=" * 50)
    print(dataframe.describe())


def generate_evaluation_plots(dataframe, output_path='evaluation_report.png'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # boxplot
    ax = axes[0, 0]
    dataframe.boxplot(ax=ax)
    ax.set_title('metric distributions')
    ax.set_ylabel('score')
    
    # mean scores
    ax = axes[0, 1]
    means = dataframe.mean()
    ax.bar(means.index, means.values)
    ax.set_title('average scores')
    ax.set_ylabel('mean score')
    ax.set_ylim([0, max(means.values) * 1.2])
    
    # score trends
    ax = axes[1, 0]
    for col in dataframe.columns:
        ax.plot(dataframe[col], label=col, alpha=0.7)
    ax.set_title('score per file')
    ax.set_xlabel('file index')
    ax.set_ylabel('score')
    ax.legend()
    
    # histogram
    ax = axes[1, 1]
    dataframe.hist(bins=20, ax=ax, alpha=0.7)
    ax.set_title('score histograms')
    ax.set_xlabel('score')
    ax.set_ylabel('frequency')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nsaved: {output_path}")


def save_results_csv(dataframe, output_path='evaluation_results.csv'):
    dataframe.to_csv(output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='batch evaluate audio metrics')
    parser.add_argument('--generated-dir', '-g', type=str, default='audio/clean')
    parser.add_argument('--ground-truth-dir', '-gt', type=str, default='audio/noisy')
    parser.add_argument('--output-plot', '-o', type=str, default='evaluation_report.png')
    parser.add_argument('--output-csv', '-c', type=str, default='evaluation_results.csv')
    
    args = parser.parse_args()
    
    matching_files = get_matching_audio_files(args.generated_dir, args.ground_truth_dir)
    filenames, results = evaluate_batch_files(matching_files)
    dataframe = create_results_dataframe(filenames, results)
    
    print_summary_statistics(dataframe)
    generate_evaluation_plots(dataframe, args.output_plot)
    save_results_csv(dataframe, args.output_csv)
    
    plt.show()