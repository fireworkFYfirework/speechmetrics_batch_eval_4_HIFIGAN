from speechmetrics import load
import os

# Check if files exist
generated_audio = 'audio/noisy_testset_wav/p232_003.wav'
ground_truth_audio = 'audio/clean_testset_wav/p232_003.wav'

print(f"Checking files...")
print(f"Generated exists: {os.path.exists(generated_audio)}")
print(f"Reference exists: {os.path.exists(ground_truth_audio)}")

if not os.path.exists(generated_audio):
    print(f"ERROR: Cannot find {generated_audio}")
    exit(1)

if not os.path.exists(ground_truth_audio):
    print(f"ERROR: Cannot find {ground_truth_audio}")
    exit(1)

try:
    # Load metrics
    print("\nLoading metrics...")
    metrics = load(['nb_pesq', 'stoi', 'sisdr'], window=None)
    
    # Calculate scores
    print("Calculating scores...")
    scores = metrics(generated_audio, ground_truth_audio)
    
    # Print results
    print("\n=== Evaluation Results ===")
    for metric_name, score in scores.items():
        print(f"{metric_name.upper():10s}: {score:.4f}")
        
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()