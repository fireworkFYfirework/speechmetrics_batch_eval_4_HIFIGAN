from speechmetrics import load

# Step 1: Load the metrics you want
# For relative metrics (need reference audio):
metrics = load([ 'mosnet', 'stoi', 'sisdr'], window=None)

# Step 2: Specify your audio files
generated_audio = 'audio/clean_testset_wav/p232_003.wav'
ground_truth_audio = 'audio/noisy_testset_wav/p232_003.wav'

# Step 3: Calculate scores
# IMPORTANT: estimate first, reference second!
scores = metrics(generated_audio, ground_truth_audio)

# Step 4: Print results
print("Evaluation Results:")
for metric_name, score in scores.items():
    print(f"{metric_name}: {score}")