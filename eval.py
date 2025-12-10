from speechmetrics import load
from pysepm import fwSNRseg
from scipy.io import wavfile
from srmrpy import srmr
from semetrics import composite
import numpy as np
import librosa
import soundfile as sf
import warnings
import sys
from io import StringIO

warnings.filterwarnings('ignore', category=UserWarning)

def ensure_16k(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    
    if sr not in [8000, 16000]:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
        temp_path = audio_path.replace('.wav', '_16k.wav')
        sf.write(temp_path, audio, sr)
        return temp_path
    
    return audio_path


def evaluate_audio_metrics(generated_audio_path, ground_truth_audio_path):
    scores = {}

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
       intrusive_metrics = load(['mosnet', 'stoi', 'sisdr'], window=None)
    finally:
       sys.stdout = old_stdout
    intrusive_scores = intrusive_metrics(generated_audio_path, ground_truth_audio_path)
    scores.update(intrusive_scores)
    
    fs_gen, audio_gen = wavfile.read(generated_audio_path)
    audio_gen = audio_gen.astype(float)
    srmr_score = srmr(audio_gen, fs_gen, fast=True, norm=False)
    scores['srmr'] = srmr_score[0] if isinstance(srmr_score, tuple) else srmr_score
    
    fs_gt, audio_gt = wavfile.read(ground_truth_audio_path)
    audio_gt = audio_gt.astype(float)
    if fs_gen == fs_gt:
        fwssnr_score = fwSNRseg(audio_gt, audio_gen, fs_gen)
        scores['fwssnr'] = fwssnr_score
    
    gen_path = ensure_16k(generated_audio_path)
    gt_path = ensure_16k(ground_truth_audio_path)
    csig, cbak, covl, pesq_score, segsnr = composite(gt_path, gen_path)
    scores['csig'] = csig
    scores['cbak'] = cbak
    scores['covl'] = covl
    scores['pesq'] = pesq_score
    scores['segsnr'] = segsnr
    
    # clean scalar values
    for metric_name, score in scores.items():
        if isinstance(score, (list, np.ndarray)):
            while isinstance(score, (list, np.ndarray)) and len(score) > 0:
                score = score[0]
            scores[metric_name] = score
    
    return scores


def print_scores(scores):
    print("\nevaluation results:")
    print("=" * 50)
    for metric_name, score in scores.items():
        if isinstance(score, (int, float, np.number)):
            print(f"{metric_name:20s}: {score:.4f}")
        else:
            print(f"{metric_name:20s}: {score}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='evaluate audio metrics')
    parser.add_argument('--generated', '-g', type=str, default='audio/clean_testset_wav/p232_003.wav')
    parser.add_argument('--ground-truth', '-gt', type=str, default='audio/noisy_testset_wav/p232_003.wav')
    
    args = parser.parse_args()
    
    scores = evaluate_audio_metrics(args.generated, args.ground_truth)
    
    print_scores(scores)