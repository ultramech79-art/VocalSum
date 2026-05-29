import numpy as np
import mir_eval
from rouge_score import rouge_scorer

def evaluate_separation(reference_vocals, estimated_vocals, reference_accomp, estimated_accomp):
    """
    Computes Signal-to-Distortion Ratio (SDR), Signal-to-Interference Ratio (SIR), 
    and Signal-to-Artifacts Ratio (SAR) using mir_eval.
    Expects 1D numpy audio waveforms.
    """
    # mir_eval expects shape (n_sources, n_samples)
    references = np.stack([reference_vocals, reference_accomp], axis=0)
    estimates = np.stack([estimated_vocals, estimated_accomp], axis=0)
    
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(references, estimates)
    
    return {
        "vocal_sdr": float(sdr[0]),
        "vocal_sir": float(sir[0]),
        "vocal_sar": float(sar[0]),
        "accomp_sdr": float(sdr[1]),
        "accomp_sir": float(sir[1]),
        "accomp_sar": float(sar[1])
    }


def evaluate_summarization(predictions, references):
    """
    Computes ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores for text summarization.
    predictions and references should be lists of strings.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    
    r1, r2, rl, rlsum = [], [], [], []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores['rouge1'].fmeasure)
        r2.append(scores['rouge2'].fmeasure)
        rl.append(scores['rougeL'].fmeasure)
        rlsum.append(scores['rougeLsum'].fmeasure)
        
    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rl)),
        "rougeLsum": float(np.mean(rlsum))
    }


if __name__ == "__main__":
    # Small self-test
    ref_txt = ["The quick brown fox jumps over the lazy dog."]
    pred_txt = ["A fast brown fox leaped over a lazy dog."]
    print("ROUGE Self-Test:", evaluate_summarization(pred_txt, ref_txt))
