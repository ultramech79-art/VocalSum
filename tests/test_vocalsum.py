import os
import pytest
import torch
import numpy as np
from vocalsum.audio import UNet, compute_stft, reconstruct_istft
from vocalsum.models import Transcriber, HindiEnglishTranslator, PEGASUSSummarizer

def test_unet_dimensions():
    """
    Validates that the U-Net model forward pass runs and preserves dimensions.
    """
    model = UNet()
    # Mixed spectrogram shape mock: [Batch, Channels, Frequency, Time]
    # Input has 2049 frequency bins (typical for n_fft=4096)
    x = torch.randn(1, 1, 2049, 100)
    out = model(x)
    assert out.shape == x.shape, f"Output shape {out.shape} does not match input shape {x.shape}"


def test_stft_istft_reconstruction():
    """
    Validates that STFT followed by ISTFT reconstructs the signal correctly.
    """
    # Create 1 second of random noise at sample rate 22050
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz Sine Wave
    
    magnitude, phase = compute_stft(signal)
    reconstructed = reconstruct_istft(magnitude, phase)
    
    # Check length compatibility (might differ slightly due to framing/hop padding)
    min_len = min(len(signal), len(reconstructed))
    # Assert correlation or reconstruction fidelity is high
    assert np.allclose(signal[:min_len], reconstructed[:min_len], atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU accelerator for quick neural model initialization")
def test_translator():
    """
    Validates that translator performs basic translations.
    """
    translator = HindiEnglishTranslator()
    hindi_text = "नमस्ते दुनिया"
    english_text = translator.translate(hindi_text)
    assert len(english_text) > 0
    assert "hello" in english_text.lower() or "world" in english_text.lower()
