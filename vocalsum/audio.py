import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import librosa
import soundfile as sf

    """
    U-Net convolutional neural network adapted for time-frequency spectrogram source separation.
    Consists of 6 Encoder blocks, a Bottleneck, and 6 Decoder blocks with skip connections.
    """
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        # Input shape: [Batch, 1, Frequency_bins (2049), Time_frames]
        # We will pad input to be divisible by 64 (2048 frequency bins)
        self.enc1 = self._conv_block(1, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        self.enc4 = self._conv_block(64, 128)
        self.enc5 = self._conv_block(128, 256)
        self.enc6 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder (with skip connections)
        self.dec6 = self._deconv_block(512, 256, dropout=0.5)
        self.dec5 = self._deconv_block(512, 128, dropout=0.5) # skip connection doubles input channels
        self.dec4 = self._deconv_block(256, 64, dropout=0.5)
        self.dec3 = self._deconv_block(128, 32, dropout=0.0)
        self.dec2 = self._deconv_block(64, 16, dropout=0.0)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _deconv_block(self, in_channels, out_channels, dropout=0.0):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Pad frequency dimension to be divisible by 64 (from 2049 to 2112)
        orig_height, orig_width = x.shape[2], x.shape[3]
        pad_h = (64 - orig_height % 64) % 64
        pad_w = (64 - orig_width % 64) % 64
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        
        # Encoder passes
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        
        # Bottleneck
        b = self.bottleneck(e6)
        
        # Decoder passes with skip connections
        d6 = self.dec6(b)
        d6_cat = torch.cat([d6, e5], dim=1)
        
        d5 = self.dec5(d6_cat)
        d5_cat = torch.cat([d5, e4], dim=1)
        
        d4 = self.dec4(d5_cat)
        d4_cat = torch.cat([d4, e3], dim=1)
        
        d3 = self.dec3(d4_cat)
        d3_cat = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e1], dim=1)
        
        mask = self.dec1(d2_cat)
        
        # Crop back to original shape
        mask = mask[:, :, :orig_height, :orig_width]
        return mask


def preprocess_audio(audio_path, target_sr=22050):
    """
    Resamples raw audio input to target_sr and converts to mono.
    """
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as e:
        # Fallback using soundfile if librosa fails
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr


def compute_stft(y, n_fft=4096, hop_length=1024):
    """
    Computes Short-Time Fourier Transform (STFT) of a signal.
    Returns: magnitude spectrogram, phase spectrogram
    """
    stft_matrix = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    return magnitude, phase


def reconstruct_istft(magnitude, phase, hop_length=1024):
    """
    Reconstructs waveform from magnitude and phase using Inverse STFT.
    """
    stft_matrix = magnitude * np.exp(1j * phase)
    return librosa.istft(stft_matrix, hop_length=hop_length)


    """
    Orchestrates the Audio Source Separation stage using U-Net.
    """
    def __init__(self, model_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def separate(self, audio_path, output_dir=None):
        """
        Separates vocal and accompaniment from the given audio file.
        Returns: vocal_waveform, accompaniment_waveform, sample_rate
        """
        y, sr = preprocess_audio(audio_path)
        magnitude, phase = compute_stft(y)
        
        # Normalize magnitude spectrogram (per-frequency mean/std normalization)
        mean = magnitude.mean(axis=1, keepdims=True)
        std = magnitude.std(axis=1, keepdims=True) + 1e-10
        norm_mag = (magnitude - mean) / std
        
        # Prepare for PyTorch model input: [Batch=1, Channels=1, Freq, Time]
        tensor_mag = torch.FloatTensor(norm_mag).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            vocal_mask = self.model(tensor_mag).squeeze(0).squeeze(0).cpu().numpy()
            
        # Vocal magnitude is the predicted mask multiplied by the original magnitude
        vocal_mag = vocal_mask * magnitude
        accomp_mag = (1.0 - vocal_mask) * magnitude
        
        # Reconstruct waveforms
        vocal_wave = reconstruct_istft(vocal_mag, phase)
        accomp_wave = reconstruct_istft(accomp_mag, phase)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            vocal_out = os.path.join(output_dir, "vocals.wav")
            accomp_out = os.path.join(output_dir, "accompaniment.wav")
            sf.write(vocal_out, vocal_wave, sr)
            sf.write(accomp_out, accomp_wave, sr)
            
        return vocal_wave, accomp_wave, sr


    """
    A simple PyTorch dataset interface for preprocessed MUSDB18 spectrogram tracks.
    Assumes numpy/pytorch tensors of mixed/vocal/accompaniment spectrograms.
    """
    def __init__(self, data_list):
        # data_list is a list of dicts: {"mixed": np.array, "vocals": np.array}
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        mixed = torch.FloatTensor(item["mixed"])
        vocals = torch.FloatTensor(item["vocals"])
        return mixed, vocals