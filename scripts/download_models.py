import os
import torch
from transformers import MarianMTModel, MarianTokenizer, PegasusForConditionalGeneration, PegasusTokenizer
from vocalsum.audio import UNet

def download_checkpoints():
    print("Initializing/downloading model checkpoints...")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Download/Initialize U-Net Checkpoint
    unet_path = "checkpoints/unet.pt"
    if not os.path.exists(unet_path):
        print("U-Net weights not found. Creating a random initialization state-dict for testing...")
        model = UNet()
        torch.save(model.state_dict(), unet_path)
        print(f"Random initialization saved to {unet_path}")
    else:
        print("U-Net checkpoint already exists.")
        
    # 2. Download MarianMT model (Hindi-English translation)
    print("Downloading Helsinki-NLP/opus-mt-hi-en (MarianMT) weights...")
    hi_en_model = "Helsinki-NLP/opus-mt-hi-en"
    MarianTokenizer.from_pretrained(hi_en_model)
    MarianMTModel.from_pretrained(hi_en_model)
    
    # 3. Download PEGASUS model (Abstractive summarization)
    print("Downloading google/pegasus-cnn_dailymail (PEGASUS) weights...")
    pegasus_model = "google/pegasus-cnn_dailymail"
    PegasusTokenizer.from_pretrained(pegasus_model)
    PegasusForConditionalGeneration.from_pretrained(pegasus_model)
    
    print("All checkpoints downloaded and initialized successfully!")

if __name__ == "__main__":
    download_checkpoints()
