import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from .audio import UNet, MUSDBDataset

# U-Net Training
def train_unet(dataset_dir, epochs=100, lr=1e-4, batch_size=8, save_path="checkpoints/unet.pt"):
    """
    Trains the U-Net separation model on preprocessed MUSDB18 magnitude spectrograms.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training U-Net on {device}...")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Placeholder: In practice, load your preprocessed numpy files
    # Here we mock training with synthetic spectrogram matrices if dataset is missing
    # Shape format: [Batch, 1, Freq_bins (2049), Time_frames]
    mock_data = []
    for _ in range(32):
        mixed = torch.randn(1, 2049, 128)
        vocals = torch.randn(1, 2049, 128)
        mock_data.append({"mixed": mixed, "vocals": vocals})
        
    dataset = MUSDBDataset(mock_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for mixed, vocals in dataloader:
            mixed, vocals = mixed.to(device), vocals.to(device)
            
            # Simple U-Net Mask Separation optimization:
            # We predict the vocal mask, then multiply by mixed to get vocal spectrogram
            # Or simply train U-Net to map mixed spectrogram to vocal spectrogram directly
            pred_mask = model(mixed)
            pred_vocals = pred_mask * mixed
            
            loss = criterion(pred_vocals, vocals)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f}")
            
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


# PEGASUS Fine-Tuning on SAMSum
def train_pegasus(epochs=10, lr=1e-4, batch_size=1, grad_accum=16, save_path="checkpoints/pegasus"):
    """
    Fine-tunes the PEGASUS summarization model on the SAMSum dialogue dataset.
    """
    print("Fine-tuning PEGASUS on SAMSum dialogue dataset...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = "google/pegasus-cnn_dailymail"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # Load SAMSum dataset from Hugging Face Hub
    dataset = load_dataset("samsum")
    
    def preprocess_function(examples):
        inputs = examples["dialogue"]
        targets = examples["summary"]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
        labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["id", "dialogue", "summary"])
    
    training_args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(save_path)
    print(f"PEGASUS fine-tuned model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VocalSum Models")
    parser.add_argument("--model", type=str, choices=["unet", "pegasus"], required=True, help="Model to train")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset directory path")
    args = parser.parse_args()
    
    if args.model == "unet":
        epochs = args.epochs or 100
        train_unet(dataset_dir=args.dataset_dir, epochs=epochs)
    elif args.model == "pegasus":
        epochs = args.epochs or 10
        train_pegasus(epochs=epochs)
