"""
Training script for Tool Selection
Project 1: Lightweight Tool Selector for Edge AI Agents

Usage:
    python train.py --model tinybert --epochs 3
    python train.py --model all  # Train all models
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from dataset import ToolDataset, TOOLS
from model import ToolClassifier, MODEL_CONFIGS


def compute_metrics(predictions, labels, threshold=0.5):
    """Compute evaluation metrics"""
    # Convert to numpy
    preds = (predictions > threshold).cpu().numpy()
    labs = labels.cpu().numpy()
    
    # Compute metrics
    recall = recall_score(labs, preds, average='samples', zero_division=0)
    precision = precision_score(labs, preds, average='samples', zero_division=0)
    f1 = f1_score(labs, preds, average='samples', zero_division=0)
    
    # Tool recall (what TinyAgent reports)
    tool_recall = recall_score(labs.flatten(), preds.flatten(), zero_division=0)
    
    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'tool_recall': tool_recall
    }


def evaluate(model, dataloader, device):
    """Evaluate model on dataloader"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits)
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics = compute_metrics(all_preds, all_labels)
    return metrics


def train_model(
    model_name,
    train_path='data/train.json',
    val_path='data/val.json',
    output_dir='models',
    epochs=3,
    batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    max_length=128,
    seed=42
):
    """
    Train a single model.
    
    Args:
        model_name: One of "tinybert", "mobilebert", "distilbert", "deberta"
        train_path: Path to training data JSON
        val_path: Path to validation data JSON
        output_dir: Directory to save models
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio for scheduler
        max_length: Maximum sequence length
        seed: Random seed
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS[model_name])
    
    # Load datasets
    train_dataset = ToolDataset(train_path, tokenizer, max_length)
    val_dataset = ToolDataset(val_path, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize model
    model = ToolClassifier(model_name).to(device)
    num_params = model.get_num_parameters()
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_recall = 0.0
    history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluate
        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        print(f"  Tool Recall: {val_metrics['tool_recall']:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            **val_metrics
        })
        
        # Save best model
        if val_metrics['recall'] > best_recall:
            best_recall = val_metrics['recall']
            
            # Save model
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
            tokenizer.save_pretrained(model_dir)
            
            # Save config
            config = {
                'model_name': model_name,
                'hf_model': MODEL_CONFIGS[model_name],
                'num_labels': len(TOOLS),
                'max_length': max_length,
                'best_recall': best_recall
            }
            with open(os.path.join(model_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"  ✓ Saved best model (recall: {best_recall:.4f})")
    
    # Save training history
    history_path = os.path.join(output_dir, model_name, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best recall: {best_recall:.4f}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Tool Selection Models')
    parser.add_argument('--model', type=str, default='tinybert',
                        choices=['tinybert', 'mobilebert', 'distilbert', 'deberta', 'all'],
                        help='Model to train (or "all" for all models)')
    parser.add_argument('--train_path', type=str, default='data/train.json')
    parser.add_argument('--val_path', type=str, default='data/val.json')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model(s)
    if args.model == 'all':
        models_to_train = ['tinybert', 'mobilebert', 'distilbert', 'deberta']
    else:
        models_to_train = [args.model]
    
    results = {}
    for model_name in models_to_train:
        model, history = train_model(
            model_name=model_name,
            train_path=args.train_path,
            val_path=args.val_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed
        )
        results[model_name] = history[-1] if history else {}
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name}: Recall={metrics.get('recall', 0):.4f}, F1={metrics.get('f1', 0):.4f}")


if __name__ == "__main__":
    main()
