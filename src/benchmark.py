"""
Benchmarking script for Tool Selection
Project 1: Lightweight Tool Selector for Edge AI Agents

This script measures:
- Inference latency (ms)
- Accuracy metrics
- Model size

Usage:
    python benchmark.py --model tinybert
    python benchmark.py --model all --device cpu
"""

import os
import argparse
import json
import time
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import ToolDataset, TOOLS, IDX2TOOL
from model import ToolClassifier, MODEL_CONFIGS, MODEL_SIZES


def measure_latency(model, tokenizer, queries, device, num_runs=100, warmup_runs=10):
    """
    Measure inference latency.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        queries: List of sample queries
        device: Device to run on
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs
    
    Returns:
        dict: Latency statistics in milliseconds
    """
    model.eval()
    model.to(device)
    
    # Prepare sample input
    sample_query = queries[0] if queries else "Send an email to John about the meeting"
    inputs = tokenizer(
        sample_query, 
        return_tensors='pt', 
        padding=True, 
        truncation=True,
        max_length=128
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Warmup runs
    print(f"  Warmup ({warmup_runs} runs)...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_ids, attention_mask)
    
    # Synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure latency
    print(f"  Measuring ({num_runs} runs)...")
    latencies = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            # Start timer
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            # Inference
            _ = model(input_ids, attention_mask)
            
            # End timer
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            # Record latency in milliseconds
            latencies.append((end - start) * 1000)
    
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p90_ms': np.percentile(latencies, 90),
        'p99_ms': np.percentile(latencies, 99),
        'num_runs': num_runs
    }


def measure_accuracy(model, tokenizer, test_path, device, threshold=0.5):
    """
    Measure accuracy metrics on test set.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_path: Path to test data
        device: Device to run on
        threshold: Prediction threshold
    
    Returns:
        dict: Accuracy metrics
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    model.eval()
    model.to(device)
    
    # Load test data
    test_dataset = ToolDataset(test_path, tokenizer)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"  Evaluating on {len(test_dataset)} samples...")
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            labels = sample['labels']
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs > threshold).float()
            
            all_preds.append(preds.squeeze())
            all_labels.append(labels)
            all_probs.append(probs.squeeze())
    
    all_preds = torch.stack(all_preds).numpy()
    all_labels = torch.stack(all_labels).numpy()
    all_probs = torch.stack(all_probs).numpy()
    
    # Compute metrics
    recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
    tool_recall = recall_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)
    
    # Average tools retrieved
    avg_tools = np.mean(np.sum(all_preds, axis=1))
    
    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'tool_recall': tool_recall,
        'avg_tools_retrieved': avg_tools,
        'num_samples': len(test_dataset)
    }


def get_model_size(model):
    """Get model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def benchmark_model(
    model_name,
    model_dir='models',
    test_path='data/test.json',
    device='cpu',
    num_runs=100
):
    """
    Benchmark a single model.
    
    Args:
        model_name: Model name
        model_dir: Directory containing saved models
        test_path: Path to test data
        device: Device to run on ("cpu" or "cuda")
        num_runs: Number of latency measurement runs
    
    Returns:
        dict: Complete benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name.upper()}")
    print(f"{'='*60}")
    
    device = torch.device(device)
    print(f"Device: {device}")
    
    # Load model
    model_path = os.path.join(model_dir, model_name)
    
    if os.path.exists(os.path.join(model_path, 'model.pt')):
        # Load trained model
        print(f"Loading trained model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = ToolClassifier(model_name)
        model.load_state_dict(torch.load(
            os.path.join(model_path, 'model.pt'),
            map_location=device
        ))
    else:
        # Load pretrained model (for testing without training)
        print(f"Loading pretrained model (not fine-tuned)")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS[model_name])
        model = ToolClassifier(model_name)
    
    model.to(device)
    model.eval()
    
    # Get model info
    num_params = model.get_num_parameters()
    model_size_mb = get_model_size(model)
    
    print(f"Parameters: {num_params / 1e6:.2f}M")
    print(f"Model size: {model_size_mb:.2f}MB")
    
    # Sample queries for latency measurement
    sample_queries = [
        "Send an email to John about the meeting",
        "Create a calendar event for tomorrow",
        "Set a reminder to call mom",
        "Summarize the project report PDF",
        "Show me directions to the coffee shop"
    ]
    
    # Measure latency
    print("\nMeasuring latency...")
    latency_results = measure_latency(
        model, tokenizer, sample_queries, device, num_runs=num_runs
    )
    
    print(f"  Mean: {latency_results['mean_ms']:.2f}ms ± {latency_results['std_ms']:.2f}ms")
    print(f"  P50: {latency_results['p50_ms']:.2f}ms")
    print(f"  P99: {latency_results['p99_ms']:.2f}ms")
    
    # Measure accuracy (if test data exists)
    accuracy_results = {}
    if os.path.exists(test_path):
        print("\nMeasuring accuracy...")
        accuracy_results = measure_accuracy(model, tokenizer, test_path, device)
        print(f"  Recall: {accuracy_results['recall']:.4f}")
        print(f"  Precision: {accuracy_results['precision']:.4f}")
        print(f"  F1: {accuracy_results['f1']:.4f}")
        print(f"  Tool Recall: {accuracy_results['tool_recall']:.4f}")
        print(f"  Avg Tools: {accuracy_results['avg_tools_retrieved']:.2f}")
    
    # Compile results
    results = {
        'model_name': model_name,
        'hf_model': MODEL_CONFIGS[model_name],
        'device': str(device),
        'num_params_millions': num_params / 1e6,
        'model_size_mb': model_size_mb,
        'latency': latency_results,
        'accuracy': accuracy_results
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark Tool Selection Models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['tinybert', 'mobilebert', 'distilbert', 'deberta', 'all'],
                        help='Model to benchmark')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--test_path', type=str, default='data/test.json')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--output', type=str, default='results/benchmark_results.json')
    
    args = parser.parse_args()
    
    # Models to benchmark
    if args.model == 'all':
        models = ['tinybert', 'mobilebert', 'distilbert', 'deberta']
    else:
        models = [args.model]
    
    # Run benchmarks
    all_results = {}
    for model_name in models:
        results = benchmark_model(
            model_name=model_name,
            model_dir=args.model_dir,
            test_path=args.test_path,
            device=args.device,
            num_runs=args.num_runs
        )
        all_results[model_name] = results
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'Params':<10} {'Size(MB)':<10} {'Latency(ms)':<15} {'Recall':<10}")
    print("-" * 80)
    
    for model_name, results in all_results.items():
        params = f"{results['num_params_millions']:.1f}M"
        size = f"{results['model_size_mb']:.1f}"
        latency = f"{results['latency']['mean_ms']:.2f}±{results['latency']['std_ms']:.2f}"
        recall = f"{results['accuracy'].get('recall', 0):.4f}" if results['accuracy'] else "N/A"
        print(f"{model_name:<15} {params:<10} {size:<10} {latency:<15} {recall:<10}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
