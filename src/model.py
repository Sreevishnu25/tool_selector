"""
Model definitions for Tool Selection
Project 1: Lightweight Tool Selector for Edge AI Agents
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# Model configurations - HuggingFace model names
MODEL_CONFIGS = {
    "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
    "mobilebert": "google/mobilebert-uncased",
    "distilbert": "distilbert-base-uncased",
    "deberta": "microsoft/deberta-v3-small"
}

# Model sizes (approximate parameters in millions)
MODEL_SIZES = {
    "tinybert": 14.5,
    "mobilebert": 25.3,
    "distilbert": 66.0,
    "deberta": 44.0
}


class ToolClassifier(nn.Module):
    """
    Multi-label classifier for tool selection.
    
    Architecture:
        Input Query -> BERT Encoder -> [CLS] token -> FC Layer -> Sigmoid -> Tool Probabilities
    
    Args:
        model_name: One of "tinybert", "mobilebert", "distilbert", "deberta"
        num_labels: Number of tools (default: 16)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, model_name, num_labels=16, dropout=0.1):
        super().__init__()
        
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(MODEL_CONFIGS[model_name])
        hidden_size = self.encoder.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Store config for later use
        self.hidden_size = hidden_size
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            logits: Raw logits [batch_size, num_labels]
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classifier
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits
    
    def predict(self, input_ids, attention_mask, threshold=0.5):
        """
        Predict tools with threshold.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            threshold: Probability threshold (default: 0.5)
        
        Returns:
            predictions: Binary predictions [batch_size, num_labels]
            probabilities: Sigmoid probabilities [batch_size, num_labels]
        """
        logits = self.forward(input_ids, attention_mask)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()
        return predictions, probabilities
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_encoder(self):
        """Freeze encoder weights (for transfer learning)"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights"""
        for param in self.encoder.parameters():
            param.requires_grad = True


def get_model_info(model_name):
    """Get model information"""
    model = ToolClassifier(model_name)
    return {
        "name": model_name,
        "hf_name": MODEL_CONFIGS[model_name],
        "hidden_size": model.hidden_size,
        "params_millions": model.get_num_parameters() / 1e6,
        "approx_size_mb": MODEL_SIZES[model_name] * 4  # Rough estimate for FP32
    }


if __name__ == "__main__":
    # Test all models
    print("=" * 60)
    print("Model Information")
    print("=" * 60)
    
    for model_name in MODEL_CONFIGS.keys():
        info = get_model_info(model_name)
        print(f"\n{model_name.upper()}")
        print(f"  HuggingFace: {info['hf_name']}")
        print(f"  Hidden Size: {info['hidden_size']}")
        print(f"  Parameters: {info['params_millions']:.2f}M")
        print(f"  Approx Size: {info['approx_size_mb']:.1f}MB (FP32)")
    
    # Quick forward pass test
    print("\n" + "=" * 60)
    print("Forward Pass Test")
    print("=" * 60)
    
    from transformers import AutoTokenizer
    
    model_name = "tinybert"  # Smallest for quick test
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS[model_name])
    model = ToolClassifier(model_name)
    model.eval()
    
    query = "Send an email to John and create a meeting"
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.sigmoid(logits)
    
    print(f"\nQuery: {query}")
    print(f"Output shape: {logits.shape}")
    print(f"Top 3 tool probabilities:")
    
    from dataset import IDX2TOOL
    top_indices = torch.argsort(probs[0], descending=True)[:3]
    for idx in top_indices:
        print(f"  {IDX2TOOL[idx.item()]}: {probs[0][idx]:.4f}")
