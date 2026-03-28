"""
Dataset class for Tool Selection
Project 1: Lightweight Tool Selector for Edge AI Agents
"""

import json
import torch
from torch.utils.data import Dataset

# 16 Tools from TinyAgent
TOOLS = [
    "compose_new_email",
    "reply_to_email", 
    "forward_email",
    "get_email_address",
    "get_phone_number",
    "send_sms",
    "create_calendar_event",
    "get_zoom_meeting_link",
    "create_note",
    "open_note",
    "append_note_content",
    "create_reminder",
    "open_and_get_file_path",
    "summarize_pdf",
    "maps_open_location",
    "maps_show_direction"
]

NUM_TOOLS = len(TOOLS)
TOOL2IDX = {tool: idx for idx, tool in enumerate(TOOLS)}
IDX2TOOL = {idx: tool for idx, tool in enumerate(TOOLS)}


class ToolDataset(Dataset):
    """
    Dataset for multi-label tool classification.
    
    Expected JSON format:
    [
        {"query": "Send email to John", "tools": ["compose_new_email", "get_email_address"]},
        {"query": "Create meeting tomorrow", "tools": ["create_calendar_event"]}
    ]
    """
    
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize query
        encoding = self.tokenizer(
            item["query"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Multi-label encoding (one-hot)
        labels = torch.zeros(NUM_TOOLS)
        for tool in item["tools"]:
            if tool in TOOL2IDX:
                labels[TOOL2IDX[tool]] = 1.0
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }


def create_sample_dataset():
    """Create a small sample dataset for testing"""
    samples = [
        {"query": "Send an email to John about the meeting", "tools": ["compose_new_email", "get_email_address"]},
        {"query": "Reply to Sarah's email with the document attached", "tools": ["reply_to_email", "get_email_address"]},
        {"query": "Create a calendar event for tomorrow at 3pm with the team", "tools": ["create_calendar_event", "get_email_address"]},
        {"query": "Send a text message to Mom saying I'll be late", "tools": ["send_sms", "get_phone_number"]},
        {"query": "Create a note called meeting notes in my work folder", "tools": ["create_note"]},
        {"query": "Set a reminder to call the dentist tomorrow", "tools": ["create_reminder"]},
        {"query": "Summarize the project report PDF", "tools": ["summarize_pdf", "open_and_get_file_path"]},
        {"query": "Show me directions to the nearest coffee shop", "tools": ["maps_show_direction"]},
        {"query": "Schedule a Zoom meeting with the marketing team", "tools": ["create_calendar_event", "get_zoom_meeting_link", "get_email_address"]},
        {"query": "Forward the budget email to my manager", "tools": ["forward_email", "get_email_address"]}
    ]
    return samples


if __name__ == "__main__":
    # Create sample data files
    import os
    
    samples = create_sample_dataset()
    
    # Split into train/val/test
    train_data = samples[:7]
    val_data = samples[7:9]
    test_data = samples[9:]
    
    os.makedirs("data", exist_ok=True)
    
    with open("data/train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open("data/val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    
    with open("data/test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print("Sample dataset created in data/ folder")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
