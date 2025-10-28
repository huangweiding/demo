import json
from dataclasses import dataclass
from typing import Dict, List, Any

from torch.utils.data import Dataset


class MultiTurnJsonlDataset(Dataset):
    """
    Each JSONL line:
    {
      "conversation": [{"role": "user|assistant", "content": "..."}, ...],
      "target": "optional reference for last assistant reply"
    }
    """

    def __init__(self, jsonl_path: str):
        self.samples: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        # Ensure minimal fields
        conversation = item.get("conversation", [])
        assert conversation and conversation[-1]["role"].lower() == "user", (
            "Last turn must be a user message for prompting"
        )
        return item


def collate_for_generation(batch: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    # We keep raw conversation; GRPOTrainer will use formatting_func to build prompts
    conversations = [b["conversation"] for b in batch]
    last_users = [b["conversation"][-1]["content"] for b in batch]
    targets = [b.get("target", "") for b in batch]
    return {
        "conversation": conversations,
        "last_user": last_users,
        "target": targets,
    }


