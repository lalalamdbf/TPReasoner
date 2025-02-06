from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq


@dataclass
class TPCLDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        padded_labels = []
        for feature, (prompt_len, answer_len) in zip(batch, positions):
            if self.tokenizer.padding_side == "left":
                start, end = feature.size(0) - answer_len, feature.size(0)
                raise ValueError("Left padding is not supported by Path contrastive learning.")
            else:
                start, end = prompt_len, prompt_len + answer_len
            padded_tensor = self.label_pad_token_id * torch.ones_like(feature)
            padded_tensor[start:end] = feature[start:end]
            padded_labels.append(padded_tensor)
        return torch.stack(padded_labels, dim=0).contiguous()  # in contiguous memory

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        waiting
        """

        concatenated_features = []
        label_positions = []
        extract_indices = []
        anchor_num = []
        positive_num_0 = []
        positive_num_1 = []
        negative_num = []
        for feature in features:
            for i in range(2):
                prompt_len, answer_len = len(feature["prompt_ids"][i]), len(feature["response_ids"][i])
                concatenated_features.append(
                    {
                        "input_ids": feature["prompt_ids"][i] + feature["response_ids"][i],
                        "attention_mask": [1] * (prompt_len + answer_len),
                    }
                )
                label_positions.append((prompt_len, answer_len))
                extract_indices.append(feature["extract_indices"][i])
                anchor_num.append(feature["anchor_num"][i])
                positive_num_0.append(feature["positive_num_0"][i])
                positive_num_1.append(feature["positive_num_1"][i])
            
      
        batch = self.tokenizer.pad(
            concatenated_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        batch["labels"] = self._pad_labels(batch["input_ids"], label_positions)
        batch["anchor_num"] = torch.tensor(anchor_num, dtype=torch.long)
        batch["positive_num_0"] = torch.tensor(positive_num_0, dtype=torch.long)
        batch["positive_num_1"] = torch.tensor(positive_num_1, dtype=torch.long)
        
        # Determine the maximum length of extract_indices
        max_len = max(len(item) for sublist in extract_indices for item in sublist)
        # Pad the lists with zeros and convert to tensor
        batch["extract_indices"] = torch.tensor([[item + [0] * (max_len - len(item)) for item in sublist] for sublist in extract_indices], dtype=torch.long) 
        
        return batch
