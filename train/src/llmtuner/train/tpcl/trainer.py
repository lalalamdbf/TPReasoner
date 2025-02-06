from collections import defaultdict
from contextlib import nullcontext
import math
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union
# from collator import CPODataCollatorWithPadding
import torch
from transformers import BatchEncoding, Trainer
from trl import DPOTrainer
from trl.trainer.utils import disable_dropout_in_model
import torch.nn.functional as F

from ...extras.constants import IGNORE_INDEX
from ..utils import create_custom_optimzer, create_custom_scheduler
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import FinetuningArguments

an_logits_list = []
positive_logits_mean_list = []
class CustomTPCLTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        # ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            # if ref_model is not None:
            #     disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.reference_free = False
        self.use_dpo_data_collator = False # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        # self.ref_model = ref_model
        self.beta = finetuning_args.dpo_beta
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.loss_type = finetuning_args.dpo_loss
        self.ftx_gamma = finetuning_args.dpo_ftx
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")
        
        # if ref_model is not None:
        #     if self.is_deepspeed_enabled:
        #         if not (
        #             getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
        #         ):  # quantized models are already set on the correct device
        #             self.ref_model = self._prepare_deepspeed(self.ref_model)
        #     else:
        #         self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)


    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def sft_loss(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.Tensor:
        r"""
        Computes supervised cross-entropy loss of given labels under the given logits.

        Returns:
            A tensor of shape (batch_size,) containing the cross-entropy loss of each samples.
        """
        all_logps = self.get_batch_logps(logits, labels, average_log_prob=True)
        return -all_logps
        
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch_copied = BatchEncoding({k: v.detach().clone() for k, v in batch.items()})  # avoid error

        dict = model(
            input_ids=batch_copied["input_ids"],
            attention_mask=batch_copied["attention_mask"],
            output_hidden_states=True,
            return_dict=True)
        output = dict.hidden_states[-1]
        all_logits = dict.logits.to(torch.float32)
        all_logits_softmax = all_logits.clone().softmax(-1)
        
        return output, all_logits, all_logits_softmax
    
    def swap_elements(
        self, input: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = input.shape[0]
        
        indices = torch.arange(batch_size, device=input.device)
        indices[0:batch_size:2], indices[1:batch_size:2] = indices[1:batch_size:2].clone(), indices[0:batch_size:2].clone()
        
        return input[indices]
    

    
    def sep_output(
        self,
        output,
        batch):
        
        batch_size = output.shape[0]
        rows, cols= batch["extract_indices"].shape[1], batch["extract_indices"].shape[2]
        batch_indices = torch.arange(batch_size, device=output.device).view(-1, 1, 1).expand(-1, rows, cols)
        extract_paths = output[batch_indices, batch["extract_indices"]]
        
        extract_indices_mask = (batch["extract_indices"] > 0).long()
        extract_indices_mask = extract_indices_mask.unsqueeze(-1).type_as(extract_paths)
      
        masked_extract_paths = extract_paths * extract_indices_mask
   
        average_extract_paths = masked_extract_paths.sum(dim=2) / extract_indices_mask.sum(dim=2).clamp(min=1)
 
        average_extract_paths_copied = average_extract_paths.clone()
        swapped_average_extract_paths = self.swap_elements(average_extract_paths_copied)
        
        anchor_paths = torch.gather(average_extract_paths, 1, batch["anchor_num"].unsqueeze(-1).expand(-1, -1, average_extract_paths.size(-1))).squeeze(1)
        negative_paths = torch.gather(swapped_average_extract_paths, 1, batch["anchor_num"].unsqueeze(-1).expand(-1, -1, swapped_average_extract_paths.size(-1))).squeeze(1)
        positive_paths_0 = torch.gather(average_extract_paths, 1, batch["positive_num_0"].unsqueeze(-1).expand(-1, -1, average_extract_paths.size(-1))).squeeze(1)
        # swapped_positive_paths_0 = self.swap_elements(positive_paths_0.clone())
        positive_paths_1 = torch.gather(average_extract_paths, 1, batch["positive_num_1"].unsqueeze(-1).expand(-1, -1, average_extract_paths.size(-1))).squeeze(1)
        # swapped_positive_paths_1 = self.swap_elements(positive_paths_1.clone())
        
        return anchor_paths, negative_paths, positive_paths_0, positive_paths_1
    
    def tpcl_loss(
        self,
        anchor_paths: torch.FloatTensor,
        negative_paths: torch.FloatTensor,
        positive_paths_0: torch.FloatTensor,
        positive_paths_1: torch.FloatTensor,
        # ref_anchor_paths: torch.FloatTensor,
        # ref_negative_paths: torch.FloatTensor,
        # ref_positive_paths_0: torch.FloatTensor,
        # ref_positive_paths_1: torch.FloatTensor
    ):
        swapped_positive_paths_0 = self.swap_elements(positive_paths_0.clone())
        swapped_positive_paths_1 = self.swap_elements(positive_paths_1.clone())
        # swapped_reference_positive_paths_0 = self.swap_elements(ref_positive_paths_0.clone())
        # swapped_reference_positive_paths_1 = self.swap_elements(ref_positive_paths_1.clone())
        
        
        an_simratios = F.cosine_similarity(anchor_paths, negative_paths, -1)
        positive_simratios_0 = F.cosine_similarity(positive_paths_0, swapped_positive_paths_0, -1)
        positive_simratios_1 = F.cosine_similarity(positive_paths_1, swapped_positive_paths_1, -1)
        
        
        # if self.reference_free:
        #     raise ValueError("Not supported.")
        # else:
        #     ref_an_simratios =  F.cosine_similarity(ref_anchor_paths, ref_negative_paths, -1)
        #     ref_positive_simratios_0 = F.cosine_similarity(ref_positive_paths_0, swapped_reference_positive_paths_0 , -1)
        #     ref_positive_simratios_1 = F.cosine_similarity(ref_positive_paths_1, swapped_reference_positive_paths_1, -1)

        an_simratios = an_simratios.to(self.accelerator.device)
        positive_simratios_0 = positive_simratios_0.to(self.accelerator.device)
        positive_simratios_1 = positive_simratios_1.to(self.accelerator.device)
        
        # ref_an_simratios = ref_an_simratios.to(self.accelerator.device)
        # ref_positive_simratios_0 = ref_positive_simratios_0.to(self.accelerator.device)
        # ref_positive_simratios_1 = ref_positive_simratios_1.to(self.accelerator.device)
        
        # an_logits = an_simratios - ref_an_simratios
        # positive_logits_0 = positive_simratios_0 - ref_positive_simratios_0
        # positive_logits_1 = positive_simratios_1 - ref_positive_simratios_1
        
        an_logits = an_simratios
        positive_logits_0 = positive_simratios_0
        positive_logits_1 = positive_simratios_1

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        # self.beta = 1 / self.beta
        if self.loss_type == "sigmoid":
            losses = (
                - F.logsigmoid(1 / self.beta * ( positive_logits_0 - an_logits ))
                - F.logsigmoid(1 / self.beta * ( positive_logits_1 - an_logits ))
            ) / 2
            
            # losses = (
            #     - F.logsigmoid(an_logits)
                
            # ) 
        else:
            raise ValueError("Not supported.")

        
        return losses, an_logits, positive_logits_0, positive_logits_1
    
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Computes the infoNCE loss and sft loss.
        """
        metrics = {}
        output, logits, logits_softmax = self.concatenated_forward(model, batch)
        
        if self.beta > 1e-6:
            (anchor_paths,
            negative_paths,
            positive_paths_0,
            positive_paths_1) = self.sep_output(output, batch)
            
            cl_losses, an_logits, positive_logits_0, positive_logits_1 = self.tpcl_loss(
                anchor_paths,
                negative_paths,
                positive_paths_0,
                positive_paths_1,
            )


        if self.ftx_gamma > 1e-6:
            sft_loss = self.sft_loss(logits, batch["labels"])
            losses = self.ftx_gamma * sft_loss
            if self.beta > 1e-6:
                losses += cl_losses
            
        
        prefix = "eval_" if train_eval == "eval" else ""
        
        
        if self.ftx_gamma > 1e-6:
            metrics[f"{prefix}sft_loss"] = sft_loss.detach().cpu().mean()
        
        
        if self.beta > 1e-6:
            positive_logits_mean = (positive_logits_0 + positive_logits_1) / 2
            metrics[f"{prefix}an_logits"] = an_logits.detach().cpu().mean()
            metrics[f"{prefix}positive_logits_0"] = positive_logits_0.detach().cpu().mean()
            metrics[f"{prefix}positive_logits_1"] = positive_logits_1.detach().cpu().mean()
            metrics[f"{prefix}positive_logits_mean"] = positive_logits_mean.detach().cpu().mean()

        return losses.mean(), metrics
