from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from transformers import Trainer
from transformers.integrations import WandbCallback, rewrite_logs

"""Model & Loss"""
class ElectraModel(nn.Module): 
    def __init__(self, generator, discriminator, hf_tokenizer, sampling_method):
        super().__init__()
        self.generator, self.discriminator = generator, discriminator
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)
        self.hf_tokenizer = hf_tokenizer
        self.sampling_method = sampling_method

    def to(self, *args, **kwargs):
        "Also set dtype and device of contained gumbel distribution if needed"
        super().to(*args, **kwargs)
        a_tensor = next(self.parameters())
        device, dtype = a_tensor.device, a_tensor.dtype
        if self.sampling_method=='fp32_gumbel': dtype = torch.float32
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=dtype), 
                                                            torch.tensor(1., device=device, dtype=dtype))

    def forward(self, masked_inputs, sentA_lengths, is_mlm_applied, labels):
        """
        masked_inputs (Tensor[int]): (N, L)
        sentA_lenths (Tensor[int]): (N, )
        is_mlm_applied (Tensor[boolean]): (N, L), True for positions chosen by mlm probability 
        labels (Tensor[int]): (N, L), -100 for positions where are not mlm applied
        """

        attention_mask, token_type_ids = self._get_pad_mask_and_token_type(masked_inputs, sentA_lengths)

        gen_output = self.generator(masked_inputs, attention_mask, token_type_ids)
        # Prediction scores of the language modeling head (before softmax) (N, L, vocab_size)
        gen_logits = gen_output.logits  # gen_logits[0] == gen_logits.logits

        # reduce size to save space and speed
        # is_mlm_applied (N, L) 1 for applied
        mlm_gen_logits = gen_logits[is_mlm_applied, :] # (mlm_positions, vocab_size)
    
        with torch.no_grad():

            pred_toks = self.sample(mlm_gen_logits) # (mlm_positions, )

            # produce inputs for discriminator
            generated = masked_inputs.clone() # (N, L)

            generated[is_mlm_applied] = pred_toks # (N, L) 
            # produce labels for discriminator
            is_replaced = is_mlm_applied.clone() # (N, L)

            correct_tokens_mask = (pred_toks == labels[is_mlm_applied])
            is_replaced[is_mlm_applied] = ~correct_tokens_mask # (B,L)

            gen_acc = correct_tokens_mask.sum() / is_mlm_applied.sum() # accuracy
            mlm_mask_num = is_replaced.sum()

        disc_logits = self.discriminator(generated, attention_mask, token_type_ids)[0] # (B, L)

        # [float](mlm_postions, vocab_size) [float](N, L) [bool](N, L) [bool](N, L) [bool](N, L)
        return ((mlm_gen_logits, disc_logits, is_replaced, attention_mask, is_mlm_applied), labels,
                                    {'gen_acc': gen_acc.float(), 'mlm_mask_num': mlm_mask_num})

    def _get_pad_mask_and_token_type(self, input_ids, sentA_lengths):
        """
        Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask and token_type_ids and won't be unnecessarily large, thus, prevent cpu processes loading batches from consuming lots of cpu memory and slow down the machine. 
        """
        attention_mask = input_ids != self.hf_tokenizer.pad_token_id
        seq_len = input_ids.shape[1]
        token_type_ids = torch.tensor([([0]*len + [1]*(seq_len-len)) for len in sentA_lengths.tolist()],  
                                        device=input_ids.device)
        return attention_mask, token_type_ids

    def sample(self, logits):
        "Reimplement gumbel softmax cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16 (https://github.com/pytorch/pytorch/issues/41663). Gumbel softmax is equal to what official ELECTRA code do, standard gumbel dist. = -ln(-ln(standard uniform dist.))"

        if self.sampling_method == 'fp32_gumbel':
            return (logits.float() + self.gumbel_dist.sample(logits.shape)).argmax(dim=-1)
        elif self.sampling_method == 'fp16_gumbel': # 5.06 ms
            return (logits + self.gumbel_dist.sample(logits.shape)).argmax(dim=-1)

        elif self.sampling_method == 'multinomial': # 2.X ms
            return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()


class ElectraLoss():
    def __init__(self, loss_weights=(1.0, 50.0)):
        self.loss_weights = loss_weights
        self.gen_loss_fc = nn.CrossEntropyLoss()
        self.disc_loss_fc = nn.BCEWithLogitsLoss()
    
    def __call__(self, pred, targ_ids):
        # non_pad: attention mask -> false for pad
        # targ_ids: label (N, L) -100 to ignore
        # mlm_gen_logits (mlm_postions, vocab_size) targ_ids[is_mlm_applied] (mlm_postions, )
        mlm_gen_logits, disc_logits, is_replaced, non_pad, is_mlm_applied = pred
        gen_loss = self.gen_loss_fc(mlm_gen_logits.float(), targ_ids[is_mlm_applied]) * self.loss_weights[0]

        disc_logits_flat = disc_logits.masked_select(non_pad) # [float](N, L) -> 1d tensor
        is_replaced_flat = is_replaced.masked_select(non_pad) # [bool](N, L) -> 1d tensor

        # caculate metrics: disc_real_mlm_acc, disc_acc
        with torch.no_grad():
            is_replaced_logits = disc_logits[is_replaced]
            disc_real_mlm_acc = (is_replaced_logits >= 0.).sum() / is_replaced_logits.size(0)
            disc_acc = ((disc_logits_flat >= 0.) == is_replaced_flat).sum() / disc_logits_flat.size(0)

        disc_loss = self.disc_loss_fc(disc_logits_flat.float(), is_replaced_flat.float()) * self.loss_weights[1]
        return (gen_loss + disc_loss, 
                {'disc_real_mlm_acc': disc_real_mlm_acc, 'disc_acc': disc_acc, 'gen_loss': gen_loss, 'disc_loss': disc_loss})

class ElectraTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_func = kwargs.pop('loss_func', None)
        super().__init__(*args, **kwargs)
        self.train_metrics = TrainMetric()

    def compute_loss(self, model, inputs, return_outputs=False):

        outputs = model(**inputs)
        model_outputs, labels, model_metrics = outputs
        self.train_metrics.update(model_metrics)

        loss_outputs = self.loss_func(model_outputs, labels)
        loss, loss_metrics = loss_outputs
        self.train_metrics.update(loss_metrics)

        self.state.train_metrics = self.train_metrics

        return (loss, model_outputs) if return_outputs else loss

@dataclass
class TrainMetric():
    gen_acc: torch.FloatTensor = field(default=torch.tensor(0.))
    disc_acc: torch.FloatTensor = field(default=torch.tensor(0.))
    disc_mlm_acc: torch.FloatTensor = field(default=torch.tensor(0.))
    mlm_mask_num: torch.LongTensor = field(default=torch.tensor(0.).long())

    def update(self, metric_dict: Dict):
        for k, v in metric_dict.items(): setattr(self, k, v)
            

class ElectraWandbCallback(WandbCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            logs.update(vars(state.train_metrics))
            self._wandb.log({**logs, "train/global_step": state.global_step})