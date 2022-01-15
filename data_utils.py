import os
import re

import datasets

import torch
from torch.nn.utils.rnn import pad_sequence

class MyConfig(dict):
  def __getattr__(self, name): return self[name]
  def __setattr__(self, name, value): self[name] = value

class ElectraDataProcessor():
    def __init__(self, hf_dset, hf_tokenizer, max_length, text_col='text', lines_delimiter='\n', apply_filter=True):
        self.hf_dset = hf_dset
        self.hf_tokenizer = hf_tokenizer
        self._max_length = max_length

        self.text_col = text_col
        self.lines_delimiter = lines_delimiter
        self.apply_filter = apply_filter

    def map(self, **kwargs) -> datasets.arrow_dataset.Dataset:
        num_proc = kwargs.pop('num_proc', os.cpu_count())
        cache_file_name = kwargs.pop('cache_file_name', None)
        if cache_file_name is not None:
            if not cache_file_name.endswith('.arrow'): 
                cache_file_name += '.arrow'        
            if '/' not in cache_file_name: 
                cache_dir = os.path.abspath(os.path.dirname(self.hf_dset.cache_files[0]['filename']))
                cache_file_name = os.path.join(cache_dir, cache_file_name)

        return self.hf_dset.map(
            function=self,
            batched=True,
            cache_file_name=cache_file_name,
            remove_columns=self.hf_dset.column_names,
            disable_nullable=True,
            input_columns=[self.text_col],
            writer_batch_size=10**4,
            num_proc=num_proc,
            **kwargs     
        )

    def __call__(self, texts):
        processed_data = {'input_ids':[], 'sentA_length':[]}
        
        for text in texts:
            self.old_line = []
            lines = re.split(self.lines_delimiter, text)
            for i, line in enumerate(lines):
                if re.fullmatch(r'\s*', line): continue
                line = line.strip().replace("\n", " ").replace("()","")
                
                line = self.hf_tokenizer.tokenize(line)
                if len(line) < 32 and self.apply_filter: continue
                line = line[:self._max_length - 2]

                if len(self.old_line) + len(line) <= self._max_length - 2:
                    self.old_line += line
                    if i != len(lines) - 1: continue
                    else: final_line = self.old_line
                else:
                    final_line = self.old_line
                    self.old_line = line
                
                token_ids, sentA_length = self.encode(final_line)                              
                processed_data['sentA_length'].append(sentA_length)
                processed_data['input_ids'].append(token_ids)

                if i == len(lines) -1 and len(self.old_line) + len(line) > self._max_length - 2:
                    token_ids, sentA_length = self.encode(self.old_line)                              
                    processed_data['sentA_length'].append(sentA_length)
                    processed_data['input_ids'].append(token_ids)

        return processed_data

    def encode(self, final_line):
        token_ids = self.hf_tokenizer.convert_tokens_to_ids(final_line)
        token_ids = [self.hf_tokenizer.cls_token_id] + token_ids + [self.hf_tokenizer.sep_token_id]
        
        sentA_length = len(token_ids)
        return token_ids, sentA_length 


class ElectraDataCollator():
    def __init__(self, hf_tokenizer, max_length):
        self.hf_tokenizer = hf_tokenizer
        self._max_length = max_length

    def __call__(self, samples):
        input_ids, sentA_lengths = [], []

        for s in samples:
            input_ids.append(s['input_ids'])
            sentA_lengths.append(s['sentA_length'].unsqueeze(0))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.hf_tokenizer.pad_token_id).long()
        sentA_lengths = torch.cat(sentA_lengths)

        masked_inputs, labels, is_mlm_applied = self._mask_tokens(input_ids, 
                                                                  mask_token_index =self.hf_tokenizer.mask_token_id, 
                                                                  special_token_indices=self.hf_tokenizer.all_special_ids) 
        return {
            'masked_inputs':  masked_inputs, 
            'sentA_lengths':  sentA_lengths, 
            'is_mlm_applied': is_mlm_applied, 
            'labels':         labels
        }

    def _mask_tokens(self, inputs, mask_token_index, special_token_indices, mlm_probability=0.15, original_prob=0.15, ignore_index=-100):

        device = inputs.device
        labels = inputs.clone() 

        # Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
        probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
        special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)

        for sp_id in special_token_indices:
            special_tokens_mask = special_tokens_mask | (inputs == sp_id) 
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0) 
        is_mlm_applied = torch.bernoulli(probability_matrix).bool()
        labels[~is_mlm_applied] = ignore_index  # We only compute loss on mlm applied tokens

        # mask  (mlm_probability * (1-orginal_prob))
        mask_prob = 1 - original_prob # 0.85
        mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & is_mlm_applied
        inputs[mask_token_mask] = mask_token_index 

        return inputs, labels, is_mlm_applied

