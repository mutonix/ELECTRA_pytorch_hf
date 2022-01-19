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
        input_ids, sentA_length = [], []

        for s in samples:
            input_ids.append(s['input_ids'])
            sentA_length.append(s['sentA_length'].unsqueeze(0))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.hf_tokenizer.pad_token_id).long()
        sentA_length = torch.cat(sentA_length)

        return {
            'input_ids':  input_ids, 
            'sentA_length':  sentA_length, 
        }



