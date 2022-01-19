import os, random
from pathlib import Path
from functools import partial
from datetime import datetime, timezone, timedelta
import argparse

import numpy as np
import torch

import datasets
from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining
from transformers import TrainingArguments, AdamW

from data_utils import MyConfig, ElectraDataProcessor, ElectraDataCollator
from train_utils import ElectraModel, ElectraLoss, ElectraTrainer, ElectraWandbCallback

os.environ['WANDB_PROJECT'] = 'electra_pretrain_large'
os.environ['WANDB_WATCH'] = 'false'


c = MyConfig({
    'base_run_name': 'origin', # run_name = {base_run_name}_{seed}
    'seed': 11081, # 11081 36 1188 76 1 4 4649 7 # None/False to randomly choose seed from [0,999999]

    'adam_bias_correction': False,
    'sampling_method': 'fp32_gumbel',
    'from_pretrained': True,

    'size': 'small',
    'datas': ['my_text'],

    'logger': 'wandb',
    'preprocess_dsets_num_proc': 1,
    'num_workers': 8,
    'n_gpus': 8, # only used for caculating the per device batch size
    'grad_acc_step':16,  
})


"""Pass arguments"""
parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--base_run_name", default=c.base_run_name, type=str)
parser.add_argument("--size", default=c.size, choices=["small", "base", "large"])
parser.add_argument("--datas", default=c.datas, nargs='+', type=str)
parser.add_argument("--num_workers", default=c.num_workers, type=int)
parser.add_argument("--n_gpus", default=c.n_gpus, type=int)
parser.add_argument("--local_rank", default=-1, type=int)

args = parser.parse_args()

for k, v in vars(args).items():
    c.update({k: v})

# Check and Default
assert c.sampling_method in ['fp32_gumbel', 'fp16_gumbel', 'multinomial']
for data in c.datas: assert data in ['wikipedia', 'bookcorpus', 'openwebtext', 'my_text']
assert c.logger in ['wandb', 'neptune', None, False]
if not c.base_run_name: c.base_run_name = str(datetime.now(timezone(timedelta(hours=+8))))[6:-13].replace(' ','').replace(':','').replace('-','')
if not c.seed: c.seed = random.randint(0, 999999)
c.run_name = f'{c.base_run_name}_{c.seed}'

# Setting of different sizes
i = ['small', 'base', 'large'].index(c.size)
c.mask_prob = [0.15, 0.15, 0.25][i]
c.lr = [5e-4, 2e-4, 2e-4][i]
c.bs = [128, 256, 2048][i]
c.steps = [10**6, 766*1000, 400*1000][i]
c.max_length = [128, 512, 512][i]
generator_size_divisor = [4, 3, 4][i]
disc_config = ElectraConfig.from_pretrained(f'google/electra-{c.size}-discriminator')
gen_config = ElectraConfig.from_pretrained(f'google/electra-{c.size}-generator')

# note that public electra-small model is actually small++ and don't scale down generator size 
gen_config.hidden_size = int(disc_config.hidden_size / generator_size_divisor)
gen_config.num_attention_heads = disc_config.num_attention_heads // generator_size_divisor
gen_config.intermediate_size = disc_config.intermediate_size // generator_size_divisor
hf_tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-{c.size}-generator")

# Path to data
Path('./datasets').mkdir(exist_ok=True)
Path('./checkpoints/pretrain').mkdir(exist_ok=True, parents=True)

dsets = []
ElectraProcessor = partial(ElectraDataProcessor, hf_tokenizer=hf_tokenizer, max_length=c.max_length, apply_filter=False)

if 'wikipedia' in c.datas:
    print('load/download wiki dataset')
    wiki = datasets.load_dataset('wikipedia', '20200501.en', cache_dir='./datasets')['train']
    print('load/create data from wiki dataset for ELECTRA')
    e_wiki = ElectraProcessor(wiki).map(cache_file_name=f"electra_wiki_{c.max_length}.arrow", num_proc=c.preprocess_dsets_num_proc)
    dsets.append(e_wiki)

# OpenWebText
if 'openwebtext' in c.datas:
    print('load/download OpenWebText Corpus')
    owt = datasets.load_dataset('openwebtext', cache_dir='./datasets')['train']
    print('load/create data from OpenWebText Corpus for ELECTRA')
    e_owt = ElectraProcessor(owt).map(cache_file_name=f"electra_owt_{c.max_length}.arrow", num_proc=c.preprocess_dsets_num_proc)
    dsets.append(e_owt)

if 'my_text' in c.datas:
    print('load/download my text')
    mt = datasets.load_dataset("text", data_files={"train": "urlsf_subset00-944_data.txt"}, cache_dir='./datasets')['train']
    print('load/create data from my text for ELECTRA')
    e_mytext = ElectraProcessor(mt).map(cache_file_name=f"electra_mt_{c.max_length}.arrow", num_proc=c.preprocess_dsets_num_proc)
    dsets.append(e_mytext)

assert len(dsets) == len(c.datas)

electra_dset = datasets.concatenate_datasets(dsets)
electra_dset.set_format(type='torch', columns=['input_ids', 'sentA_length'])

"""Pre-Training"""
torch.backends.cudnn.benchmark = True
random.seed(c.seed)
np.random.seed(c.seed)
torch.manual_seed(c.seed)

if c.size in ['base', 'large'] and c.from_pretrained:
    generator = ElectraForMaskedLM.from_pretrained(f"google/electra-{c.size}-generator")
    discriminator = ElectraForPreTraining.from_pretrained(f"google/electra-{c.size}-discriminator")
else:
    generator = ElectraForMaskedLM(gen_config)
    discriminator = ElectraForPreTraining(disc_config)
    discriminator.electra.embeddings = generator.electra.embeddings
    generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight

electra_model = ElectraModel(generator, discriminator, hf_tokenizer, sampling_method=c.sampling_method)
electra_loss_func = ElectraLoss(loss_weights=(1.0, 50.0))

electra_data_collator = ElectraDataCollator(hf_tokenizer, c.max_length)
AdamW_no_bias = AdamW(electra_model.parameters(), lr=c.lr, betas=(0.9, 0.99), eps=1e-5, weight_decay=0.01, correct_bias=False)

print('Initialize args')
training_args = TrainingArguments(
    run_name=f'{c.base_run_name}-{c.size}',
    output_dir=f'./pretrain/checkpoints/{c.base_run_name}-{c.size}',          # output directory
    logging_dir='./logs',            # directory for storing logs
    logging_steps=1,
    save_steps=10000,     # Number of updates steps before two checkpoint saves. default: 500
    save_total_limit=20, # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.
    dataloader_num_workers=c.num_workers,
    remove_unused_columns=False,
    gradient_accumulation_steps=c.grad_acc_step,
    per_device_train_batch_size=c.bs // c.n_gpus // c.grad_acc_step,  # batch size per device during training
    max_grad_norm=1.0,
    warmup_steps=10000,
    max_steps=100000,   # 100k
    seed=c.seed,
    fp16=True,
    local_rank=c.local_rank,
    # deepspeed='ds_config.json',
    # report_to=['wandb'],  # not needed, for using `ElectraWandbCallback`
)

print('Initialize trainer')
trainer = ElectraTrainer(
    model=electra_model,                 # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=electra_dset,          # training dataset
    data_collator=electra_data_collator,
    optimizers=(AdamW_no_bias, None),
    callbacks=[ElectraWandbCallback],
    loss_func=electra_loss_func,
)

print('Start training at ', datetime.now())
trainer.train()


