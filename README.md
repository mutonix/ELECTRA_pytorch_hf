### ELECTRA <br/>implemented by huggingface `Trainer`

****
#### How to use
distributed training supported
- Without deepspeed
<br >

    ```
    python -m torch.distributed.launch --nproc_per_node=8 \
    electra_pretrain.py \
        --datas wikipedia \
        --size large \
        --base_run_name electra_wiki
    # AdamW optimizer with no bias correction
    ```
    <br >

- Using deepspeed
    ```
    deepspeed --num_gpus=8 \
    electra_pretrain_dp.py \
        --datas wikipedia \
        --size large \
        --base_run_name electra_wiki
    # **Require gcc >= 5 and gcc <= 7 if using CUDA 10.2**
    # AdamW optimizer with bias correction
    ```
    
