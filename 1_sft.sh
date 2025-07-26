#!/bin/bash

# 你希望的显存阈值（单位：MiB）
THRESHOLD=81000

# 检查频率（秒）
INTERVAL=10

# 你的 Python 指令
PYTHON_CMD="nohup python -u train.py model=blank_model model.name_or_path=/data1/llms/Llama-3.2-3B-Instruct/  model.block_name=LlamaDecoderLayer datasets=[pp_all] loss=sft exp_name=Llama-3.2-3B-Instruct gradient_accumulation_steps=2 batch_size=32 eval_batch_size=16 trainer=FSDPTrainer sample_during_eval=false > sft26.txt"

echo "开始监控 CUDA:0 的显存空闲状态..."

while true; do
    # 提取 cuda:0 的显存信息（显存总量 - 使用量）
    MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)

    echo "当前空闲显存：${MEM_FREE} MiB"

    if [ "$MEM_FREE" -gt "$THRESHOLD" ]; then
        echo "显存充足，执行命令：$PYTHON_CMD"
        $PYTHON_CMD
        break
    else
        echo "显存不足，等待 $INTERVAL 秒后重试..."
        sleep $INTERVAL
    fi
done