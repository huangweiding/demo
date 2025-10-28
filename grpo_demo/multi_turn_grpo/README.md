## Multi-Turn GRPO (TRL + TinyLlama)

### Dataset format (JSONL)
Each line contains a `conversation` array of `{role, content}` turns. The last turn must be `user`.

```json
{"conversation": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好，我能帮你什么？"}, {"role": "user", "content": "帮我写一个简短的中文问候语，适合群聊开场。"}]}
```

### Quickstart

```bash
python -u grpo_demo/multi_turn_grpo/train_multiturn_grpo.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data_path grpo_demo/multi_turn_grpo/sample_data.jsonl \
  --output_dir grpo_demo/multi_turn_grpo/runs/tinyllama_multiturn \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --max_new_tokens 256 \
  --bf16 True
```

### Notes
- Uses TRL GRPOTrainer with a light rule-based reward.
- See `dataset.py`, `prompting.py`, `reward_fns.py` for extensibility.
- For larger models/GPUs, adjust batch size and generations.


