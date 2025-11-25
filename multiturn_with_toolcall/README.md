# 包含工具调用的多轮 GRPO 训练

这个项目展示了如何使用 GRPO (Group Relative Policy Optimization) 训练一个能够进行多轮对话并使用工具的语言模型。

## 功能特性

1. **多轮对话支持**：模型可以进行多轮对话，保持上下文连贯性
2. **工具调用**：支持多种工具调用，包括：
   - `calculator`: 数学计算
   - `weather_query`: 天气查询
   - `search`: 信息搜索
   - `get_current_time`: 获取当前时间
3. **自动工具执行**：自动检测模型输出中的工具调用请求，执行工具，并将结果返回给模型
4. **智能奖励函数**：基于工具使用质量、回答准确性等多维度评估模型表现
5. **完整 GRPO 训练流程**：实现完整的 GRPO 算法，包括相对优势计算、损失函数等

## 项目结构

```
multiturn_with_toolcall/
├── main.py                 # 主训练脚本
├── tools.py                # 工具定义和执行器
├── tool_parser.py          # 工具调用解析器
├── multi_turn_handler.py   # 多轮对话处理器
├── reward_functions.py     # 奖励函数
├── sample_data.jsonl       # 示例训练数据
└── README.md              # 本文档
```

## 核心组件说明

### 1. 工具系统 (`tools.py`)

定义了可用的工具和工具执行器：

- **Tool**: 工具定义类，包含工具名称、描述和参数定义
- **ToolExecutor**: 工具执行器，负责执行工具调用并返回结果

### 2. 工具调用解析 (`tool_parser.py`)

从模型输出中检测和解析工具调用：

- **ToolCallParser**: 解析模型输出中的 `<tool_call>...</tool_call>` 格式的工具调用
- 支持 JSON 格式的工具调用参数解析

### 3. 多轮对话处理 (`multi_turn_handler.py`)

处理包含工具调用的多轮对话：

- **MultiTurnHandler**: 
  - 检测工具调用请求
  - 执行工具
  - 将工具结果返回给模型
  - 继续对话直到完成

### 4. 奖励函数 (`reward_functions.py`)

评估模型表现的多维度奖励函数：

- **ToolCallRewardFunction**: 评估：
  - 工具调用格式和参数正确性
  - 工具使用的必要性
  - 工具结果的使用情况
  - 最终回答质量
  - 对话连贯性

### 5. 主训练脚本 (`main.py`)

整合所有组件，实现完整的 GRPO 训练流程：

- 加载模型和数据集
- 生成多个候选响应
- 计算奖励和优势
- 执行 GRPO 优化

## 使用方法

### 1. 准备数据

数据格式为 JSONL，每行包含一个对话：

```json
{"conversation": [{"role": "user", "content": "帮我计算一下 25 * 37 等于多少？"}]}
```

### 2. 运行训练

```bash
python main.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data_path sample_data.jsonl \
  --output_dir ./checkpoints \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 \
  --num_epochs 1 \
  --num_generations 4 \
  --max_new_tokens 256 \
  --temperature 0.9 \
  --seed 42 \
  --bf16
```

### 3. 参数说明

- `--model_name`: 预训练模型名称
- `--data_path`: 训练数据路径
- `--output_dir`: 模型保存目录
- `--batch_size`: 批次大小
- `--gradient_accumulation_steps`: 梯度累积步数
- `--learning_rate`: 学习率
- `--num_epochs`: 训练轮数
- `--num_generations`: 每个 prompt 生成的候选数量（GRPO 核心参数）
- `--max_new_tokens`: 最大生成 token 数
- `--temperature`: 采样温度
- `--bf16`: 是否使用 bfloat16 精度

## GRPO 算法原理

GRPO (Group Relative Policy Optimization) 是一种强化学习算法，专门用于语言模型微调：

1. **生成多个候选**：对每个 prompt 生成 G 个候选响应
2. **计算奖励**：对每个候选计算奖励分数
3. **计算相对优势**：在组内计算相对优势：`A_i = (r_i - mean(r)) / std(r)`
4. **优化策略**：使用裁剪的 surrogate objective 优化模型

### 工具调用的多轮训练

在多轮工具调用场景中：

1. **第一轮**：模型生成响应，可能包含工具调用
2. **工具执行**：系统检测并执行工具调用
3. **结果返回**：将工具结果返回给模型
4. **继续对话**：模型基于工具结果生成最终回答
5. **奖励计算**：评估整个对话过程的质量

## 工具调用格式

模型需要按照以下格式输出工具调用：

```xml
<tool_call>
{
    "name": "calculator",
    "arguments": {
        "expression": "25 * 37 + 128"
    }
}
</tool_call>
```

工具执行后，系统会返回结果：

```xml
<tool_result>
工具: calculator
结果: 1053
</tool_result>
```

## 奖励函数设计

奖励函数综合考虑以下因素：

1. **工具调用正确性** (30%)：工具调用格式和参数是否正确
2. **工具使用必要性** (20%)：是否在需要时使用了工具
3. **工具结果使用** (20%)：是否正确使用了工具结果
4. **最终回答质量** (20%)：最终回答是否准确、完整
5. **对话连贯性** (10%)：多轮对话是否连贯

## 扩展和定制

### 添加新工具

在 `tools.py` 中添加新工具定义：

```python
AVAILABLE_TOOLS["new_tool"] = Tool(
    name="new_tool",
    description="新工具描述",
    parameters={
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "参数1"}
        },
        "required": ["param1"]
    }
)
```

然后在 `ToolExecutor` 中添加执行逻辑。

### 自定义奖励函数

修改 `reward_functions.py` 中的 `RewardConfig` 和 `ToolCallRewardFunction` 类来调整奖励计算逻辑。

### 调整多轮对话参数

在 `main.py` 中修改 `MultiTurnHandler` 的参数：
- `max_turns`: 最大对话轮数
- `max_tool_calls_per_turn`: 每轮最大工具调用次数

## 注意事项

1. **计算安全性**：当前 `calculator` 工具使用 `eval()`，在生产环境中应使用更安全的表达式求值方法
2. **模型兼容性**：确保使用的模型支持工具调用格式的训练
3. **内存使用**：多轮对话和多个候选生成会消耗较多内存，注意调整批次大小
4. **训练稳定性**：GRPO 训练可能不稳定，建议使用较小的学习率和梯度裁剪

## 参考资源

- GRPO 论文和实现
- SWIFT 框架文档（多轮训练）
- TRL (Transformers Reinforcement Learning) 库
- OpenAI Function Calling 格式

## 许可证

本项目仅供学习和研究使用。

