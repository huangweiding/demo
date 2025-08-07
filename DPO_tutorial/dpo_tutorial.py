import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
import numpy as np


class PreferenceDataset(Dataset):
    """偏好数据集，包含prompt、chosen和rejected回答"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'prompt': item['prompt'],
            'chosen': item['chosen'],
            'rejected': item['rejected']
        }


def create_dummy_preference_data(num_samples: int = 100) -> List[Dict]:
    """创建示例偏好数据"""
    data = []
    prompts = [
        "解释什么是机器学习",
        "写一个Python函数",
        "什么是深度学习",
        "如何优化代码性能"
    ]
    
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        data.append({
            'prompt': prompt,
            'chosen': f"好的回答 {i}",
            'rejected': f"不好的回答 {i}"
        })
    
    return data


class DPOLoss(nn.Module):
    """DPO损失函数"""
    
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta
    
    def forward(self, policy_chosen_logps: torch.Tensor, 
                policy_rejected_logps: torch.Tensor,
                reference_chosen_logps: torch.Tensor,
                reference_rejected_logps: torch.Tensor) -> torch.Tensor:
        """
        计算DPO损失
        
        Args:
            policy_chosen_logps: 策略模型对chosen回答的log概率
            policy_rejected_logps: 策略模型对rejected回答的log概率
            reference_chosen_logps: 参考模型对chosen回答的log概率
            reference_rejected_logps: 参考模型对rejected回答的log概率
        """
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        
        losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
        return losses.mean()


class DPOTrainer:
    """DPO训练器"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", beta: float = 0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 策略模型（要训练的模型）
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # 参考模型（固定不变）
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        self.dpo_loss = DPOLoss(beta=beta)
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=1e-5)
    
    def get_logps(self, model: nn.Module, input_ids: torch.Tensor, 
                  attention_mask: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
        """计算log概率"""
        if not requires_grad:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
        # 计算每个token的log概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取实际token的log概率
        gathered_logps = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        
        # 计算序列的平均log概率
        return (gathered_logps * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1)
    
    def prepare_inputs(self, prompts: List[str], responses: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备模型输入"""
        texts = [f"{prompt} {response}" for prompt, response in zip(prompts, responses)]
        
        # Tokenize
        encodings = self.tokenizer(texts, padding=True, truncation=True, 
                                  max_length=512, return_tensors="pt")
        
        return encodings['input_ids'].to(self.device), encodings['attention_mask'].to(self.device)
    
    def train_step(self, batch: Dict) -> float:
        """单步训练"""
        self.policy_model.train()
        
        # 准备输入
        chosen_input_ids, chosen_attention_mask = self.prepare_inputs(
            batch['prompt'], batch['chosen'])
        rejected_input_ids, rejected_attention_mask = self.prepare_inputs(
            batch['prompt'], batch['rejected'])
        
        # 计算策略模型的log概率（需要梯度）
        policy_chosen_logps = self.get_logps(self.policy_model, chosen_input_ids, chosen_attention_mask, requires_grad=True)
        policy_rejected_logps = self.get_logps(self.policy_model, rejected_input_ids, rejected_attention_mask, requires_grad=True)
        
        # 计算参考模型的log概率（不需要梯度）
        reference_chosen_logps = self.get_logps(self.reference_model, chosen_input_ids, chosen_attention_mask, requires_grad=False)
        reference_rejected_logps = self.get_logps(self.reference_model, rejected_input_ids, rejected_attention_mask, requires_grad=False)
        
        # 计算DPO损失
        loss = self.dpo_loss(policy_chosen_logps, policy_rejected_logps,
                           reference_chosen_logps, reference_rejected_logps)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, train_dataloader: DataLoader, num_epochs: int = 3):
        """训练循环"""
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_dataloader:
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
                
                if num_batches % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss:.4f}")
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """生成回答"""
        self.policy_model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()


def main():
    """主函数：演示DPO训练"""
    print("=== DPO (Direct Preference Optimization) 训练演示 ===")
    
    # 创建训练数据
    print("1. 创建偏好数据集...")
    preference_data = create_dummy_preference_data(num_samples=200)
    dataset = PreferenceDataset(preference_data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 初始化DPO训练器
    print("2. 初始化DPO训练器...")
    trainer = DPOTrainer(model_name="microsoft/DialoGPT-small", beta=0.1)
    
    # 训练模型
    print("3. 开始训练...")
    trainer.train(dataloader, num_epochs=2)
    
    # 测试生成
    print("4. 测试生成...")
    test_prompt = "解释什么是人工智能"
    response = trainer.generate_response(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")
    
    print("DPO训练完成！")


if __name__ == "__main__":
    main()
