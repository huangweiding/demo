"""
LoRA (Low-Rank Adaptation) 训练仿真代码
=====================================

LoRA是一种参数高效的微调方法，通过低秩矩阵分解来近似全参数更新。
本代码演示了LoRA的核心概念和训练过程。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import time

class LoRALayer:
    """
    LoRA层的实现
    将原始权重矩阵 W 分解为 W + ΔW，其中 ΔW = A * B
    A 和 B 是低秩矩阵，rank << min(input_dim, output_dim)
    """
    
    def __init__(self, input_dim: int, output_dim: int, rank: int, alpha: float = 16.0):
        """
        初始化LoRA层
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度  
            rank: LoRA的秩（低秩矩阵的维度）
            alpha: 缩放因子，用于控制LoRA的影响强度
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        
        # 初始化低秩矩阵 A 和 B
        # A: [input_dim, rank], B: [rank, output_dim]
        # 使用Xavier初始化
        self.A = np.random.normal(0, np.sqrt(2.0 / input_dim), (input_dim, rank))
        self.B = np.zeros((rank, output_dim))  # B初始化为0，确保训练开始时ΔW=0
        
        # 存储梯度
        self.grad_A = np.zeros_like(self.A)
        self.grad_B = np.zeros_like(self.B)
        
        # 存储原始权重矩阵（在真实场景中，这是预训练模型的权重）
        self.original_W = np.random.normal(0, 0.1, (input_dim, output_dim))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            
        Returns:
            输出数据 [batch_size, output_dim]
        """
        # 计算LoRA的增量权重 ΔW = A * B
        delta_W = self.A @ self.B
        
        # 应用缩放因子
        scaled_delta_W = (self.alpha / self.rank) * delta_W
        
        # 计算最终权重 W_final = W_original + ΔW
        final_W = self.original_W + scaled_delta_W
        
        # 线性变换: y = x * W_final
        return x @ final_W
    
    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        反向传播
        
        Args:
            grad_output: 来自下一层的梯度 [batch_size, output_dim]
            x: 前向传播时的输入 [batch_size, input_dim]
            
        Returns:
            传递给上一层的梯度 [batch_size, input_dim]
        """
        # 计算LoRA增量权重的梯度
        delta_W = self.A @ self.B
        scaled_delta_W = (self.alpha / self.rank) * delta_W
        final_W = self.original_W + scaled_delta_W
        
        # 计算对输入的梯度
        grad_input = grad_output @ final_W.T
        
        # 计算对A的梯度: ∂L/∂A = x^T * grad_output * B^T
        self.grad_A = x.T @ grad_output @ self.B.T
        
        # 计算对B的梯度: ∂L/∂B = A^T * x^T * grad_output
        self.grad_B = self.A.T @ x.T @ grad_output
        
        return grad_input
    
    def update_parameters(self, learning_rate: float):
        """
        更新LoRA参数
        
        Args:
            learning_rate: 学习率
        """
        # 更新A矩阵
        self.A -= learning_rate * self.grad_A
        
        # 更新B矩阵
        self.B -= learning_rate * self.grad_B
        
        # 清零梯度
        self.grad_A.fill(0)
        self.grad_B.fill(0)

class LoRANetwork:
    """
    LoRA网络，包含多个LoRA层
    """
    
    def __init__(self, layer_dims: List[Tuple[int, int]], rank: int, alpha: float = 16.0):
        """
        初始化LoRA网络
        
        Args:
            layer_dims: 每层的(输入维度, 输出维度)列表
            rank: LoRA的秩
            alpha: 缩放因子
        """
        self.layers = []
        for input_dim, output_dim in layer_dims:
            layer = LoRALayer(input_dim, output_dim, rank, alpha)
            self.layers.append(layer)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        网络前向传播
        
        Args:
            x: 输入数据
            
        Returns:
            网络输出
        """
        current_input = x
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        网络反向传播
        
        Args:
            grad_output: 输出梯度
            x: 原始输入
            
        Returns:
            输入梯度
        """
        # 存储每层的输入，用于反向传播
        layer_inputs = [x]
        current_input = x
        for layer in self.layers[:-1]:
            current_input = layer.forward(current_input)
            layer_inputs.append(current_input)
        
        # 反向传播
        grad = grad_output
        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(grad, layer_inputs[i])
        
        return grad
    
    def update_parameters(self, learning_rate: float):
        """
        更新所有层的参数
        
        Args:
            learning_rate: 学习率
        """
        for layer in self.layers:
            layer.update_parameters(learning_rate)

def generate_training_data(num_samples: int, input_dim: int, output_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成训练数据
    
    Args:
        num_samples: 样本数量
        input_dim: 输入维度
        output_dim: 输出维度
        
    Returns:
        (输入数据, 目标数据)
    """
    # 生成随机输入数据
    X = np.random.normal(0, 1, (num_samples, input_dim))
    
    # 生成目标数据（简单的线性关系 + 噪声）
    true_weights = np.random.normal(0, 0.1, (input_dim, output_dim))
    y = X @ true_weights + np.random.normal(0, 0.01, (num_samples, output_dim))
    
    return X, y

def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    计算均方误差
    
    Args:
        predictions: 预测值
        targets: 目标值
        
    Returns:
        均方误差
    """
    return np.mean((predictions - targets) ** 2)

def train_lora_network(network: LoRANetwork, X: np.ndarray, y: np.ndarray, 
                      epochs: int, learning_rate: float, batch_size: int = 32) -> List[float]:
    """
    训练LoRA网络
    
    Args:
        network: LoRA网络
        X: 训练输入
        y: 训练目标
        epochs: 训练轮数
        learning_rate: 学习率
        batch_size: 批次大小
        
    Returns:
        每轮的损失值列表
    """
    losses = []
    num_samples = X.shape[0]
    
    print(f"开始训练LoRA网络...")
    print(f"网络结构: {[(layer.input_dim, layer.output_dim) for layer in network.layers]}")
    print(f"LoRA秩: {network.layers[0].rank}")
    print(f"训练样本数: {num_samples}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print("-" * 50)
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        # 随机打乱数据
        indices = np.random.permutation(num_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # 批次训练
        for i in range(0, num_samples, batch_size):
            # 获取当前批次
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # 前向传播
            predictions = network.forward(batch_X)
            
            # 计算损失
            loss = mean_squared_error(predictions, batch_y)
            epoch_loss += loss
            num_batches += 1
            
            # 计算梯度
            grad_output = 2 * (predictions - batch_y) / batch_size
            
            # 反向传播
            network.backward(grad_output, batch_X)
            
            # 更新参数
            network.update_parameters(learning_rate)
        
        # 记录平均损失
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # 每10轮打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}")
    
    return losses

def compare_parameter_efficiency():
    """
    比较LoRA和全参数微调的参数效率
    """
    print("\n" + "="*60)
    print("参数效率比较")
    print("="*60)
    
    # 网络配置
    input_dim = 512
    hidden_dim = 1024
    output_dim = 256
    rank = 16
    
    # 计算全参数微调的参数数量
    full_params = input_dim * hidden_dim + hidden_dim * output_dim
    
    # 计算LoRA的参数数量
    lora_params = (input_dim * rank + rank * hidden_dim) + (hidden_dim * rank + rank * output_dim)
    
    # 计算参数减少比例
    reduction_ratio = (full_params - lora_params) / full_params * 100
    
    print(f"全参数微调参数数量: {full_params:,}")
    print(f"LoRA参数数量: {lora_params:,}")
    print(f"参数减少比例: {reduction_ratio:.1f}%")
    print(f"压缩比: {full_params / lora_params:.1f}x")

def visualize_training(losses: List[float]):
    """
    可视化训练过程
    
    Args:
        losses: 损失值列表
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('LoRA训练损失曲线', fontsize=14)
    plt.xlabel('训练轮数', fontsize=12)
    plt.ylabel('均方误差', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数：运行LoRA训练仿真
    """
    print("LoRA (Low-Rank Adaptation) 训练仿真")
    print("="*50)
    
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 网络配置
    input_dim = 128
    hidden_dim = 256
    output_dim = 64
    rank = 8  # LoRA的秩
    alpha = 16.0  # 缩放因子
    
    # 创建LoRA网络
    layer_dims = [(input_dim, hidden_dim), (hidden_dim, output_dim)]
    network = LoRANetwork(layer_dims, rank, alpha)
    
    # 生成训练数据
    print("生成训练数据...")
    X_train, y_train = generate_training_data(1000, input_dim, output_dim)
    X_test, y_test = generate_training_data(200, input_dim, output_dim)
    
    # 训练网络
    print("\n开始训练...")
    start_time = time.time()
    losses = train_lora_network(network, X_train, y_train, 
                               epochs=100, learning_rate=0.001, batch_size=64)
    training_time = time.time() - start_time
    
    # 测试网络
    print("\n测试网络性能...")
    test_predictions = network.forward(X_test)
    test_loss = mean_squared_error(test_predictions, y_test)
    
    # 打印结果
    print(f"\n训练完成!")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"最终训练损失: {losses[-1]:.6f}")
    print(f"测试损失: {test_loss:.6f}")
    
    # 比较参数效率
    compare_parameter_efficiency()
    
    # 可视化训练过程
    print("\n绘制训练曲线...")
    visualize_training(losses)
    
    # 展示LoRA的核心概念
    print("\n" + "="*60)
    print("LoRA核心概念总结")
    print("="*60)
    print("1. 原始权重矩阵 W 保持不变")
    print("2. 学习低秩矩阵 A 和 B，使得 ΔW = A * B")
    print("3. 最终权重 = W + (α/r) * ΔW")
    print("4. 大幅减少可训练参数数量")
    print("5. 保持与全参数微调相近的性能")

if __name__ == "__main__":
    main()
