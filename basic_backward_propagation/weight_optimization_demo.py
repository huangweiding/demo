"""
权重优化示例 (使用NumPy + SGD)
============================

本代码演示如何使用SGD优化权重参数，拟合一元二次函数。
这是一个典型的机器学习回归问题。

作者: AI Assistant
日期: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

class QuadraticModel:
    """
    一元二次模型: f(x) = ax² + bx + c
    现在 a, b, c 是需要优化的权重参数
    """
    
    def __init__(self):
        """
        初始化模型权重
        """
        # 随机初始化权重
        self.a = np.random.normal(0, 0.1)
        self.b = np.random.normal(0, 0.1)
        self.c = np.random.normal(0, 0.1)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播: f(x) = ax² + bx + c
        
        Args:
            x: 输入数组
            
        Returns:
            预测值数组
        """
        return self.a * x**2 + self.b * x + self.c
    
    def compute_gradients(self, x: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
        """
        计算权重梯度
        
        Args:
            x: 输入数据
            y_true: 真实标签
            
        Returns:
            (梯度a, 梯度b, 梯度c)
        """
        y_pred = self.forward(x)
        error = y_pred - y_true
        
        # 对a的梯度: ∂L/∂a = 2x² * error
        grad_a = 2 * np.mean(x**2 * error)
        
        # 对b的梯度: ∂L/∂b = 2x * error  
        grad_b = 2 * np.mean(x * error)
        
        # 对c的梯度: ∂L/∂c = 2 * error
        grad_c = 2 * np.mean(error)
        
        return grad_a, grad_b, grad_c
    
    def compute_loss(self, x: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算均方误差损失
        
        Args:
            x: 输入数据
            y_true: 真实标签
            
        Returns:
            损失值
        """
        y_pred = self.forward(x)
        return np.mean((y_pred - y_true)**2)

class SGD:
    """
    随机梯度下降优化器
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        初始化SGD优化器
        
        Args:
            learning_rate: 学习率
            momentum: 动量系数
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_a = 0.0
        self.velocity_b = 0.0
        self.velocity_c = 0.0
        
    def update_weights(self, grad_a: float, grad_b: float, grad_c: float) -> Tuple[float, float, float]:
        """
        更新权重
        
        Args:
            grad_a: 权重a的梯度
            grad_b: 权重b的梯度
            grad_c: 权重c的梯度
            
        Returns:
            (权重a的更新量, 权重b的更新量, 权重c的更新量)
        """
        # 更新速度（动量）
        self.velocity_a = self.momentum * self.velocity_a + grad_a
        self.velocity_b = self.momentum * self.velocity_b + grad_b
        self.velocity_c = self.momentum * self.velocity_c + grad_c
        
        # 计算更新量
        update_a = -self.learning_rate * self.velocity_a
        update_b = -self.learning_rate * self.velocity_b
        update_c = -self.learning_rate * self.velocity_c
        
        return update_a, update_b, update_c

def generate_training_data(n_samples: int = 100, noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成训练数据
    
    Args:
        n_samples: 样本数量
        noise_std: 噪声标准差
        
    Returns:
        (输入数据, 真实标签)
    """
    # 真实函数: f(x) = 2x² - 4x + 1
    true_a, true_b, true_c = 2.0, -4.0, 1.0
    
    # 生成输入数据
    x = np.linspace(-2, 4, n_samples)
    
    # 生成真实标签（带噪声）
    y_true = true_a * x**2 + true_b * x + true_c + np.random.normal(0, noise_std, n_samples)
    
    return x, y_true

def train_model(model: QuadraticModel, x: np.ndarray, y_true: np.ndarray, 
                learning_rate: float, momentum: float = 0.0, 
                max_epochs: int = 100, batch_size: int = 32, 
                use_sgd: bool = False) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    训练模型
    
    Args:
        model: 二次模型
        x: 输入数据
        y_true: 真实标签
        learning_rate: 学习率
        momentum: 动量系数
        max_epochs: 最大训练轮数
        batch_size: 批次大小
        
    Returns:
        (损失历史, 权重a历史, 权重b历史, 权重c历史)
    """
    # 初始化SGD优化器
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    
    # 记录历史
    loss_history = []
    a_history = [model.a]
    b_history = [model.b]
    c_history = [model.c]
    
    print(f"开始训练模型...")
    print(f"初始权重: a={model.a:.4f}, b={model.b:.4f}, c={model.c:.4f}")
    print(f"学习率: {learning_rate}")
    print(f"动量: {momentum}")
    print(f"训练样本数: {len(x)}")
    print()
    
    print("训练过程:")
    print("轮次\t损失\t\ta\t\tb\t\tc")
    print("-" * 60)
    
    for epoch in range(max_epochs):
        # 随机打乱数据
        indices = np.random.permutation(len(x))
        x_shuffled = x[indices]
        y_shuffled = y_true[indices]
        
        epoch_loss = 0
        num_updates = 0
        
        if use_sgd:
            # 真正的随机梯度下降：每次只用一个样本
            for i in range(len(x)):
                # 获取单个样本
                single_x = np.array([x_shuffled[i]])
                single_y = np.array([y_shuffled[i]])
                
                # 计算梯度（单个样本）
                grad_a, grad_b, grad_c = model.compute_gradients(single_x, single_y)
                
                # 更新权重
                update_a, update_b, update_c = optimizer.update_weights(grad_a, grad_b, grad_c)
                model.a += update_a
                model.b += update_b
                model.c += update_c
                
                # 计算损失
                loss = model.compute_loss(single_x, single_y)
                epoch_loss += loss
                num_updates += 1
        else:
            # 小批量梯度下降
            for i in range(0, len(x), batch_size):
                # 获取当前批次
                batch_x = x_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # 计算梯度
                grad_a, grad_b, grad_c = model.compute_gradients(batch_x, batch_y)
                
                # 更新权重
                update_a, update_b, update_c = optimizer.update_weights(grad_a, grad_b, grad_c)
                model.a += update_a
                model.b += update_b
                model.c += update_c
                
                # 计算损失
                loss = model.compute_loss(batch_x, batch_y)
                epoch_loss += loss
                num_updates += 1
        
        # 记录历史
        avg_loss = epoch_loss / num_updates
        loss_history.append(avg_loss)
        a_history.append(model.a)
        b_history.append(model.b)
        c_history.append(model.c)
        
        # 每10轮打印一次
        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1:3d}\t{avg_loss:8.4f}\t{model.a:8.4f}\t{model.b:8.4f}\t{model.c:8.4f}")
    
    print(f"\n最终权重: a={model.a:.4f}, b={model.b:.4f}, c={model.c:.4f}")
    print(f"真实权重: a=2.0000, b=-4.0000, c=1.0000")
    
    return loss_history, a_history, b_history, c_history

def compare_learning_rates():
    """
    比较不同学习率的效果
    """
    print("=" * 60)
    print("不同学习率权重优化效果比较")
    print("=" * 60)
    
    # 生成训练数据
    x, y_true = generate_training_data(n_samples=100, noise_std=0.1)
    
    learning_rates = [0.001, 0.01, 0.02]
    results = {}
    
    for lr in learning_rates:
        print(f"\n学习率: {lr}")
        
        # 创建新模型
        model = QuadraticModel()
        
        # 训练模型
        loss_history, a_history, b_history, c_history = train_model(
            model, x, y_true, lr, max_epochs=50
        )
        
        results[lr] = {
            'loss': loss_history,
            'a': a_history,
            'b': b_history,
            'c': c_history,
            'final_model': model
        }
    
    return results, x, y_true

def compare_gradient_descent_methods():
    """
    比较不同梯度下降方法的效果
    """
    print("=" * 60)
    print("不同梯度下降方法效果比较")
    print("=" * 60)
    
    # 生成训练数据
    x, y_true = generate_training_data(n_samples=100, noise_std=0.1)
    
    methods = [
        ("SGD (batch_size=1)", True, 1),
        ("Mini-batch (batch_size=8)", False, 8),
        ("Mini-batch (batch_size=32)", False, 32),
        ("Mini-batch (batch_size=64)", False, 64)
    ]
    
    results = {}
    
    for method_name, use_sgd, batch_size in methods:
        print(f"\n方法: {method_name}")
        
        # 创建新模型
        model = QuadraticModel()
        
        # 训练模型
        loss_history, a_history, b_history, c_history = train_model(
            model, x, y_true, learning_rate=0.01, max_epochs=30, 
            batch_size=batch_size, use_sgd=use_sgd
        )
        
        results[method_name] = {
            'loss': loss_history,
            'a': a_history,
            'b': b_history,
            'c': c_history,
            'final_model': model,
            'batch_size': batch_size,
            'use_sgd': use_sgd
        }
    
    return results, x, y_true

def visualize_training(results: dict, x: np.ndarray, y_true: np.ndarray):
    """
    可视化训练过程
    
    Args:
        results: 训练结果
        x: 输入数据
        y_true: 真实标签
    """
    plt.figure(figsize=(15, 10))
    
    # 子图1: 损失曲线
    plt.subplot(2, 3, 1)
    for lr, result in results.items():
        plt.plot(result['loss'], label=f'LR = {lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 子图2: 权重a的变化
    plt.subplot(2, 3, 2)
    for lr, result in results.items():
        plt.plot(result['a'], label=f'LR = {lr}')
    plt.axhline(y=2.0, color='r', linestyle='--', label='True a = 2.0')
    plt.xlabel('Epoch')
    plt.ylabel('Weight a')
    plt.title('Weight a Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 权重b的变化
    plt.subplot(2, 3, 3)
    for lr, result in results.items():
        plt.plot(result['b'], label=f'LR = {lr}')
    plt.axhline(y=-4.0, color='r', linestyle='--', label='True b = -4.0')
    plt.xlabel('Epoch')
    plt.ylabel('Weight b')
    plt.title('Weight b Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4: 权重c的变化
    plt.subplot(2, 3, 4)
    for lr, result in results.items():
        plt.plot(result['c'], label=f'LR = {lr}')
    plt.axhline(y=1.0, color='r', linestyle='--', label='True c = 1.0')
    plt.xlabel('Epoch')
    plt.ylabel('Weight c')
    plt.title('Weight c Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图5: 模型拟合效果
    plt.subplot(2, 3, 5)
    x_range = np.linspace(-2, 4, 100)
    
    # 绘制真实函数
    y_true_func = 2 * x_range**2 - 4 * x_range + 1
    plt.plot(x_range, y_true_func, 'r-', linewidth=2, label='True function')
    
    # 绘制训练数据
    plt.scatter(x, y_true, alpha=0.6, label='Training data')
    
    # 绘制不同学习率的拟合结果
    colors = ['blue', 'green', 'orange']
    for i, (lr, result) in enumerate(results.items()):
        model = result['final_model']
        y_pred = model.forward(x_range)
        plt.plot(x_range, y_pred, color=colors[i], linewidth=2, 
                label=f'LR = {lr} (a={model.a:.2f}, b={model.b:.2f}, c={model.c:.2f})')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model Fitting Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图6: 权重收敛对比
    plt.subplot(2, 3, 6)
    true_weights = [2.0, -4.0, 1.0]
    weight_names = ['a', 'b', 'c']
    
    for i, (lr, result) in enumerate(results.items()):
        final_weights = [result['a'][-1], result['b'][-1], result['c'][-1]]
        errors = [abs(w - true) for w, true in zip(final_weights, true_weights)]
        plt.bar([f'{name}\nLR={lr}' for name in weight_names], errors, 
               alpha=0.7, label=f'LR = {lr}')
    
    plt.ylabel('Absolute Error')
    plt.title('Final Weight Errors')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数：运行权重优化演示
    """
    print("权重优化示例 - 拟合一元二次函数")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 比较不同梯度下降方法
    print("1. 不同梯度下降方法比较")
    results, x, y_true = compare_gradient_descent_methods()
    
    # 2. 可视化结果
    print("\n" + "="*60)
    print("2. 可视化结果")
    print("正在生成图表...")
    visualize_training(results, x, y_true)
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("1. 权重优化是机器学习中的核心问题")
    print("2. 不同的梯度下降方法有不同的特点:")
    print("   - SGD: 更新频繁，噪声大，可能跳出局部最优")
    print("   - Mini-batch: 平衡了效率和稳定性")
    print("   - 批次大小影响收敛速度和稳定性")
    print("3. 通过梯度下降可以学习到数据的潜在模式")
    print("4. 这是深度学习模型训练的基础")

if __name__ == "__main__":
    main()
