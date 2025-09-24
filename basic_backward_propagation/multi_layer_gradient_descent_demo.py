import torch
import numpy as np
class SingleLayer(torch.nn.Module):
    """
    y = ax**2 + bx+c
    """
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A  = torch.normal(0, 0.1, (in_features, out_features))
        self.B = torch.normal(0, 0.1, (in_features, out_features))
        self.C = torch.normal(0, 0.1, (1, out_features))

        self.grad_A = torch.zeros_like(self.A)
        self.grad_B = torch.zeros_like(self.B)
        self.grad_C = torch.zeros_like(self.C)

    def forward(self, x):
        # x [batch_size, in_features]
        # A [in_features, out_features]
        # B [in_features, out_features]
        # C [1, out_features]
        y = x**2 @ self.A+ x@self.B+self.C
        return y

    def backward(self, grad_output, x):

        y = x**2 @ self.A+ x@self.B+self.C

        # grad_input: 对输入x的梯度
        # dy/dx = dy/d(x^2) * d(x^2)/dx + dy/dx * dx/dx
        # dy/d(x^2) = A, dy/dx = B
        # d(x^2)/dx = 2x
        # 所以: grad_input = grad_output @ A.T * 2x + grad_output @ B.T
        grad_input = grad_output @ self.A.T * (2 * x) + grad_output @ self.B.T

        self.grad_A = (x**2).T @ grad_output

        self.grad_B = x.T @ grad_output

        self.grad_C = grad_output.sum(dim=0, keepdim=True)
        
        # 检查梯度是否包含NaN或Inf
        if torch.isnan(self.grad_A).any() or torch.isinf(self.grad_A).any():
            print("警告: grad_A 包含 NaN 或 Inf")
            self.grad_A = torch.zeros_like(self.grad_A)
        if torch.isnan(self.grad_B).any() or torch.isinf(self.grad_B).any():
            print("警告: grad_B 包含 NaN 或 Inf")
            self.grad_B = torch.zeros_like(self.grad_B)
        if torch.isnan(self.grad_C).any() or torch.isinf(self.grad_C).any():
            print("警告: grad_C 包含 NaN 或 Inf")
            self.grad_C = torch.zeros_like(self.grad_C)

        return grad_input

    def update_parameters(self, learning_rate):
        # 检查参数是否包含NaN或Inf
        if torch.isnan(self.A).any() or torch.isinf(self.A).any():
            print("警告: 参数A包含NaN或Inf，跳过更新")
            return
        if torch.isnan(self.B).any() or torch.isinf(self.B).any():
            print("警告: 参数B包含NaN或Inf，跳过更新")
            return
        if torch.isnan(self.C).any() or torch.isinf(self.C).any():
            print("警告: 参数C包含NaN或Inf，跳过更新")
            return
            
        self.A -= learning_rate * self.grad_A
        self.B -= learning_rate * self.grad_B
        self.C -= learning_rate * self.grad_C
        
        # 清零梯度，避免累积
        self.grad_A.zero_(0)
        self.grad_B.zero_(0)
        self.grad_C.zero_(0)


class MultiLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, num_layers=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = []
        for _ in range(num_layers):
            if _ == 0:
                self.layers.append(SingleLayer(self.in_features, self.out_features))
            else:
                self.layers.append(SingleLayer(self.out_features, self.out_features))


    def forward(self, x):
        input_list = [x]
        for i in range(len(self.layers)):
            current_layers = self.layers[i]
            next_input = current_layers.forward(input_list[-1])
            input_list.append(next_input)

        return input_list[-1]

    def backward(self, grad_output, x):
        input_list = [x]
        for i in range(len(self.layers)):
            current_layers = self.layers[i]
            next_input = current_layers.forward(input_list[-1])
            input_list.append(next_input)


        grad = grad_output
        for j in range(len(self.layers)-1, -1, -1):
            current_layer = self.layers[j]
            grad = current_layer.backward(grad, input_list[j])

        return grad

    def update_parameters(self, learning_rate):
        for i in range(len(self.layers)):
            self.layers[i].update_parameters(learning_rate)


def generate_training_data(num_samples, in_features, out_features, num_layers):
    X = np.random.normal(0, 0.1, (num_samples, in_features))

    # 为每一层生成真实参数（与 MultiLayer 的层维度匹配）
    A_trues, B_trues, C_trues = [], [], []
    current_in = in_features
    for i in range(num_layers):
        current_out = out_features if i == 0 else out_features
        A_true = np.random.normal(0, 0.1, (current_in, current_out))
        B_true = np.random.normal(0, 0.1, (current_in, current_out))
        C_true = np.random.normal(0, 0.1, (1, current_out))
        A_trues.append(A_true)
        B_trues.append(B_true)
        C_trues.append(C_true)
        current_in = current_out

    # 按层生成目标 Y
    H = X
    for i in range(num_layers):
        H = H**2 @ A_trues[i] + H @ B_trues[i] + C_trues[i]
    Y = H

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
        [torch.tensor(A, dtype=torch.float32) for A in A_trues],
        [torch.tensor(B, dtype=torch.float32) for B in B_trues],
        [torch.tensor(C, dtype=torch.float32) for C in C_trues],
    )


def mean_square_loss(y_prediction, y_true):
    return torch.mean((y_prediction-y_true)**2)

def compare_parameters(model_layer, A_true, B_true, C_true):
    """
    对比训练后的参数和真实参数
    """
    print("=== 参数对比 ===")
    
    # A参数对比
    A_diff = torch.mean((model_layer.A - A_true)**2)
    print(f"A参数均方误差: {A_diff.item():.6f}")
    print(f"A参数最大差异: {torch.max(torch.abs(model_layer.A - A_true)).item():.6f}")
    
    # B参数对比
    B_diff = torch.mean((model_layer.B - B_true)**2)
    print(f"B参数均方误差: {B_diff.item():.6f}")
    print(f"B参数最大差异: {torch.max(torch.abs(model_layer.B - B_true)).item():.6f}")
    
    # C参数对比
    C_diff = torch.mean((model_layer.C - C_true)**2)
    print(f"C参数均方误差: {C_diff.item():.6f}")
    print(f"C参数最大差异: {torch.max(torch.abs(model_layer.C - C_true)).item():.6f}")
    
    return A_diff, B_diff, C_diff

def compare_parameters_multilayer(model, A_trues, B_trues, C_trues):
    """
    对比多层网络每一层的参数与真实参数列表。
    """
    print("=== 多层参数对比 ===")
    num_layers = len(model.layers)
    for i in range(num_layers):
        print(f"-- 第{i}层 --")
        compare_parameters(model.layers[i], A_trues[i], B_trues[i], C_trues[i])

def train():
    in_features = 256
    out_features = 512
    num_layers = 4
    num_samples = 200
    batch_size = 128

    learning_rate = 0.0001  # 降低学习率防止梯度爆炸

    X, Y, A_trues, B_trues, C_trues = generate_training_data(num_samples, in_features, out_features, num_layers)
    model = MultiLayer(in_features, out_features, num_layers)

    epoch = 10
    for _ in range(epoch):
        epoch_loss = []
        num_batches = 0
        indices = np.random.permutation(num_samples)
        X = X[indices]
        Y = Y[indices]

        for i in range(0, num_samples, batch_size):
            batch_x = X[i:i+batch_size]
            batch_y = Y[i:i+batch_size]
            
            # 确保批次大小一致
            if batch_x.shape[0] != batch_size:
                continue

            y_prediction = model.forward(batch_x)

            loss = mean_square_loss(y_prediction, batch_y)
            epoch_loss.append(loss)

            # 正确的梯度计算：对MSE损失求导
            grad_output = 2*(y_prediction - batch_y) / batch_size
            
            # 梯度裁剪，防止梯度爆炸
            grad_norm = torch.norm(grad_output)
            if grad_norm > 1.0:
                grad_output = grad_output / grad_norm

            model.backward(grad_output, batch_x)

            model.update_parameters(learning_rate)
        
        print("avg_loss %s" %np.mean(epoch_loss))
    # 训练结束后对比每一层参数（可选）
    compare_parameters_multilayer(model, A_trues, B_trues, C_trues)

if __name__ == "__main__":
    train()








