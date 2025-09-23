import numpy as np
from matplotlib import pyplot as plt

class lora_layer():
    def __init__(self, in_features, out_features, alpha, rank):
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.rank = rank


        self.A = np.random.normal(0, np.sqrt(2/in_features), (in_features, rank))
        self.B = np.zeros((rank, out_features))

        self.grad_A = np.zeros_like(self.A)
        self.grad_B = np.zeros_like(self.B)

        self.origin_W = np.random.normal(0, 0.01, (in_features, out_features))

    def forward(self, x):
        # [batch, in_features],  [in_features, out_features]
        delta_W = self.A @self.B
        y = x@(self.origin_W +(self.alpha/self.rank)*delta_W)

        return y

    def backward(self, grad_output, x):

        # self.origin_W [in_features, out_features]
        # self.A = [in_features, rank]
        # self.B = [rank, out_features]

        delta_W = self.A @ self.B
        final_W = self.origin_W + (self.alpha/self.rank)*delta_W

        grad_input = grad_output @ final_W.T

        scale = (self.alpha/self.rank)
        self.grad_A = scale * (x.T @ grad_output @ self.B.T)
        self.grad_B = scale * (self.A.T @ x.T @ grad_output)

        return grad_input
    
    def upadate_parameters(self, learning_rate):
        self.A -= learning_rate * self.grad_A
        self.B -= learning_rate * self.grad_B

        self.grad_A.fill(0)
        self.grad_B.fill(0)


class LoraNet():
    def __init__(self, in_features=128, out_features=256, alpha=0.1, rank=2, num_layers=4) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.rank = rank
        self.num_layers = num_layers
        self.layers = []

        for _ in range(num_layers):
            if _ == 0:
                self.layers.append(lora_layer(in_features, out_features, alpha, rank))
            else:
                self.layers.append(lora_layer(out_features, out_features, alpha, rank))

    def forward(self, x):
        # x [batch_size, in_features]
        
        for layer in self.layers:
            x = layer.forward(x)

        return x


    def backward(self, grad_output, x):
        input_list = [x]

        for layer in self.layers[:-1]:
            next_input = layer.forward(input_list[-1])
            input_list.append(next_input)

        grad = grad_output
        for i in range(len(self.layers)-1, -1, -1):
            current_layer = self.layers[i]
            grad = current_layer.backward(grad, input_list[i])


    def upadate_parameters(self, learning_rate):
        for layer in self.layers:
            layer.upadate_parameters(learning_rate)



def generate_training_data(num_samples, in_features, out_features):
    X = np.random.normal(0, 1, (num_samples, in_features))

    true_W = np.random.normal(0, 0.1, (in_features, out_features))

    Y = X@true_W + np.random.normal(0, 0.01, (num_samples, out_features))

    return X, Y

def mean_square_loss(y_prediction, y_true):
    return np.mean((y_prediction - y_true)**2)

def train():
    num_samples = 2000
    in_features = 128
    out_features = 256
    alpha = 16
    rank = 8
    num_layers = 2
    learning_rate = 0.0001

    # batch_size = num_samples
    batch_size = 256

    np.random.seed(42)
    X, Y = generate_training_data(num_samples, in_features, out_features)

    network = LoraNet(in_features, out_features, alpha, rank, num_layers)

    loss_list = []
    epoches = 100
    for _ in range(epoches):
        epoch_loss = 0
        num_batches = 0

        indices = np.random.permutation(num_samples)
        X = X[indices]
        Y = Y[indices]

        for i in range(0, num_samples, batch_size):

            batch_X = X[i:i+batch_size]
            batch_Y = Y[i:i+batch_size]

            prediction_y = network.forward(batch_X)

            loss = mean_square_loss(prediction_y, batch_Y)

            grad_output = 2*(prediction_y-batch_Y)/batch_size

            epoch_loss += loss
            num_batches += 1

            network.backward(grad_output, batch_X)

            network.upadate_parameters(learning_rate)
        avg_loss = epoch_loss /num_batches
        print("avg_loss %s" %avg_loss)
        loss_list.append(avg_loss)
    plt.plot(loss_list)
    plt.show()




            
if __name__ == "__main__":
    train()














