##########
#build y = a*x**2 + b*x +c
# with gradient descent and single layer
#
import numpy as np

class gradient_descent():
    def __init__(self, learning_rate, momentum=0.0):
        self.velocity_A = 0.0
        self.velocity_B = 0.0
        self.velocity_C = 0.0
        self.momentum = momentum
        self.learning_rate = learning_rate

    def update_parameters(self, grad_A, grad_B, grad_C):

        self.velocity_A = self.velocity_A * self.momentum + grad_A
        self.velocity_B = self.velocity_B * self.momentum + grad_B
        self.velocity_C = self.velocity_C * self.momentum + grad_C

        grad_A = -self.learning_rate * grad_A
        grad_B = -self.learning_rate * grad_B
        grad_C = -self.learning_rate * grad_C

        return grad_A, grad_B, grad_C



class SimpleLayer():
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C

    def forward(self, x):
        return self.A*x**2+self.B*x+self.C

    def backward(self, y_true, y_pred, x):
        gradA = np.mean(2*(y_pred-y_true)*x**2)
        gradB = np.mean(2*(y_pred-y_true)*x)
        gradC = np.mean(2*(y_pred-y_true))
        
        return gradA, gradB, gradC

def mean_square_loss(y_true, y_pred):
    loss = np.sqrt((y_pred-y_true)**2)
    return loss

def generate_training_data(sample_num, true_A, true_B, true_C):
    x = np.linspace(0, 10, sample_num)
    y = true_A*x**2+true_B*x+true_C
    return x, y

def train():
    sample_num = 500000
    A, B, C = 1, 2, 0.5
    epoch_num = 500
    batch_size = 1000
    learning_rate = 0.0001
    X, Y = generate_training_data(sample_num, A, B, C)
    randomA = np.random.normal()
    randomB = np.random.normal()
    randomC = np.random.normal()

    model = SimpleLayer(randomA, randomB, randomC)
    GD = gradient_descent(learning_rate)


    for _ in range(epoch_num):
        for i in range(0, sample_num, batch_size):
            if i+batch_size >= sample_num:
                break
            batch_x = X[i:i+batch_size ]
            batch_y = Y[i:i+batch_size]

            y_pred = model.forward(batch_x)
            # loss = mean_square_loss(batch_y, y_pred)

            gradA, gradB, gradC = model.backward(batch_y, y_pred, batch_x)
            update_a, update_b, update_c = GD.update_parameters(gradA, gradB, gradC)
            print(update_a, update_b, update_c)

            model.A += update_a
            model.B += update_b
            model.C += update_c

    print(model.A, model.B, model.C)

if __name__ == "__main__":
    train()
