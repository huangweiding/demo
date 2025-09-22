import numpy as np
class SGD():
    def __init__(self, learning_rate=0.01, momentum=0.0) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.a_velocity = 0.0
        self.b_velocity = 0.0
        self.c_velocity = 0.0


    def compute_gradient(self, a_gradient, b_gradient, c_gradient):

        self.a_velocity = self.a_velocity* self.momentum + a_gradient
        self.b_velocity = self.b_velocity* self.momentum + b_gradient
        self.c_velocity = self.c_velocity* self.momentum + c_gradient

        update_a = -self.learning_rate * self.a_velocity
        update_b = -self.learning_rate * self.b_velocity
        update_c = -self.learning_rate * self.c_velocity

        return update_a, update_b, update_c



class Model():
    def __init__(self) -> None:
        # assume function a*x**2+b*x+c
        # use MSE (y_pred-y_true)**2
        self.a = np.random.normal()
        self.b = np.random.normal()
        self.c = np.random.normal()
        print("initial a, b, c")
        print(self.a, self.b, self.c)

    def forward(self, x):
        a = self.a*x**2+self.b*x+self.c
        print("forward")
        print(a)
        return self.a*x**2+self.b*x+self.c

    def backward(self, y_pred, y_true, batch_x):
        error =  y_pred - y_true
        print("error")
        print(error)

        a_gradient = 2*(np.mean(error*batch_x**2))

        b_gradient = 2*(np.mean(error*batch_x))

        c_gradient = 2*(np.mean(error))

        return a_gradient, b_gradient, c_gradient

def generate_data(a=4, b=1.7, c=1):
    y_list = []
    x_list = []
    x_list = np.linspace(0, 10, num=100)
    for x in x_list:
        y_list.append(a*x**2+b*x+c)
    return x_list, y_list



def main():

    x_list, y_list = generate_data()

    learning_rate = 0.0001
    sgd = SGD(learning_rate=learning_rate)

    model = Model()
    batch_size = 4

    for epoch in range(5000):
        for i in range(0, len(x_list), batch_size):
            print("iteration")
            print(i)
            # batch_x = np.array(x_list[i:i+batch_size])
            # batch_y = np.array(y_list[i:i+batch_size])
            batch_x = np.array(x_list[i*batch_size:(i*batch_size)+batch_size])
            batch_y = np.array(y_list[i*batch_size:(i*batch_size)+batch_size])
            if i*batch_size + batch_size > len(x_list):
                break

            y_pred = model.forward(batch_x)
            print("y_pred")
            print(y_pred)

            a_gradient, b_gradient, c_gradient = model.backward(y_pred, batch_y, batch_x)

            update_a, update_b, update_c = sgd.compute_gradient(a_gradient, b_gradient, c_gradient)

            model.a += update_a
            model.b += update_b
            model.c += update_c
            print(model.a, model.b, model.c)

    print("final result")
    print("a, b, c")
    print(model.a, model.b, model.c)




if __name__ == "__main__":
    main()


