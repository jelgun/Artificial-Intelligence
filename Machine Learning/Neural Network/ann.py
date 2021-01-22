import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

class ANN():
    def __init__(self, hidden_units):
        self.hidden_units = hidden_units
        
    def predict(self, x):  
        if self.hidden_units == 0:
            return x*self.w + self.b
        
        hidden_net = np.dot(self.w[0], x) + self.b_hidden
        hidden_out = self.activation(hidden_net)
        output_net = np.dot(hidden_out, self.w[1]) + self.b_output
        return hidden_out, output_net
    
    def activation(self, x):
        return 1/(1 + np.exp(-x))
    
    def activation_deriv(self, x):
        return x * (1 - x)
    
    def loss(self, net, y):
        return (net - y)**2 / 2
    
    def loss_deriv(self, net, y):
        return y - net
            
    
    def fit(self, X, y, weight_range, learning_rate, num_epochs, learning_method, momentum, stopping_loss):
        if self.hidden_units == 0:
            self.b = 0
            self.w = np.random.uniform(low=weight_range[0], high=weight_range[1], size=(1,))
            prev_delta_b = 0
        else:
            self.b_hidden = np.zeros(self.hidden_units)
            self.b_output = np.random.uniform(low=weight_range[0], high=weight_range[1], size=(1,))
            self.w = np.random.uniform(low=weight_range[0], high=weight_range[1], size=(2, self.hidden_units))
            prev_delta_b_hidden = np.zeros(self.b_hidden.shape)
            prev_delta_b_output = 0
        
        prev_delta_w = np.zeros(self.w.shape)
        
        m = len(X)
        for epoch in range(num_epochs):
            avg_loss = 0
            if learning_method == "batch":
                if self.hidden_units == 0:
                    output_nets = [self.predict(x) for x in X]
                    loss_derivs = [self.loss_deriv(output_nets[i], y[i]) for i in range(m)]

                    delta_w = learning_rate * sum(loss_derivs[i] * X[i] for i in range(m))
                    delta_b = learning_rate * sum(loss_derivs)
                    self.w += (1 - momentum) * delta_w + momentum * prev_delta_w
                    self.b += (1 - momentum) * delta_b + momentum * prev_delta_b
                    prev_delta_w = delta_w
                    prev_delta_b = delta_b
                    avg_loss = 1/m*sum(self.loss(output_nets[i], y[i]) for i in range(m))
                else:
                    delta_w = np.zeros(self.w.shape)
                    delta_b_hidden = np.zeros(self.b_hidden.shape)
                    delta_b_output = 0
                    
                    for t in range(m):
                        hidden_out, output_net = self.predict(X[t])
                        delta_w[1] += learning_rate * self.loss_deriv(output_net, y[t]) * hidden_out
                        delta_w[0] += learning_rate * self.loss_deriv(output_net, y[t]) * self.w[1] * self.activation_deriv(hidden_out) * X[t]
                        delta_b_hidden += learning_rate * self.loss_deriv(output_net, y[t]) * self.w[1] * self.activation_deriv(hidden_out)
                        delta_b_output += learning_rate * self.loss_deriv(output_net, y[t])
                        avg_loss += self.loss(output_net, y[t])/m
                    
                    self.w += (1 - momentum) * delta_w + momentum * prev_delta_w
                    self.b_hidden += (1 - momentum) * delta_b_hidden + momentum * prev_delta_b_hidden
                    self.b_output += (1 - momentum) * delta_b_output + momentum * prev_delta_b_output
                    prev_delta_w = delta_w
                    prev_delta_b_hidden = delta_b_hidden
                    prev_delta_b_output = delta_b_output
                    
            elif learning_method == "stochastic":    
                if self.hidden_units == 0:
                    for t in indices:
                        output_net = self.predict(X[t])
                        loss_deriv = self.loss_deriv(output_net, y[t])

                        delta_w = learning_rate * loss_deriv * X[t]
                        delta_b = learning_rate * loss_deriv
                        self.w += (1 - momentum) * delta_w + momentum * prev_delta_w
                        self.b += (1 - momentum) * delta_b + momentum * prev_delta_b
                        prev_delta_w = delta_w
                        prev_delta_b = delta_b
                        avg_loss += self.loss(output_net, y[t])/m
                else:
                    delta_w = np.zeros(self.w.shape)
                    for _ in range(m):
                        t = np.random.randint(0, m)
                        hidden_out, output_net = self.predict(X[t])
                        delta_w[1] = learning_rate * self.loss_deriv(output_net, y[t]) * hidden_out
                        delta_w[0] = learning_rate * self.loss_deriv(output_net, y[t]) * self.w[1] * self.activation_deriv(hidden_out) * X[t]
                        delta_b_hidden = learning_rate * self.loss_deriv(output_net, y[t]) * self.w[1] * self.activation_deriv(hidden_out)
                        delta_b_output = learning_rate * self.loss_deriv(output_net, y[t])
                        avg_loss += self.loss(output_net, y[t])/m

                        self.w += (1 - momentum) * delta_w + momentum * prev_delta_w
                        self.b_hidden += (1 - momentum) * delta_b_hidden + momentum * prev_delta_b_hidden
                        self.b_output += (1 - momentum) * delta_b_output + momentum * prev_delta_b_output
                        prev_delta_w = delta_w
                        prev_delta_b_hidden = delta_b_hidden
                        prev_delta_b_output = delta_b_output
                        avg_loss += self.loss(output_net, y[t])/m
            
            if avg_loss < stopping_loss:
                break

            if epoch % 1000 == 0:
                print(epoch, avg_loss)


def read_data(filename):
    X = []
    y = []
    data = open(filename).readlines()
    for i in data:
        row = i.strip().split()
        row = [float(i) for i in row]
        
        X.append(row[0])
        y.append(row[1])
    
    X = np.array(X)
    y = np.array(y)
    return X, y


def read_normalize(train_file, test_file):
    X_train, y_train = read_data(train_file)
    X_test, y_test = read_data(test_file)
    X_mean = X_train.mean()
    X_std = X_train.std()
    y_mean = y_train.mean()
    y_std = y_train.std()
    X_train = (X_train - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std
    X_test = (X_test - X_mean) / X_std
    y_test = (y_test - y_mean) / y_std
    
    return X_train, y_train, X_test, y_test


def plot(X, y, model, label):
    X_sample = np.random.uniform(low=min(X), high=max(X), size=(10000,))
    if model.hidden_units == 0:
        y_pred = [model.predict(i) for i in X_sample]
    else:
        y_pred = [model.predict(i)[1] for i in X_sample]

    plt.scatter(X, y,  color='black')
    plt.scatter(X_sample, y_pred, color='green')
    plt.xlabel(label)
    plt.xticks([])
    plt.yticks([])


X_train, y_train, X_test, y_test = read_normalize("train1.txt", "test1.txt")

ann = ANN(2)
ann.fit(
    X = X_train, 
    y = y_train, 
    weight_range = (-6, 6), 
    learning_rate = 0.001,
    num_epochs = 300001, 
    learning_method = "batch", 
    momentum = 0,
    stopping_loss = 0.017
)

train_losses = [ann.loss(ann.predict(X_train[i])[1], y_train[i]) for i in range(len(y_train))]
test_losses = [ann.loss(ann.predict(X_test[i])[1], y_test[i]) for i in range(len(y_test))]
print("Train loss: mean:", np.mean(train_losses), "std:", np.std(train_losses))
print("Test loss: mean:", np.mean(test_losses), "std:", np.std(test_losses))


plot(X_train, y_train, ann, "ANN-{} Train".format(ann.hidden_units))
plt.show()

X_sample = np.random.uniform(low=min(X_train), high=max(X_train), size=(10000,))
fig, axs = plt.subplots(1, ann.hidden_units)
for k in range(ann.hidden_units):
    y_pred = [ann.predict(i)[0][k] for i in X_sample]
    axs[k].scatter(X_sample, y_pred)
    axs[k].set_title(k+1)
    axs[k].set_yticklabels([])
    axs[k].set_xticklabels([])
    xleft, xright = axs[k].get_xlim()
    ybottom, ytop = axs[k].get_ylim()
    axs[k].set_aspect(abs((xright-xleft)/(ybottom-ytop)))

plt.show()
