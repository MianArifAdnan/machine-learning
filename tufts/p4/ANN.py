import numpy as np
#from sys import platform as sys_pf
#if sys_pf == 'darwin':
#    import matplotlib
#    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import sys
import copy
import ANN_math
import pdb

np.set_printoptions(threshold=np.inf)
np.random.seed(0)

#def handle_close(evt):
#    print('Closed Figure!')

class ArtificialNeuralNetwork:
    WEIGHT_INIT_LOWER = -0.1
    WEIGHT_INIT_UPPER = 0.1
    STEP_SIZE = 0.1
    BIAS = -1
    SIGMOID_THRESH = -50

    def __init__(self, train_data, test_data, width, depth, iters=3000):
        self.data = {"train": train_data, "test": test_data}
        self.num_examples = {
            "train": np.shape(self.data["train"]["features"])[0],
            "test": np.shape(self.data["test"]["features"])[0]}
        self.input_width = np.size(self.data["train"]["features"][0])
        self.hidden_width = width
        self.output_width = np.size(self.data["train"]["classes"])
        self.depth = depth
        self.weights = [None] * (depth + 1)
        self.activations = {"train": [None] * (depth + 2),
                            "test": [None] * (depth + 2)}
        self.zs = {"train": [None] * (depth + 1),
                   "test": [None] * (depth + 1)}
        self.activations["train"][0] = self.process_input("train")
        self.activations["test"][0] = self.process_input("test")
        self.iters = iters
        self.err_rates = {"train": [], "test": []}

        self.one_hot_data()
        self.init_nodes()
        self.set_weights()
        self.learn_weights()

    def process_input(self, data_set):
        input = self.data[data_set]["features"].astype(float)
        bias_inputs = self.add_bias(input)
        return bias_inputs

    def one_hot(self, labels):
        one_hot_labels = np.zeros((np.size(labels), self.output_width),
            dtype=int)
        for i, label in enumerate(labels):
            if self.data["train"]["classes"][0][0] == 1:
                one_hot_labels[i][label - 1] = 1
            else:
                one_hot_labels[i][label] = 1
        return one_hot_labels
    
    def one_hot_data(self):
        for set in self.data:
            self.data[set]["labels"] = self.one_hot(self.data[set]["labels"])
            self.data[set]["classes"] = self.one_hot(self.data[set]["classes"])

    def init_weight(self, num_rows, num_cols, weight_num):
        self.weights[weight_num] = np.random.uniform(self.WEIGHT_INIT_LOWER,
            self.WEIGHT_INIT_UPPER, (num_rows, num_cols))

    def set_weights(self):
        weight_num = 0
        if self.depth == weight_num:
            self.init_weight(self.output_width, self.input_width + 1,
                weight_num)
        else:
            self.init_weight(self.hidden_width, self.input_width + 1,
                weight_num)
            for weight_num in range(1, self.depth):
                self.init_weight(self.hidden_width, self.hidden_width + 1,
                    weight_num)
            weight_num += 1
            self.init_weight(self.output_width, self.hidden_width + 1,
                weight_num)

    def init_nodes(self):
        for set in self.data:
            for i in range(self.depth + 1):
                if i == self.depth:
                    self.zs[set][i] = np.zeros((self.num_examples[set], self.output_width), dtype=float)
                    self.activations[set][i + 1] = np.zeros((self.num_examples[set], self.output_width), dtype=float)
                else:
                    self.zs[set][i] = np.zeros((self.num_examples[set],
                    self.hidden_width), dtype=float)
                    
                    self.activations[set][i + 1] = np.zeros((self.num_examples[set],
                    self.hidden_width + 1), dtype=float)

    def add_bias(self, x):
        bias = np.full((np.shape(x)[0], 1), self.BIAS)
        return np.hstack((bias, x))
    
    def forwardprop(self, data_set, ex_slice, weights):
        for i, weight_matrix in enumerate(weights):
            activ = self.activations[data_set][i][ex_slice]
            self.zs[data_set][i][ex_slice] = np.matmul(activ,
                np.transpose(weight_matrix))
#            if i == len(self.weights) - 1:
            activation_layer = ANN_math.sigmoid(self.zs[data_set][i][ex_slice], threshold=self.SIGMOID_THRESH)
#            else:
#                activation_layer = ANN_math.sigmoid(self.zs[data_set][i][ex_slice], threshold=self.SIGMOID_THRESH) # TODO: Change back to ReLU
            if i == len(self.weights) - 1:
                self.activations[data_set][i + 1][ex_slice] = activation_layer
            else:
                self.activations[data_set][i + 1][ex_slice] = self.add_bias(
                    activation_layer)

    def backprop(self):
        new_weights = copy.deepcopy(self.weights)
        d = [None] * len(self.weights)
        D = [np.full(np.shape(w), 0) for i, w in enumerate(self.weights)]
        d[-1] = np.multiply(self.activations["train"][-1] -
            self.data["train"]["labels"], ANN_math.sigmoid_gradient(
            self.zs["train"][-1], self.SIGMOID_THRESH))
        for i in range(len(d) - 2, -1, -1):
            no_bias_w = self.weights[i + 1][:, 1:]
            weighted_d = np.matmul(d[i + 1], no_bias_w)
            d[i] = np.multiply(weighted_d, ANN_math.relu_gradient(
                self.zs["train"][i]))

        for i in range(len(D)):
            C = 1.0 / self.num_examples["train"]
            D[i] = C * np.matmul(np.transpose(d[i]),
                self.activations["train"][i])
            new_weights[i] -= self.STEP_SIZE * D[i]
        self.weights = new_weights

    def get_slice(self, num):
        return slice(num, num + 1)

    def error(self, data_set):
        y = self.data[data_set]["labels"]
        x_N = self.activations[data_set][-1]
        return (1/2) * np.sum((y - x_N) ** 2)

    def backprop_stochastic(self):
        N = len(self.weights)
        for ex_num in range(len(self.data["train"]["features"])):
            ex_slice = self.get_slice(ex_num)
            self.forwardprop("train", ex_slice, self.weights)
            y = np.expand_dims(self.data["train"]["labels"][ex_num], axis=1)
            x = self.activations["train"]
            s = self.zs["train"]
            d = [None] * N
            new_weights = copy.deepcopy(self.weights)
            for i in range(N - 1, -1, -1):
                x_i = np.expand_dims(x[i + 1][ex_num], axis=1)
                x_j = np.expand_dims(x[i][ex_num], axis=1)
                s_i = np.expand_dims(s[i][ex_num], axis=1)
                s_i_grad = ANN_math.sigmoid_gradient(s_i, self.SIGMOID_THRESH)
                if i == N - 1:
                    d[i] = np.multiply(-(y - x_i), s_i_grad)
                else:
                    reverse_weights = np.transpose(self.weights[i + 1][:, 1:])
                    weighted_d = np.matmul(reverse_weights, d[i + 1])
                    d[i] = np.multiply(weighted_d, s_i_grad)
                deriv_err = np.matmul(d[i], np.transpose(x_j[1:]))
                new_weight = new_weights[i][:, 1:] - self.STEP_SIZE * deriv_err
                new_weights[i] = self.add_bias(new_weight)
            
            err_old = self.error("train")
            self.forwardprop("train", ex_slice, new_weights)
            err_new = self.error("train")
#            if err_new < err_old:
            self.weights = copy.deepcopy(new_weights)

    def number_mistakes(self, data_set): # Input correct?
        actuals = self.data[data_set]["labels"].argmax(axis=1)
        predictions = self.activations[data_set][-1].argmax(axis=1)
        return np.sum((actuals != predictions) * 1)

    def error_rate(self, data_set):
        err_rate = self.number_mistakes(data_set) / self.num_examples[data_set]
        return err_rate

    def correct_div_incorrect_examples(self, data_set):
        num_mistakes = self.number_mistakes(data_set)
        num_correct = self.num_examples[data_set] - num_mistakes
        if num_mistakes == 0:
            return num_correct * np.inf
        else:
            return num_correct / num_mistakes

    def learn_weights(self):
        for iter in range(self.iters):
            self.backprop_stochastic()
            self.err_rates["train"].append(self.error_rate("train"))
            all_examples = slice(0, self.num_examples["test"])
            self.forwardprop("test", all_examples, self.weights)
            self.err_rates["test"].append(self.error_rate("test"))
#            if iter == 1:
#                sys.exit()

    def plot_error_v_iter(self, description, data_set):
        data_set_str = data_set[0].upper() + data_set[1:]
        fig = plt.figure()
        plt.title(description + "\n" + data_set_str +
            " Error Rate vs. Iterations")

#        fig.canvas.mpl_connect('close_event', handle_close)
        plt.xlabel("Iterations")
        plt.ylabel("Error Rate")
        plt.plot(range(1, self.iters + 1), self.err_rates[data_set])
        plt.show()
        plt.close()
    
    def layer2_activations(self):
        classes = self.data["train"]["classes"]
        print("Classes:\n", classes)
        w_trans_no_bias = np.transpose(self.weights[0][:, 1:])
        inputs = self.data["train"]["features"]
        layer2_activs = ANN_math.sigmoid(np.matmul(inputs, w_trans_no_bias),
            threshold=self.SIGMOID_THRESH)
        print("Layer 2 activations:\n", layer2_activs)

    def test_data(self):
        print("\nTest data:")
        print(self.data["train"]["labels"][:5])
        print(self.data["train"]["classes"][:5])

    def test_one_hot(self):
        print("\nTest one_hot:")
        print(self.data["train"]["labels"][:15])
        print(self.data["train"]["classes"])
    
    def test_parameters(self):
        print("\nNumber of Training Examples:")
        print(self.num_examples["train"])
    
    def test_weights(self):
        print("\nTest weights:")
        print(self.weights)
        for i in range(len(self.weights)):
            print("Size of weight [", i, "]:", np.shape(self.weights[i]))
    
    def test_activations(self):
        print("\nTest activations:")
        for i in range(len(self.activations["train"])):
            print("Activation ", i)
            print(self.activations["train"][i][:5])

    def test(self):
        print("\nNumber of examples to print: ", 5)
        self.test_one_hot()
        self.test_parameters()
        self.test_weights()
        self.test_activations()
