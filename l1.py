import sys, argparse, traceback, enum
import numpy as np
from abc import ABC, abstractmethod

# Генерируем синтетические последовательности данных для обучения
#data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                 [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
#target = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

#for debug purposes
def to_file(M):
    try:
        with open('out.txt', mode='wt', encoding='utf8') as out:
            for row in M:
                out.write(', '.join(list(map(str, row))) + '\n')

    except FileNotFoundError:
        traceback.print_exc()
        sys.exit(1)

def get_dimension_sizes(raw, dimensions):
    sizes = [None] * dimensions
    for i in range(dimensions):
        sizes[i] = int.from_bytes(raw[i*4:i*4+4], byteorder='big', signed=False)
    return sizes

def bytes_to_arr(raw, data_size, is_signed):
    ret_len = int(len(raw) / data_size)
    ret = [None] *  ret_len
    for i in range(ret_len):
        ret[i] = int.from_bytes(raw[i*data_size:(i+1)*data_size], byteorder='big', signed=is_signed)
    return ret

def ReadIDX(filename, imgs_to_read):
    with open(filename, mode='rb') as f:
        magic = f.read(4)
        #Check for valid magic number
        if magic[:2] != bytes([0x0, 0x0]):
            raise RuntimeError('Invalid magic number for ' + filename)
        magic = int.from_bytes(magic, byteorder='big', signed=False)
        #Get data type
        data_type = (magic >> 8) & 0xFF
        #Get number of dimensions
        dimensions = magic & 0xFF
        #Get sizes in each dimension in lists
        sizes = get_dimension_sizes(f.read(dimensions * 4), dimensions)
        num_of_imgs = min(imgs_to_read, sizes[0])
        if data_type == DataSize.BYTE.value or data_type == DataSize.CHAR.value:
            data_size = 1
        elif data_type == DataSize.SHORT.value:
            data_size = 2
        elif data_type == DataSize.INT.value or data_type == DataSize.FLOAT.value:
            data_size = 4
        elif data_type == DataSize.DOUBLE.value:
            data_size = 8
        #Read data
        sizes[0] = num_of_imgs
        size_of_elem = np.prod(sizes)
        data = f.read(data_size * size_of_elem)
    return data, sizes, data_size, data_type

def MSE(y, y_hat):
    return (y - y_hat).sum() / y.size

def RMSE(y, y_hat):
    return (y - y_hat).sum() / y.size


#Probable troublemaker
def softmax(Z):
    Z_exp = np.exp(Z - np.max(Z))
    return  Z_exp / Z_exp.sum()

def softmax_unsafe(Z):
    Z_exp = np.exp(Z)
    return  Z_exp / Z_exp.sum()

def reLU(Z):
    return np.maximum(0, Z)

def deriv_reLU(Z):
    return Z > 0
    
def deriv_tanh(Z):
    cosh_Z = np.cosh(Z)
    return 1 / (cosh_Z * cosh_Z)

def Rescale(X, factor, offset):
    operator = lambda x: x * factor + offset
    return operator(X)

def Normalize(X, mean, std):
    m, n = X.shape
    for j in range(n):
        col_mean, col_std = mean[j], std[j]
        for i in range(m):
            X[i][j] = (X[i][j] - col_mean) / col_std
    return X

class DataSize(enum.Enum):
    BYTE = 0x08
    CHAR = 0x09
    SHORT = 0x0B
    INT = 0x0C
    FLOAT = 0x0D
    DOUBLE = 0x0E

# One-hot Y must be of dimension
# num_of_valid_outputs x num_of_input_samples,
# e.g. for MNIST it's 10 x number of training samples
# (N of rows of the output matrix) 
def one_hot(Y, num_of_rows):
    oh_Y = np.zeros([Y.size, num_of_rows])
    oh_Y[np.arange(Y.size), Y] = 1
    return oh_Y.T
    
class Layer():
    def __init__(self, num_of_inputs, num_of_neurons, af, elems_dtype=np.float64):
        if not callable(af):
            raise ValueError("af is not a callable object")
        rng = np.random.default_rng()
        self.W = rng.random([num_of_neurons, num_of_inputs], dtype=elems_dtype)
        self.b = rng.random([num_of_neurons, 1], dtype=elems_dtype)
        self.Z = []
        self.A = []
        self.AF = af

    def activate(self, X):
        self.Z = self.W.dot(X) + self.b
        #Activation function is responsible for activation algorithm
        self.A = self.AF(self.Z)

    def update(self, dW, db, alpha):
        self.W = self.W - alpha * dW
        self.b = self.b - alpha * db

class Model(ABC):
    def __init__(self, alpha):
        self.alpha = alpha
        
    @abstractmethod
    def construct(self):
        raise NotImplementedError()
    
    @abstractmethod
    def forward_prop(self, input):
        raise NotImplementedError()
    
    @abstractmethod
    def backward_prop(self, output, true_output):
        raise NotImplementedError()
    
    @abstractmethod
    def loss(self, output, true_output):
        raise NotImplementedError()
    
    @abstractmethod
    def infer(self, input):
        raise NotImplementedError()
        
class MyModel(Model):
    def __init__(self, alpha, layer):
        super().__init__(alpha)
        self.layer = layer

    def construct(self):
        raise NotImplementedError()
    
    def forward_prop(self, input):
        for layer in self.layers:
            layer.activate(input)
            input = layer.A
    
    def backward_prop(self, X, true_output):
        #output - A2 (output of the output layer, stored in the model)
        #true_output - Y (array of labels in one-hot)
        A1 = self.layers[0].A
        b1 = self.layers[0].b
        W1 = self.layers[0].W
        Z1 = self.layers[0].Z
        A2 = self.layers[1].A
        b2 = self.layers[1].b
        W2 = self.layers[1].W
        Z2 = self.layers[1].Z
        oh_Y = one_hot(true_output, A2.shape[0])
        k = 1 / true_output.size # N of training samples
        dZ2 = A2 - oh_Y
        dW2 = k * dZ2.dot(A1.T)
        db2 = k * np.sum(dZ2)
        dZ1 = k * W2.T.dot(dZ2) * deriv_tanh(Z1)
        dW1 = k * dZ1.dot(X.T)
        db1 = k * np.sum(dZ1)
        lW = [dW1, dW2]
        lb = [db1, db2]
        for i in range(len(self.layers)):
            self.layers[i].update(lW[i], lb[i], self.alpha)

    def loss(self, output, true_output):
        raise NotImplementedError()
    
    def infer(self, input):
        raise NotImplementedError()
    
    def save_to_file(self, filename):
        try:
            with open(filename, mode='wt', encoding='utf8') as out:
                out.write(str(self.alpha) + '\n')
                for layer in self.layers:
                    out.write(";".join(str(x) for x in list(layer.W.shape)) + '\n')
                    out.write(";".join(str(x) for x in layer.W.flat) + '\n')
                    out.write(";".join(str(x) for x in list(layer.b.shape)) + '\n')
                    out.write(";".join(str(x) for x in layer.b.flat) + '\n')

        except FileNotFoundError:
            traceback.print_exc()
            sys.exit(1)
    
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(preds, Y):
    print(preds, Y)
    return np.sum(preds == Y) / Y.size

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--alpha', action='store', default=0.003, help="альфа-аргумент (скорость обучения; по умолчанию 0.003)")
parser.add_argument('-n', '--number', action='store', default=10000, help="число изображений для обработки (по умолчанию 10000)")
parser.add_argument('-e', '--epochs', action='store', default=100, help="количество эпох (по умолчанию 100)")
args = parser.parse_args()
number = int(args.number)
try:
        X_train, X_sizes, X_data_size, X_data_type = ReadIDX('train-images.idx3-ubyte', number)
        Y_train, Y_sizes, Y_data_size, Y_data_type  = ReadIDX('train-labels.idx1-ubyte', number)
        X_dev, X_dev_sizes, _, _ = ReadIDX('t10k-images-idx3-ubyte', number)
        Y_dev, _, _, _ = ReadIDX('t10k-labels-idx1-ubyte', number)
except FileNotFoundError:
    print("Входной файл не найден")
    traceback.print_exc()
    sys.exit(1)
except RuntimeError:
    print("Ошибка")
    traceback.print_exc()
    sys.exit(1)

size_of_row = X_sizes[1] * X_sizes[2]
X_train = np.array(bytes_to_arr(X_train, X_data_size, X_data_type != DataSize.BYTE.value), dtype=np.float64, copy=False, order='C').reshape([X_sizes[0], size_of_row])
Y_train = np.array(bytes_to_arr(Y_train, Y_data_size, Y_data_type != DataSize.BYTE.value), dtype=int, copy=False, order='C')
X_dev = np.array(bytes_to_arr(X_dev, X_data_size, X_data_type != DataSize.BYTE.value), dtype=np.float64, copy=False, order='C').reshape([X_dev_sizes[0], size_of_row])
Y_dev = np.array(bytes_to_arr(Y_dev, Y_data_size, Y_data_type != DataSize.BYTE.value), dtype=int, copy=False, order='C')

m, n = X_train.shape
#np.random.shuffle(X_train)

X_train = X_train.T
X_dev = X_dev.T

Rescale(X_train, 1.0 / 255.0, 0)
Normalize(X_train, mean=X_train.mean(axis=0), std=X_train.std(axis=0))

#to_file(X_train)

model = MyModel(np.float64(args.alpha), [Layer(784, 10, np.tanh), Layer(10, 10, softmax)])
iters = int(args.epochs)
for i in range(iters):
    model.forward_prop(X_train)
    model.backward_prop(X_train, Y_train)
    if(i % 10 == 0):
        print("Iteration ", i)
        print("Accuracy ", get_accuracy(get_predictions(model.layers[1].A), Y_train))
to_file(model.layers[1].A)
model.save_to_file('model.csv')
