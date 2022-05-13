from PIL import Image
from math import exp
from random import randint
from statistics import mean
from tensorflow.keras.datasets import mnist

'''переменные: размеры каринки, нейронов на каждом из слоёв'''
width = 28
height = 28
neuronNum = 784
neuronL1 = 1024
neuronL2 = 256
neuronLast = 10


'''функции: сигмоида, производная сигмоиды, среднеквадратичная ошибка'''


def sigmoid(num):
    return 1 / (1 + exp(-num))


def d_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(arr_pred, arr_true):
    new_arr = []
    for _i in range(len(arr_pred)):
        new_arr.append((arr_true[_i] - arr_pred[_i])**2)
    return mean(new_arr)


def fileread(filename):
    f = open(filename, "r")
    arr = [x.split() for x in f]
    for _i in range(len(arr)):
        for _j in range(len(arr[_i])):
            arr[_i][_j] = float(arr[_i][_j])
    return arr


'''функция для обучения'''


'''класс слоя'''


class Layer:
    def __init__(self, nn, nl, inp, name, w_s=None):
        self.n = nl
        self.n2 = nn
        self.inpt = inp
        self.name = name
        if w_s is None:
            '''инициализируем рандомные веса'''
            self.w = []
            for _i in range(nl):
                self.w.append([0.0] * nn)
            self.w2 = [0.0] * nn
            for _i in range(nl):
                for _j in range(nn):
                    self.w2[_j] = float(randint(-100, 100) / 100)
                self.w[_i] = self.w2[::]
        else:
            self.w = w_s
        '''получаем массив выходных значений текущего слоя'''
        self.s = [0.0] * nl
        self.out = [0.0] * nl
        for _i in range(nl):
            for _j in range(nn):
                self.s[_i] += (inp[_j] * self.w[_i][_j])
            self.out[_i] = sigmoid(self.s[_i])

    '''метод для вывода весов в отдельный файл'''
    def output(self):
        fin = open(self.name, "w")
        for _i in range(self.n):
            for _j in range(self.n2):
                fin.write(str(self.w[_i][_j]) + " ")
            if _i != self.n-1:
                fin.write("\n")
        fin.close()

    def train(self, actual, expected):
        if actual.index(max(actual)) != expected.index(max(expected)):
            epoch = 0
            '''while epoch < 100:'''
            while self.out.index(max(self.out)) != expected.index(max(expected)):
                for _i in range(self.n):
                    if _i != expected.index(max(expected)):
                        for _j in range(self.n2):
                            self.w[_i][_j] -= 0.01
                            if self.w[_i][_j] < -1:
                                self.w[_i][_j] = float(randint(-100, 100) / 100)
                    else:
                        for _j in range(self.n2):
                            self.w[_i][_j] += 0.01
                            if self.w[_i][_j] > 1:
                                self.w[_i][_j] = float(randint(-100, 100) / 100)
                print(epoch, ": ")
                self.s = [0.0] * self.n
                self.out = [0.0] * self.n
                for _k in range(self.n):
                    for _c in range(self.n2):
                        self.s[_k] += (self.inpt[_c] * self.w[_k][_c])
                    self.out[_k] = sigmoid(self.s[_k])
                print(self.out)
                epoch += 1



'''получаем данные с картинки, тоесть создаём массив входного слоя'''
learn = input("Do you want to learn?[y/n]: ")

if learn == "n":
    image_name = str(input("Enter your image's name: "))
    img = Image.open(image_name)
    entrance = [0] * width * height
    count = 0
    for i in range(height):
        for j in range(width):
            if img.getpixel((i, j)) == (0, 0, 0):
                entrance[count] = 1
            else:
                entrance[count] = 0
            count += 1

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''инициализируем первый скрытый слой(2 слой)'''

    first_layer = Layer(neuronNum, neuronL1, entrance, "w1.txt", fileread("w1.txt"))

    print(first_layer.out)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''инициализируем второй скрытый слой(3 слой)'''

    second_layer = Layer(neuronL1, neuronL2, first_layer.out, "w2.txt", fileread("w2.txt"))

    print(second_layer.out)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    '''инициализируем выходной слой(4 слой)'''

    last_layer = Layer(neuronL2, neuronLast, second_layer.out, "w3.txt", fileread("w3.txt"))

    print(last_layer.out)
    print("Number: ", last_layer.out.index(max(last_layer.out)))
elif learn == "y":
    '''images = [
        "0_1.bmp",
        "1_1.bmp",
        "2_1.bmp",
        "3_1.bmp",
        "4_1.bmp",
        "5_1.bmp",
        "6_1.bmp",
        "7_1.bmp",
        "8_1.bmp",
        "9_1.bmp"
    ]'''

    '''expect = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]'''
    expect = []
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    for i in range(len(x_train)):
        expect.append([0]*10)
        expect[i][y_train[i]] = 1

    for i in range(len(x_train)):
        entrance_t = [0] * width * height
        count_t = 0
        for k in range(height):
            for j in range(width):
                if x_train[i][k][j] == 0:
                    entrance_t[count_t] = int(0)
                else:
                    entrance_t[count_t] = int(1)
                count_t += 1
        first_layer = Layer(neuronNum, neuronL1, entrance_t, "w1.txt")
        second_layer = Layer(neuronL1, neuronL2, first_layer.out, "w2.txt")
        last_layer = Layer(neuronL2, neuronLast, second_layer.out, "w3.txt")
        last_layer.train(last_layer.out, expect[i])
        print("Picture: ", i)
    first_layer.output()
    second_layer.output()
    last_layer.output()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'''выводим веса в файлы для последующей работы с ними во время обучения'''

'''first_layer.output()
second_layer.output()
last_layer.output()'''
'''

expect = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

#last_layer.train(last_layer.out, expect)
print(last_layer.out)
print("Number: ", last_layer.out.index(max(last_layer.out)))

first_layer.output()
second_layer.output()
last_layer.output()
'''


