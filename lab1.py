import numpy
from math import sqrt
import random
import matplotlib.pyplot as plt

EPS = 1e-1


def createVectors():
    x = [i for i in range(-10, 6)]
    y = [i for i in range(-5, 11)]
    return x, y


def buildVectorZ(x, y):
    return [
        (y[int(i / 2)] if i % 2 == 0 else x[int((i - 1) / 2)])
        for i in range(len(x) + len(y))
    ]


def infNorm(vector):
    return max(vector)


def firstNorm(vector):
    res = 0
    for elem in vector:
        res += abs(elem)
    return res


def secondNorm(vector):
    res = 0
    for elem in vector:
        res += elem * elem
    return sqrt(res)


def task1():
    x = numpy.linspace(-10, 10)
    plt.figure()
    plt.plot(x, numpy.sin(x), label="sin(x)")
    plt.plot(x, numpy.cos(x), label="cos(x)")
    plt.legend()


# Function to calculate weighted norm
# In: vector, weight's vector
# Return: weighted norm
def weightedNorm(vector, weight):
    for elem in weight:
        if elem <= 0:
            print("Weight can not be negative!")
            return -1
    if abs(sum(weight) - 1) >= EPS:
        print("Weight must be normalized!")
        return -1
    res = 0
    for i in range(len(vector)):
        res += abs(vector[i]) * weight[i]
    return res


# Task 2
# Function print two vectors
def task2():
    print("Task 2:")
    x, y = createVectors()
    print("x vector: ", x)
    print("y vector: ", y)


# Task 3
# Fucntion to generate new z vector and sort it
def task3():
    print("\nTask 3:")
    x, y = createVectors()
    z = buildVectorZ(x, y)
    print("z vector: ", z)
    print("Sorted z vector: ", sorted(z))


def generateWeight(size):
    ans = [random.uniform(0, 1) for _ in range(size)]
    ans = [x / sum(ans) for x in ans]
    return ans


def task4():
    print("\nTask 4:")
    x, y = createVectors()
    z = buildVectorZ(x, y)

    weight = generateWeight(len(x))
    print("Weight for norm", weight)
    print("x vector's inf norm: ", infNorm(x))
    print("x vector's first norm: ", firstNorm(x))
    print("x vector's second norm: ", secondNorm(x))
    print("x vector's weight norm: ", weightedNorm(x, weight))

    print("y vector's inf norm: ", infNorm(y))
    print("y vector's first norm: ", firstNorm(y))
    print("y vector's second norm: ", secondNorm(y))
    print("y vector's weight norm: ", weightedNorm(y, weight))

    weight = generateWeight(len(z))
    print("Weight for norm", weight)
    print("z vector's inf norm: ", infNorm(z))
    print("z vector's first norm: ", firstNorm(z))
    print("z vector's second norm: ", secondNorm(z))
    print("z vector's weight norm: ", weightedNorm(z, weight))


def task5():
    print("task 5")

    matrix = numpy.arange(1, 101).reshape(10, 10)
    print("matrix A: ", matrix)

    sum_cols = numpy.sum(matrix, axis=0)
    print("sums of cols: ", sum_cols)

    sum_rows = numpy.sum(matrix, axis=1)
    print("sums of rows: ", sum_rows)

    print("matrix A * sums of cols: ", matrix.dot(sum_cols))
    print("matrix A * sums of rows: ", matrix.dot(sum_rows))

    print("matrix B: ", matrix[5:10, 5:10])

    print()


def task6():
    print("\nTask 6:")
    num = int(input("Enter integer to calculate factorial:"))
    print("Factorial of {0} is {1}".format(num, numpy.math.factorial(num)))


def task7():
    print("\nTask 7:")
    values = []
    string = str(input("Enter five numbers:"))
    try:
        values = [float(x) for x in string.split()]
    except:
        print("Error in parsing your input")
        return
    if len(values) != 5:
        print("Incorrect array size")
        return
    print("Input array: ", values)
    print("Max value of array: ", max(values))
    print("Min value of array: ", min(values))
    print("Sum of array's values: ", sum(values))


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()
    task7()
    plt.show()
