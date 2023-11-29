import math
import numpy as np
import pandas as pd
import scipy as sp
from dateutil import parser, rrule
from datetime import datetime, time, date
import scipy.linalg
import csv
from matplotlib import pyplot as plt
import cmath


# task 1
N = 500
h = 0.05


def model_series():
    return np.array(
        [
            sum(
                [
                    (k * np.exp(-h * i / k) * np.cos(4 * np.pi * k * h * i + np.pi / k))
                    for k in range(1, 4)
                ]
            )
            for i in range(1, N + 1)
        ]
    )


def prony(x: np.array, T: float):
    if len(x) % 2 == 1:
        x = x[: len(x) - 1]

    p = len(x) // 2

    shift_x = [0] + list(x)
    a = scipy.linalg.solve([shift_x[p + i : i : -1] for i in range(p)], -x[p::])

    z = np.roots([*a[::-1], 1])

    h = scipy.linalg.solve([z**n for n in range(1, p + 1)], x[:p])

    f = 1 / (2 * np.pi * T) * np.arctan(np.imag(z) / np.real(z))
    alfa = 1 / T * np.log(np.abs(z))
    A = np.abs(h)
    fi = np.arctan(np.imag(h) / np.real(h))

    return f, alfa, A, fi


def ema(data, a):
    y = []
    y.append((data[0] + data[1]) / 2)
    for i in range(1, len(data)):
        y.append(a * data[i] + (1 - a) * y[i - 1])
    return y


def sma(data, m):
    sma = [0] * len(data)
    sma[0] = data[0]
    for i in range(1, len(data) - 1):
        w = m
        while (i - w < 0) or (i + w > len(data) - 1):
            w -= 1
        sma[i] = sum(data[i - w : i + w]) / (2 * w + 1)
    sma[-1] = data[-1]
    return sma


def rotation_points(series: np.array):
    res = []
    for i in range(1, len(series) - 2):
        if (series[i] > series[i - 1] and series[i] > series[i + 1]) or (
            series[i] < series[i - 1] and series[i] < series[i + 1]
        ):
            res.append(series[i])
    return res


def kendall(series: np.array, trend: np.array):
    tail = series - trend
    r_p = rotation_points(tail)
    p_e = (2.0 / 3.0) * (len(series) - 2)
    p_d = (16 * len(series) - 29) / 90.0
    p_c = len(r_p)

    print("rotation points count: ", p_c)
    print("Kendall coefficient: ", (4 * p_c) / (N * (N - 1)) - 1)

    if p_c < p_e + p_d and p_c > p_e - p_d:
        print("random\n")
    elif p_c > p_e + p_d:
        print("rapidly oscillating\n")
    elif p_c < p_e - p_d:
        print("positively correlated\n")


if __name__ == "__main__":
    # task 1
    series = model_series()
    plt.figure()
    plt.title("Sample")
    plt.plot(series)

    plt.grid()
    plt.xlabel("k")
    plt.ylabel("$x_k$")

    n = 128
    # Time vector
    t = np.linspace(0, 1, n, endpoint=True)

    # Amplitudes and freqs
    f1, f2, f3 = 2, 7, 12
    A1, A2, A3 = 5, 1, 3

    # Signal
    x = (
        A1 * np.cos(2 * np.pi * f1 * t)
        + A2 * np.cos(2 * np.pi * f2 * t)
        + A3 * np.cos(2 * np.pi * f3 * t)
    )

    f, alfa, A, fi = prony(x, 0.1)
    plt.figure()
    plt.stem(2 * A)
    plt.plot()
    plt.grid()

    plt.show()

    # task 2
    data_raw = pd.read_csv("~/polykek/big-data/lab4/LONDON.csv")
    # Give the variables some friendlier names and convert types as necessary.
    data_raw["mean_temp"] = data_raw["mean_temp"].astype(float)

    data_raw["date"] = [datetime.strptime(str(d), "%Y%m%d") for d in data_raw["date"]]

    # Extract out only the data we need.
    data = data_raw.loc[:, ["date", "mean_temp"]]
    print(data)

    sma_trend = sma(np.array(data["mean_temp"]), 55)
    ema005_trend = ema(np.array(data["mean_temp"]), 0.05)
    ema01_trend = ema(np.array(data["mean_temp"]), 0.1)

    plt.plot(data["mean_temp"], label="temp")
    plt.plot(sma_trend, label="sma, 55")
    plt.plot(ema005_trend, label="ema 0.05")
    plt.plot(ema01_trend, label="ema 0.1")

    plt.xlabel("Date")
    plt.ylabel("Temp")
    plt.title("Mean temp")
    plt.grid()
    plt.legend()
    plt.show()

    f = np.fft.fft(series)
    f = np.abs(f[: len(series) // 2])
    freqs = np.linspace(0, 1 / (2.0), len(series) // 2)
    freq = freqs[np.argmax(f)]

    FFT_orig = np.fft.fft(np.array(data["mean_temp"]))
    FFT_orig = abs(np.fft.fft(np.array(data["mean_temp"])))

    ordi = np.linspace(0, 0.5, len(FFT_orig) // 2)

    print(f"Main freq = {ordi[np.argmax(FFT_orig[1:len(FFT_orig)//2])+1]}")
    plt.figure()
    plt.plot(ordi[1:], FFT_orig[1 : len(FFT_orig) // 2] / len(FFT_orig), label="FFT(x)")
    plt.grid()
    plt.show()

    print("kendall sma 55")
    kendall(np.array(data["mean_temp"]), np.array(sma_trend))
    print("kendall sliding exp 0.05")
    kendall(np.array(data["mean_temp"]), np.array(ema005_trend))
    print("kendall sliding exp 0.1")
    kendall(np.array(data["mean_temp"]), np.array(ema01_trend))
