import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import shapiro, kstest, chisquare, norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math

# np.random.seed(42)  # для воспроизводимости результатов

# Генерация данных
n_samples = 200
x1 = np.random.uniform(-5, 5, n_samples)
x2 = np.random.uniform(-5, 5, n_samples)
x3 = np.random.uniform(-5, 5, n_samples)
e = np.random.normal(0, 1, n_samples)  # случайный шум

# Рассчитываем отклик y
y = 1 + 3 * x1 - 2 * x2 + x3 + e

# Создаем DataFrame
data = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

# Добавляем столбец с единицами для свободного члена
X = sm.add_constant(data[["x1", "x2", "x3"]])
y = data["y"]

# Строим модель
model = sm.OLS(y, X).fit()

# Выводим результаты регрессии
print(model.summary())

lr = LinearRegression()
lr.fit(data[["x1", "x2", "x3"]], data["y"])

err = y - lr.predict(data[["x1", "x2", "x3"]])

ESS = sum((err) ** 2)
RSE = math.sqrt(ESS / (n_samples - 3 - 1) * ESS)
TSS = sum((y - np.mean(y)) ** 2)
F_stat = ((TSS - ESS) / 3) / (ESS / (n_samples - 3 - 1))
R2 = 1 - ESS / TSS
print(f"R2 = {round(R2,3)}")
print(f"F_stat = {F_stat}")
print(f"TSS = {TSS}")
print(f"RSE = {RSE}")
print(f"ESS = {ESS}")

r2_score(lr.predict(data[["x1", "x2", "x3"]]), y)
r2 = 0.814

df = pd.read_csv("data.csv", low_memory=False)
df = df.drop_duplicates()

plt.rcParams["figure.figsize"] = (20, 5)

plt.scatter(df["Year"], df["No_Smoothing"], label="Data")


for degree in range(1, 7):
    model = np.poly1d(np.polyfit(df["Year"], df["No_Smoothing"], degree))

    plt.plot(df["Year"], model(df["Year"]), label=f"Degree {degree}")

plt.legend()


plt.show()
