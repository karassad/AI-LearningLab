import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with open("../data/validate.tsv") as f:
    val = pd.read_csv(f, sep="\t")
with open("../data/users.tsv") as f:
    user_d = pd.read_csv(f, sep="\t")
with open("../data/history.tsv") as f:
    history = pd.read_csv(f, sep="\t")
with (open("../data/output1.csv") as f):
    output1 = pd.read_csv(f, sep=",")

output1 = output1[output1['avg_V'] > 0]
output1 = output1[output1['avg_A'] > 0]

#логарифмирование
output1['process'] = output1['process'].mask(output1['process'] < 1, 1)
output1['avg_V'] = output1['avg_V'].mask(output1['avg_V'] < 1, 1)

output1['avg_A_log'] = np.log(output1['avg_A'])
output1['avg_V_log'] = np.log(output1['avg_V'])

correlation = output1['avg_A_log'].corr(output1['avg_V_log'])
correlation1 = output1['avg_A'].corr(output1['avg_V'])
print(f"Корреляция между log() и process(avg_V): {correlation}")
print(f"Корреляция между avg_V и process: {correlation1}")


plt.figure(figsize=(10, 6)) # Установка размера фигуры
plt.scatter(output1["avg_V"], output1["avg_A"]) # Построение графика рассеяния
plt.title('График рассеяния')
plt.xlabel('avg_V')
plt.ylabel('avg_A')
plt.grid(True)
plt.show()
