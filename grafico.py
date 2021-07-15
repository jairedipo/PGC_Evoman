import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
f = np.arange(start=0, stop=1.1, step=0.1)
plt.plot(f, np.exp(-2.5*f))
plt.grid()
plt.ylabel("Probabilidade de mutação")
plt.xlabel("Fitness normalizado")

plt.savefig("mutacao_probabilidade.png")