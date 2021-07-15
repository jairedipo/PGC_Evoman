import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import itertools

experiment_name = '' # Colocar o nome da pasta gerada no experimento
print(experiment_name)

# Inicia o ambiente para enfrentar todos os individuos
env = Environment(experiment_name=experiment_name,
                  enemies=[1,2,3,4,5,6,7,8],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(10),
                  enemymode="static",
                  level=2,
                  speed="normal")


# Executa a simulacao
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    with open('stats.txt', 'a') as r:
        r.write(str(f))
        r.write(';')
        r.write(str(p))
        r.write(';')
        r.write(str(e))
        r.write(';')
        r.write(str(t))
        r.write(';')
        r.write('\n')
    return f

def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


bsol = np.loadtxt(experiment_name+'/best.txt')
print( '\n RUNNING SAVED BEST SOLUTION \n')
env.update_parameter('speed','normal')

# Executa 3 vezes o melhor individuo
evaluate([bsol])
print()
evaluate([bsol])
print()
evaluate([bsol])
    
sys.exit(0)