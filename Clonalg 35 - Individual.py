import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import itertools


# Quando verdadeiro, oculta a visulizacao do jogo
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Cria a pasta com os arquivos dessa execucao
experiment_name = 'clonalg_35_individual'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Inicializacao do ambiente e da arquitetura da rede
n_hidden_neurons = 10

env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  multiplemode="no",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")


env.state_to_log() 
ini = time.time()  

n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# Configuracoes
dom_u = 1               # Dominio superior
dom_l = -1              # Dominio inferior
n_pop = 100             # Tamanho da populacao
gens = 100              # Quantidade de geracoes
mutation_factor = 2.5   # Fator de mutacao
cloning_factor = 0.26   # Fator de clonagem 0.15, 0.26
n_clones = 35           # numero de individuos selecionados para a clonagem 70, 35
n_d = 10                # Numero de novos individuos

np.random.seed(420) # Sementes utilizadas: 420, 425, 430, 435, 440

# Executa a simulacao
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    with open(experiment_name+'/stats.txt', 'a') as r:
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

# Avaliacao
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

# Normalizacao
def norm(x, pfit_pop):
    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm

# Limites
def limits(x):
    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x

# Calcula quantos clones vai ter cada individuo selecionado para clonagem
# Quanto maior o fitness, maior a quantidade de clones
def clone():
    chosen = []
    for i in range(n_clones):
        qtd_clones = max(((cloning_factor*n_pop)//(i+1)), 1)
        qtd_clones = int(round(qtd_clones))
        for j in range(qtd_clones):
            chosen.append(i)
    return chosen

# Realiza a mutacao dos clones gerados. Quanto maior o fitness, menor a taxa de clonagem
def hypermutation(pop, fit_pop):
    j = 0
    for p in pop:
        # Normalizacao
        fit_normalized = np.max([((fit_pop[j] + 20)/(100 + 20)), 0]) # Alcance: -20 ate 100
        weight = np.exp(-mutation_factor * fit_normalized) # Probabilidade de mutacao
        
        # Mutacao
        for i in range(0,p.shape[0]):
            if np.random.uniform(0 ,1) <= weight:
                p[i] = p[i] + np.random.normal(0, 1)
                p[i] = limits(p[i])
        j += 1
    return pop

# Substitui n_d individuos da populacao
def remainingSelection(pop, fit_pop):
    # Gera os novos individuos e avalia seu fitness
    pop_d = np.random.uniform(dom_l, dom_u, (n_d, n_vars))
    fit_pop_d = evaluate(pop_d)

    # Insere na pop (Nd U Pr)
    pop = np.append(pop,pop_d, axis=0)
    fit_pop = np.append(fit_pop,fit_pop_d, axis=0)

    # Seleciona os melhores
    order = np.argsort(fit_pop)[::-1]
    chosen = order[0:n_pop]            
    return pop[chosen], fit_pop[chosen]


print( '\nNova Evolucao\n')

print( '\nIniciando nova populacao\n')

# Cria aleatoriamente a populacao inicial
pop = np.random.uniform(dom_l, dom_u, (n_pop, n_vars))

# Calcula o fitness dos individuos gerados
fit_pop = evaluate(pop)

# Guarda os resultados
with open(experiment_name+'/results.txt', 'a') as f:
    for i in fit_pop:
        f.write(str(i))
        f.write(';')
    f.write('\n')

# Dados
best = np.argmax(fit_pop)
mean = np.mean(fit_pop)
std = np.std(fit_pop)
ini_g = 0
solutions = [pop, fit_pop]
env.update_solutions(solutions)

# Gera os ids com a quantidade de clones que serão gerados a cada geracao, 
# com base o ranking de fitness
cloned_ids = clone()

# Inicia a execucao das geracoes
for i in range(ini_g, gens):

    print('\nPopulacao da geracao', i, '\n')

    # Seleciona os melhores individuos para serem clonados (Pn)
    order = np.argsort(fit_pop)[::-1]
    best_pop = order[0:n_clones]
    clone_pop = pop[best_pop]
    fit_clone_pop = fit_pop[best_pop]    

    # Clona e envia os clones gerados para a mutacao (C -> C*)
    cloned_pop = hypermutation(clone_pop[cloned_ids], fit_clone_pop[cloned_ids])

    print( '\nAvaliando os clones\n')
    # Avalia os clones gerados
    fit_cloned_pop = evaluate(cloned_pop)

    # Seleciona os n melhores clones para ir para a populacao 
    order = np.argsort(fit_cloned_pop)[::-1]
    best_clones = order[0:n_clones]
    cloned_pop = cloned_pop[best_clones]
    fit_cloned_pop = fit_cloned_pop[best_clones]

    # Junta os melhores clones com a populacao original (P U C* -> Pr)
    pop = np.append(pop,cloned_pop, axis=0)
    fit_pop = np.append(fit_pop,fit_cloned_pop, axis=0)
    best = np.argmax(fit_pop)

    # Seleciona os individuos que vão para a próxima geracao
    order = np.argsort(fit_pop)[::-1]
    best_pop = order[0:n_pop]
    pop = pop[best_pop]
    fit_pop = fit_pop[best_pop]

    print( '\nNovos individuos\n')
    # Define quem vai ser substituido por novos elementos (Pr U Nd -> P)
    pop, fit_pop = remainingSelection(pop, fit_pop)

    # Dados
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

    # Guarda os resultados
    with open(experiment_name+'/results.txt', 'a') as f:
        for i in fit_pop:
            f.write(str(i))
            f.write(';')
        f.write('\n')


# Ao final, demonstra o melhor individuo encontrado e guarda seus pesos
print( '\nMelhor individuo encontrado:\n')
order = np.argsort(fit_pop)[::-1]
best_element = order[0:1]
final = evaluate(pop[best_element])
np.savetxt(experiment_name+'/best.txt',pop[best_element])