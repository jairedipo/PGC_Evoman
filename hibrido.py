import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
from math import fabs,sqrt
import glob, os

# Quando verdadeiro, oculta a visulizacao do jogo
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Cria a pasta com os arquivos dessa execucao
experiment_name = 'hibrido'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Inicializacao do ambiente e da arquitetura da rede
n_hidden_neurons = 10

env = Environment(experiment_name=experiment_name,
                  enemies=[1, 4, 6, 7],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")


env.state_to_log() 
ini = time.time()

n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# Parametros do algoritmo genetico
run_mode = 'train'      # 'train' ou 'test'
dom_u = 1               # Dominio superior
dom_l = -1              # Dominio inferior
n_pop = 100             # Tamanho da populacao
gens = 100              # Quantidade de geracoes
mutation = 0.2          # Probabilidade de mutacao
mutation_factor = 2.5   # Fator de mutacao
cloning_factor = 0.26   # Fator de clonagem
n_clones = 35           # numero de individuos selecionados para a clonagem
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


# Torneio
def tournament(pop):
    c1 =  np.random.randint(0,pop.shape[0], 1)
    c2 =  np.random.randint(0,pop.shape[0], 1)

    if fit_pop[c1] > fit_pop[c2]:
        return pop[c1][0]
    else:
        return pop[c2][0]


# Limites
def limits(x):

    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x


# Crossover
def crossover(pop):

    total_offspring = np.zeros((0,n_vars))


    for p in range(0,pop.shape[0], 2):
        p1 = tournament(pop)
        p2 = tournament(pop)

        n_offspring =   np.random.randint(1,3+1, 1)[0]
        offspring =  np.zeros( (n_offspring, n_vars) )

        for f in range(0,n_offspring):
            # Crossover
            cross_prop = np.random.uniform(0,1)
            offspring[f] = p1*cross_prop+p2*(1-cross_prop)

            # Mutacao
            for i in range(0,len(offspring[f])):
                if np.random.uniform(0 ,1)<=mutation:
                    offspring[f][i] =   offspring[f][i]+np.random.normal(0, 1)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring

# Mata os piores genomas, e os substitui com novas solucoes aleatorias
def doomsday(pop,fit_pop):

    worst = int(npop/4)  # um quarto da populacao
    order = np.argsort(fit_pop)
    orderasc = order[0:worst]

    for o in orderasc:
        for j in range(0,n_vars):
            pro = np.random.uniform(0,1)
            if np.random.uniform(0,1)  <= pro:
                pop[o][j] = np.random.uniform(dom_l, dom_u) # dna aleatorio, distribuicao uniforme.
            else:
                pop[o][j] = pop[order[-1:]][0][j] # dna do melhor individuo

        fit_pop[o]=evaluate([pop[o]])

    return pop,fit_pop


# Carrega o arquivo com a melhor solucao para teste
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# Inicializa a populacao carregando uma solucao anterior ou gerando uma nova
if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # Encontra o numero da ultima geracao
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()




# Salva o resultado da primeira geracao
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()


# Evolucao
last_sol = fit_pop[best]
notimproved = 0

for i in range(gens//2):

    offspring = crossover(pop)  # Crossover
    fit_offspring = evaluate(offspring)   # Avaliacao
    pop = np.vstack((pop,offspring))
    fit_pop = np.append(fit_pop,fit_offspring)

    best = np.argmax(fit_pop) # Melhor solucao da geracao
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # Repete a melhor solucao, para problemas de instabilidade
    best_sol = fit_pop[best]

    # Selecao
    fit_pop_cp = fit_pop
    fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # Evitando probabilidades negativas, uma vez que o fitness varia de numeros negativos
    probs = (fit_pop_norm)/(fit_pop_norm).sum()
    chosen = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
    chosen = np.append(chosen[1:],best)
    pop = pop[chosen]
    fit_pop = fit_pop[chosen]


    # Procurando novas areas

    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 15:

        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\ndoomsday')
        file_aux.close()

        pop, fit_pop = doomsday(pop,fit_pop)
        notimproved = 0

    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)


    # Salvando resultados
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # Salvando o numero da geracao
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # Salvando o arquivo com a melhor solucao
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # Salvando o estado da simulacao
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()

    with open(experiment_name+'/results_1.txt', 'a') as f:
        for i in fit_pop:
            f.write(str(i))
            f.write(';')
        f.write('\n')


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

print( '\nCLONALG EVOLUTION\n')

# Gera os ids com a quantidade de clones que serão gerados a cada geracao, 
# com base o ranking de fitness
cloned_ids = clone()

with open(experiment_name+'/results_clonalg.txt', 'a') as f:
    for i in fit_pop:
        f.write(str(i))
        f.write(';')
    f.write('\n')

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


fim = time.time() # Tempo total
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')

# Demonstra o melhor individuo encontrado
print( '\nMelhor individuo encontrado:\n')
order = np.argsort(fit_pop)[::-1]
best_element = order[0:1]
final = evaluate(pop[best_element])
np.savetxt(experiment_name+'/best.txt',pop[best_element])

file = open(experiment_name+'/neuroended', 'w')  # Salva o arquivo de controle (simulacao encerrada) para o arquivo de loop bash
file.close()


env.state_to_log() # Verifica o estado do ambiente