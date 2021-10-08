###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs, sqrt, exp, e
import glob, os

# choose this for not using visuals and thus making experiments faster
headless = True #set to False to view game
if headless:
     os.environ["SDL_VIDEODRIVER"] = "dummy"

# experiment_name = 'individual_demo'
experiment_name = 'dummy_demo_Task2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  multiplemode = "yes",
                  enemies=[1, 2],  # array with 1 to 8 items
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  contacthurt='player',
                  speed="fastest",
                  randomini="yes")

# default environment fitness is assumed for experiment
env.state_to_log()  # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train'  # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

dom_u = 1
dom_l = -1
npop = 10 # 100
gens = 10  # 30
mutation = 0.2  # 0.2
last_best = 0


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)  # fitness, player, enemy, time
    return f, p - e

def simulation_fitrew(env, x, gen, fitness_previous):
    f, p, e, t = env.play(pcont=x)
    C = 1 #fixed
    alpha = 1.5 #between 1 and 2
    W = (C*gen)**alpha #(the more generations passes the bigger the reward if fitness is better than previous)
    if np.mean(fitness_previous) < np.mean(f):
        #since we are maximizing, it's not a penalty but a reward concept
        f = f + W * 1 #reward hardcoded here 
    else:
        f = f + W * 0 #reward hardcoded here 
    return f, p-e

# normalizes
def norm(x, pfit_pop):
    if (max(pfit_pop) - min(pfit_pop)) > 0:
        x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


# evaluation
def evaluate(x, gen, fitness_previous): #originally just x
    #return np.array(list(map(lambda y: simulation(env, y), x)))
    #print(np.array(list(map(lambda y: simulation(env, y), x))
    return np.array(list(map(lambda y: simulation_fitrew(env, y,gen, fitness_previous), x))) #simulation only by default


# tournament for comparing 2 individuals
def tournament(pop):
    c1 = np.random.randint(0, pop.shape[0], 1)
    c2 = np.random.randint(0, pop.shape[0], 1)

    if fit_pop[c1] > fit_pop[c2]:
        return pop[c1][0]
    else:
        return pop[c2][0]

def rank_tournament(pop):
    pop_sorted = np.argsort(-fit_pop)

    probs = []
    n = npop
    s = 1.5  # between 1 and 2: if set to two, the lowest rank will never be picked

    for i in range(len(pop_sorted)):
        p = (2 - s) / n + ((2 * i) * (s - 1)) / (n * (n - 1))
        probs.append(p)

    probs = probs / np.sum(probs)
    c1 = np.random.choice(pop_sorted, p=probs)
    c2 = np.random.choice(pop_sorted, p=probs)

    if fit_pop[c1] > fit_pop[c2]:
        return pop[c1], c1
    else:
        return pop[c2], c2

def rank_selection(pop):
    # select parent based on rank selection method

    # rank population based on fitness (high to low)
    pop_sorted = np.argsort(-fit_pop)
    probs = []
    n = npop
    s = 1.5  # between 1 and 2: if set to two, the lowest rank will never be picked

    for i in range(len(pop_sorted)):
        p = (2 - s) / n + ((2 * i) * (s - 1)) / (n * (n - 1))
        probs.append(p)

    # pick a parent based on the probs
    probs = probs / np.sum(probs)
    picked_index = np.random.choice(pop_sorted, p=probs)

    return pop[picked_index]


# limits
def limits(x):
    if x > dom_u:
        return dom_u
    elif x < dom_l:
        return dom_l
    else:
        return x

#check whether two parents are the same
def check_duplicates(p_index1, p_index2, pop):
    while True:
        if p_index1 != p_index2:
            return pop[p_index1], p_index1
        p1, p_index1 = rank_tournament(pop)

# crossover

def crossover(pop):
    total_offspring = np.zeros((0, n_vars))
    for p in range(0, pop.shape[0], 2): #make whole pop a parent (so number of couples is npop/2)
        p1, p_index1 = rank_tournament(pop)
        p2, p_index2 = rank_tournament(pop)

        p1, p_index1 = check_duplicates(p_index1, p_index2, pop)

        # n_offspring =   np.random.randint(1,3+1, 2)[0]
        n_offspring = int(2) 
        offspring = np.zeros((n_offspring, n_vars))

        for f in range(0, n_offspring):

            cross_prop = np.random.uniform(0, 1)
            offspring[f] = p1 * cross_prop + p2 * (1 - cross_prop)

            # mutation
            # mutate each gene with prob. for each offspring if rand < than mutation rate
            # mutation strategy used: uncorrelated mutation with one size step.
            # obs: still missing rule to force stepsize to be bigger than threshold

            tau = 1 / npop ** (1 / 2)
            sigma = 1
            sigma = sigma * e ** (tau * np.random.normal(0, 1))  # mutation step size
            # while sigma < 0.1: #not necessary, never get's too low when sigma starts at 1. starts at 1 because it is a Gaussian
            #    sigma = sigma * e ** (tau * np.random.normal(0, 1))

            for i in range(0, len(offspring[f])):
                if np.random.uniform(0, 1) <= mutation:  # and mutation is 0.2 by default
                    offspring[f][i] = offspring[f][i] + sigma * np.random.normal(0, 1)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring


# performs (μ and λ) selection with μ = population size and λ = offspring size
# both parents and children can survive, with higher probability if their fitness if higher
def mu_and_lambda(pop, offspring, gen, fitness_previous):
    #combine parents and offspring
    pop = np.vstack((pop,offspring))

    # sort population from highest fitness to lowest fitness
    fit_pop = evaluate(pop, gen, fitness_previous)[:, 0]
    pop_sorted = np.argsort(-fit_pop)  # returns indexes (from index with highest fitness to index with lowest fitness)

    # pick the best genomes and put them in the new population
    new_pop = np.empty((npop, n_vars))
    i = 0
    for p in pop_sorted[:npop]:
        new_pop[i] = pop[p]
        i += 1

    # make sure the population size has the size npop
    assert len(new_pop) == npop, print('PopulationSize Error')
    new_fit_pop = evaluate(new_pop, gen, fitness_previous)[:, 0]
    new_fit_pop_ind = evaluate(new_pop, gen, fitness_previous)[:, 1]

    return new_pop, new_fit_pop, new_fit_pop_ind

# performs (μ, λ) selection with μ = population size and λ = offspring size
# only (and all) children survive
def mu_lambda(pop, offspring, gen, fitness_previous):
    pop = offspring
    fit_pop = evaluate(pop, gen, fitness_previous)[:, 0]
    fit_pop_ind = evaluate(pop, gen, fitness_previous)[:, 1]

    return pop, fit_pop, fit_pop_ind


# kills the worst genomes, and replace with new best/random solutions
def doomsday(pop, fit_pop, gen, fitness_previous):
    worst = int(npop / 4)  # a quarter of the population
    order = np.argsort(fit_pop)
    orderasc = order[0:worst]

    for o in orderasc:
        for j in range(0, n_vars):
            pro = np.random.uniform(0, 1)
            if np.random.uniform(0, 1) <= pro:
                pop[o][j] = np.random.uniform(dom_l, dom_u)  # random dna, uniform dist.
            else:
                pop[o][j] = pop[order[-1:]][0][j]  # dna from best

        fit_pop[o] = evaluate([pop[o]], gen, fitness_previous)[:, 0]

    return pop, fit_pop

def explore(pop, fit_pop, gen, fitness_previous):
    n_replace = int(npop*0.20)
    to_replace = np.argsort(fit_pop)[:n_replace] #select random 15% genomes

    #create new, random genome
    for g in to_replace:
        for v in range(0, n_vars):
            pop[g][v] = np.random.uniform(-1, 1)
        fit_pop[g] = evaluate([pop[g]], gen, fitness_previous)[:, 0]

    fit_pop_ind = evaluate(pop, gen, fitness_previous)[:, 1]
    return pop, fit_pop, fit_pop_ind



# loads file with the best solution for testing
if run_mode == 'test':
    bsol = np.loadtxt(experiment_name + '/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    evaluate([bsol], 0, -100)[:, 0]

    sys.exit(0)

# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name + '/evoman_solstate'):

    print('\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop, 0, -100)[:, 0]
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print('\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux = open(experiment_name + '/gen.txt', 'r')
    ini_g = int(file_aux.readline())
    file_aux.close()

# evolution


last_sol = fit_pop[best]
notimproved = 0
j = 0
for j in range(0, 10):  # number of runs
    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop, 0, -100)[:, 0]  #fitness
    fit_pop_ind = evaluate(pop, 0, -100)[:, 1]  #gain = p-e
    best = np.argmax(fit_pop)
    best_ind = np.argmax(fit_pop_ind)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

    # saves results for first pop
    file_aux = open(experiment_name + '/results_' + str(j) + '.txt', 'a')
    # file_aux.write('\n\ngen best mean std')
    print('\n GENERATION ' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
        round(std, 6)) + ' ' + str(round(fit_pop_ind[best_ind], 6)))
    file_aux.write(
        '\n' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
            round(std, 6)) + ' ' + str(round(fit_pop_ind[best_ind], 6)))
    file_aux.close()


    for i in range(ini_g + 1, gens): # n generations
        prev_fit = fit_pop
        fit_pop = evaluate(pop, i, prev_fit)[:, 0]
        offspring = crossover(pop)
        best_sol = fit_pop[best]


        # selection
        pop, fit_pop, fit_pop_ind = mu_and_lambda(pop, offspring, i, prev_fit)

        #add some exploration to prevent getting stuck in local maximum
        if best_sol <= last_sol:
            notimproved += 1
        else:
            last_sol = best_sol
            notimproved = 0

        if notimproved >= 10:

            pop, fit_pop, fit_pop_ind = explore(pop, fit_pop)

            notimproved = 0

        # get results
        best = np.argmax(fit_pop)
        best_ind = np.argmax(fit_pop_ind)
        std = np.std(fit_pop)
        mean = np.mean(fit_pop)

        # saves results
        file_aux = open(experiment_name + '/results_' + str(j) + '.txt', 'a')
        print('\n GENERATION ' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
            round(std, 6)) + ' ' + str(round(fit_pop_ind[best_ind], 6)))
        file_aux.write(
            '\n' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
                round(std, 6)) + ' ' + str(round(fit_pop_ind[best_ind], 6)))
        file_aux.close()

        # # saves generation number. since they all have the same number of generation with no interruption all files have the same number
        # file_aux  = open(experiment_name+'/gen'+str(j)+'.txt','w')
        # file_aux.write(str(i))
        # file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name + '/best' + str(j) + '.txt', pop[best])
        np.savetxt(experiment_name + '/best_ind' + str(j) + '.txt', pop[best_ind])

        # saves simulation state
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        env.save_state()
    j += 1

fim = time.time()  # prints total execution time for experiment
print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

file = open(experiment_name + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log()  # checks environment state
