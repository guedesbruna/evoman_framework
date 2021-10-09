import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
from math import fabs, sqrt, exp, e
import glob, os

# choose this for not using visuals and thus making experiments faster
headless = True
if headless: os.environ["SDL_VIDEODRIVER"] = "dummy"

# selects the folder containing the best individuals, change it to run for another EA/enemy

en = [4,8] # change this list to change enemy group
ea = 1

if len(en) == 2: experiment_name = 'dummy_demo_Task2/EA'+str(ea)+' E' + str(en[0]) + '_' +str(en[1])
else: experiment_name = 'dummy_demo_Task2/EA'+str(ea)+' E' + str(en[0]) + '_' +str(en[1]) + '_' +str(en[2])

n_hidden_neurons = 10

for enemy in en:

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=enemy,  # array with 1 to 8 items
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

    run_mode = 'test'  # train or test

    def simulation(env, x):
        f, p, e, t = env.play(pcont=x)  # fitness, player, enemy, time
        return p - e

    def evaluate(x):
        return np.array(list(map(lambda y: simulation(env, y), x)))

    # loads file with the best solution for testing
    if run_mode == 'test':

        for j in range(0,10):
            for k in range(0,5):
                bsol = np.loadtxt(experiment_name + '/best_ind'+str(j)+'.txt')
                file_aux = open(experiment_name + '/bi_results_' + str(j) +'en'+str(enemy)+ '.txt', 'a')
                print('\n RUNNING SAVED BEST SOLUTION '+str(j)+'\n')
                env.update_parameter('speed', 'fastest')
                fit = evaluate([bsol])[0]
                file_aux.write('\n' + str(k)+ ' ' +str(round(fit, 6)))
                file_aux.close()



sys.exit(0)