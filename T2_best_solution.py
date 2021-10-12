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

en = np.array([1,2,3,4,5,6,7,8])

experiment_name = 'dummy_demo_Task2/EA2 E2_5_7' #manually specify folder for best solution
bsol = np.loadtxt(experiment_name + '/best_ind5.txt') #manually specify best individual

n_hidden_neurons = 10

for enemy in en:

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[enemy],  # array with 1 to 8 items
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
        return p, e

    def evaluate(x):
        return np.array(list(map(lambda y: simulation(env, y), x)))

    # loads file with the best solution for testing
    if run_mode == 'test':
        for k in range(0,5):

            file_aux = open('optimalSol_E'+str(enemy)+'.txt', 'a')
            env.update_parameter('speed', 'fastest')
            fit = evaluate([bsol])[0]
            player = fit[0]
            enemyl = fit[1]
            file_aux.write('\n' + str(k)+ ' ' +str(player)+' '+str(enemyl))
            file_aux.close()



sys.exit(0)