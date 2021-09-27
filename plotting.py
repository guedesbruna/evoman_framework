import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

NEXP = 10

def construct_dataset(folder, filename, type):

    # create empty dataframe

    if type == 'line': avg = pd.DataFrame()
    elif type == 'box': avg = pd.DataFrame(columns=['run', 'gain'], dtype=float)

    # change directory

    os.chdir('./dummy_demo/'+str(folder))

    for i in range(NEXP):

        # read each results file and create a dataframe

        if type == 'line': df = pd.read_csv(str(filename)+'_'+str(i)+'.txt', sep= ' ', header=None,\
                                            names=['gen', 'best', 'mean', 'std', 'gain'])

        elif type == 'box': df = pd.read_csv(str(filename)+'_'+str(i)+'.txt', sep= ' ', header=None,\
                                            names=['run', 'gain'])

        # at first iteration copies the dataframe, in successive ones computes mean of dataframes

        if type == 'line':

            if i == 0: avg = df
            else:
                avg['best'] = pd.concat([avg['best'], df]).mean(level=0)
                avg['mean'] = pd.concat([avg['mean'], df]).mean(level=0)
                avg['std'] = pd.concat([avg['std'], df]).mean(level=0)

        elif type == 'box':

            avg.at[i, 'run'] = float(i)
            avg.at[i, 'gain'] = df['gain'].mean()

    # number of generations starts from 1

    if type == 'line': avg['gen'] = avg['gen'] + np.ones(avg.shape[0])

    os.chdir('..')

    return avg

#function for plotting line plots for mean and maximum fitness

def line_plot(enemy): # enemy (int or str with enemy number)

    # change directory to find results of EA1 for specified enemy

    avg_1 = construct_dataset('EA1 E' + str(enemy), 'results', 'line')
    avg_2 = construct_dataset('EA2 E' + str(enemy), 'results', 'line')

    # plotting

    plt.plot(avg_1['gen'], avg['mean'], color = 'blue', label = 'Mean EA1')
    plt.plot(avg_1['gen'], avg['best'], linestyle= '--',  color = 'blue', label = 'Maximum EA1')
    plt.plot(avg_2['gen'], avg['mean'], color='red', label='Mean EA2')
    plt.plot(avg_2['gen'], avg['best'], linestyle='--', color='red', label='Maximum EA2')
    plt.fill_between(range(1, avg_1.shape[0] + 1), avg_1['mean'] - avg_1['std'], avg_1['mean'] + avg_1['std'],\
                     alpha = 0.5)
    plt.fill_between(range(1, avg_2.shape[0] + 1), avg_2['mean'] - avg_2['std'], avg_2['mean'] + avg_2['std'], \
                     alpha=0.5)
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness')
    plt.title('Enemy ' + str(enemy))
    plt.legend()
    plt.savefig('line_plot_E' +str(enemy)+'.png')
    plt.show()


#function for plotting boxplots for best individual

def box_plot(enemy):

    avg_1 = construct_dataset('EA1 E' + str(enemy), 'bi_results', 'box')
    avg_2 = construct_dataset('EA2 E' + str(enemy), 'bi_results', 'box')

    plt.boxplot([avg_1['gain'], avg_2['gain']), patch_artist=True, labels=['EA1', 'EA2'])
    plt.xlabel('Evolutionary Algorithm')
    plt.ylabel('Individual Gain')
    plt.title('Best Individuals - Enemy ' + str(enemy))
    plt.legend()
    plt.savefig('boxplot_E' + str(enemy) + '.png')
    plt.show()

# run the function for line plot

line_plot()

# run the function for boxplot

box_plot()