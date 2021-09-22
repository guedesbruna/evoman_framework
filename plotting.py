import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.chdir('./dummy_demo')

#function for plotting line plots for mean and maximum fitness

def line_plot(nexp = 10): #change this number to match number of runs
    avg = pd.DataFrame()
    for i in range(nexp):

        # read each results file and create a dataframe

        locals()['df'+str(i)] = pd.read_csv('results_'+str(i)+'.txt', sep= ' ', header=None,\
                                            names=['gen', 'best', 'mean', 'std', 'gain'])

        # at first iteration copies the dataframe, in successive ones computes mean of dataframes

        if i == 0:
            avg = locals()['df'+str(i)]
        else:
            avg['best'] = pd.concat([avg['best'], locals()['df'+str(i)]['best']]).mean(level=0)
            avg['mean'] = pd.concat([avg['mean'], locals()['df'+str(i)]['mean']]).mean(level=0)
            avg['std'] = pd.concat([avg['std'], locals()['df' + str(i)]['std']]).mean(level=0)

    # make number of generations start from 1

    avg['gen'] = avg['gen'] + np.ones(avg.shape[0])
    print(avg)

    # Plotting

    plt.plot(avg['gen'], avg['mean'], color = 'blue', label = 'Mean EA1')
    plt.plot(avg['gen'], avg['best'], linestyle= '--',  color = 'blue', label = 'Maximum EA1')
    plt.fill_between(range(1, avg.shape[0]+1), avg['mean'] - avg['std'], avg['mean'] + avg['std'], alpha = 0.5)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig('figure2.png')
    plt.show()


#function for plotting boxplots for best individual

def box_plot():
    pass

line_plot()