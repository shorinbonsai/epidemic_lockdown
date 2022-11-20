from lib2to3.pgen2.tokenize import generate_tokens
import sys
import os
import csv
import math
import re
import string
from operator import itemgetter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_theme(style="ticks", color_codes=True)


def getFits(dir_path: str):
    with open(dir_path, "r",  errors="ignore") as f:
        final_line = f.readlines()[-2]
        final_line = final_line.split(',')
        best = final_line[1].split(' ')
        return float(best[-1])

def getRandom(dir_path: str) -> list[int]:
    result = []
    with open(dir_path, "r",  errors="ignore") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            result.append(float(line))
    return result


def main():
    stuff = [[[] for _ in range(2)] for _ in range(2)]
    stuff2 = [[[] for _ in range(2)] for _ in range(2)]
    for dirpath, dirnames, files in os.walk('.'):
        for file_name in files:
            if file_name.endswith(".csv") or file_name.endswith(".txt"):
                direc = os.path.join(dirpath, file_name)
                exper = []
                exper = dirpath.split('/')
                if file_name.__contains__("random"):
                    strictness = 12
                    best = getRandom(direc)
                    genetics = 1
                    if file_name.__contains__("strict"):
                        strictness = 0
                    elif file_name.__contains__("medium"):
                        strictness = 1
                    else:
                        continue
                    
                    if file_name.__contains__("50"):
                        removal = 0
                    elif file_name.__contains__("70"):
                        removal = 1
                    if removal == 0:
                        stuff[genetics][strictness] = best
                    elif removal == 1:
                        stuff2[genetics][strictness] = best
                    
                    
                elif file_name.__contains__("results"): 
                    best = getFits(direc)
                    genetics = 0
                    strictness = 12
                    removal = 5
                    if dirpath.__contains__("strict"):
                        strictness = 0
                    elif dirpath.__contains__("medium"):
                        strictness = 1
                    else:
                        continue
                    
                    if dirpath.__contains__("50"):
                        removal = 0
                    elif dirpath.__contains__("70"):
                        removal = 1
                    if removal == 0:
                        stuff[genetics][strictness].append(best)
                    elif removal == 1:
                        stuff2[genetics][strictness].append(best)


    df = pd.DataFrame()
    gene1 = stuff[0]
    randoms = stuff[1]
    # gene3 = stuff[2]
    # gene4 = stuff[3]
    experraw = ["Strict", "Medium"]
    # df = pd.DataFrame(
    #     {'Parameters': experraw, 'Random': randoms, '90/10': gene1})
    # df = df[['Parameters','Random', '90/10' ]]
    # dd = pd.melt(df, id_vars=['Parameters'], value_vars=[
    #              'Random', '90/10' ], var_name='Experiments')
    df = pd.DataFrame(
        {'Parameters': experraw, '90/10': gene1})
    df = df[['Parameters', '90/10' ]]
    dd = pd.melt(df, id_vars=['Parameters'], value_vars=[
                 '90/10' ], var_name='Experiments')
    result = dd.explode('value')
    result['value'] = result['value'].astype('float')
    ax = sns.boxplot(x='Parameters', y='value', data=result,
                     hue='Experiments')
    plt.legend([],[], frameon=False)
    ax.set(xlabel="Parameters", ylabel="Total Infected")
    ax.set_title("50% Removal")
    plt.show()


    df = pd.DataFrame()
    gene1 = stuff2[0]
    randoms = stuff2[1]
    # gene3 = stuff2[2]
    # gene4  =stuff2[3]
    experraw = ["Strict", "Medium"]
    df = pd.DataFrame(
        {'Parameters': experraw, '90/10': gene1})
    df = df[['Parameters', '90/10' ]]
    dd = pd.melt(df, id_vars=['Parameters'], value_vars=[
                 '90/10' ], var_name='Experiments')
    # df = pd.DataFrame(
    #     {'Parameters': experraw, 'Random': randoms, '90/10': gene1})
    # df = df[['Parameters','Random', '90/10' ]]
    # dd = pd.melt(df, id_vars=['Parameters'], value_vars=[
    #              'Random', '90/10'], var_name='Experiments')
    result = dd.explode('value')
    result['value'] = result['value'].astype('float')
    ax = sns.boxplot(x='Parameters', y='value', data=result,
                     hue='Experiments')
    plt.legend([],[], frameon=False)
    ax.set(xlabel="Parameters", ylabel="Total Infected")
    ax.set_title("70% Removal")
    plt.show()


    return


if '__main__' == __name__:
    main()
