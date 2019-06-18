#!/usr/bin/python3
# student: J.F.P. (Richard) Scholtens
# studentnr.: s2956586
# datum: 231/05/2019

import csv
import os
from contextlib import ExitStack
import numpy as np
from matplotlib import pyplot as plt
#import pprint


# def show_plot(x_freq, x_time, y_freq, y_time):
#     """Shows a plot with purity and rand-index scores on the y-axis. On
#     the x-axis the number of clusters is shown. It also adds a title and
#     legenda."""



#     percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
#     frequency = linspace(0, (x_freq + y_freq), 1000);


#     lenght = x_freq + y_freq
#     x_perc = 100 / lenght * x_freq
#     y_perc = 100 / lenght * y_freq
#     l = [i for i in range(int(lenght))]

#     #print(l)
#     m = [percentages, x_perc, y_perc]

#     # Creates plot
#     plt.title("Cluster, purity,and rand-index relationships.")
#     plt.plot(m[0], m[1], label="Frequency")
#     plt.plot(m[0], m[2], label="Time")
#     plt.xlabel("Number of clusters. (K-Means)")
#     plt.ylabel("Purity/Rand-index")
#     plt.legend()
#     plt.show()

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def exponantional_lst(np_lst):
    sum_lst = []
    c = 0
    for i in np_lst:
        for ii in i:
            try:
                print(ii)
                c += ii.astype(np.float)
            except ValueError:
                pass
        sum_lst.append(c)
    return sum_lst


def show_plot(info_lst):
    """Shows a plot with purity and rand-index scores on the y-axis. On
    the x-axis the number of clusters is shown. It also adds a title and
    legenda."""

    e1 = info_lst[0]
    e2 = info_lst[1]
    e3 = info_lst[2]
    e4 = info_lst[3]


    e1_divided = chunks(e1[3:], 10)
    e2_divided = chunks(e2[3:], 10)
    e3_divided = chunks(e3[3:], 10)
    #e4_divided = chunks(e4, 10)

    print(e1_divided)

    e1_expo = exponantional_lst(e1_divided)
    print(e1_expo)
    e2_expo = exponantional_lst(e2_divided)
    time = exponantional_lst(e3_divided)
    #e4_expo = exponantional_lst(e4_divided)

    freq = [i for i in range(len(time))]




    m = [freq, time, freq]

    # Creates plot
    plt.title("Cluster, purity,and rand-index relationships.")
    plt.plot(m[0], m[1], label="Frequency")
    plt.plot(m[0], m[2], label="Time")
    plt.xlabel("Number of clusters. (K-Means)")
    plt.ylabel("Purity/Rand-index")
    plt.legend()
    plt.show()



def main():
    filenames = [txt_file for txt_file in os.listdir("data/s3492338")]
    separate_freq = 0
    together_freq = 0
    separate_time = 0
    together_time = 0

    lst = []
    with ExitStack() as stack:
        files = [stack.enter_context(open('data/s3492338/' + fname)) for fname in filenames]
        for f1, f2, f3, f4 in zip(*files):
            e1 = np.array(f1.split('¦'))
            e2 = np.array(f2.split('¦'))
            e3 = np.array(f3.split('¦'))
            e4 = np.array(f4.split('¦'))
            lst.append(e1)
            lst.append(e2)
            lst.append(e3)
            lst.append(e4)
            m = np.column_stack((e1, e2, e3[2:], e4))
            for i in range(len(m)):
                #print(m[i][0], m[i][1], m[i][2], m[i][3]) # , m[i][4], m[i][5], m[i][6], m[i][7])

                if m[i][3] == "_":
                    together_freq += 1
                    together_time += m[i][2].astype(np.float)
                else:
                    #print(m[i][2], type(m[i][2]))
                    separate_freq += 1
                    try:
                        separate_time += m[i][2].astype(np.float)
                    except ValueError:
                        pass

    #show_plot(together_freq, together_time, separate_time, separate_time)
    show_plot(lst)
    print(together_freq, together_time, separate_freq, separate_time)
if __name__ == '__main__':
    main()