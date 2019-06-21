#!/usr/bin/python3
# student: J.F.P. (Richard) Scholtens
# studentnr.: s2956586
# datum: 231/05/2019


from contextlib import ExitStack
import numpy as np
import glob

from matplotlib import pyplot as plt
import matplotlib.pylab as plb
from collections import Counter
import re


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_thread_lengths(lst):
    all_length = []
    length = 0
    for i in lst:
        for ii in i:
            length += len(ii)
        all_length.append(length)
    return all_length


def calculate_mean(lst):
    return np.log(sum(lst)/len(lst))
    #return sum(lst)/len(lst)


def check_time(time, counter):
    try:
        time = time.astype(np.float)
        if time >= 100:
            counter += 1
    except ValueError:
        pass
    return counter


def scatter_plot(means, xlabel, ylabel):
    """Sequential coherence"""
    interface1_thread, interface1_conv, interface2_thread, interface2_conv = zip(*means)
    plt.scatter(interface1_thread, interface1_conv, color='r')
    plt.scatter(interface2_thread, interface2_conv, color='g')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    z1 = np.polyfit(interface1_thread, interface1_conv, 1)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(interface2_thread, interface2_conv, 1)
    p2 = np.poly1d(z2)
    plb.plot(interface1_thread, p1(interface1_thread), 'm-', color='g', label='Single interface 1')
    plb.plot(interface2_thread, p2(interface2_thread), 'm-', color='r', label='Double interface 2')
    plt.legend()
    plt.show()


def interface_splitter(matrix):
    reversed = np.flip(matrix[3])
    for i in range(len(reversed)):
        if reversed[i] == '_':
            index = i
            break
    interface_matrix1 = []
    interface_matrix2 = []
    for i in range(len(matrix)):
        interface_matrix1.append(matrix[i][:-index])
        interface_matrix2.append(matrix[i][index:])
    return interface_matrix1, interface_matrix2


def thread_length_time(matrix):
    thread_length_counter = []
    time_length_counter = []
    turn_counter = 0
    time_counter = 0
    prev_o = None
    prev_s = None
    for s, o, t in zip(matrix[0], matrix[1], matrix[2]):
        if s and prev_o or o and prev_s:
            turn_counter += 1
            time_counter += t.astype(np.float)
        if o == '/' or s == '/':
            thread_length_counter.append(turn_counter)
            time_length_counter.append(time_counter)
            turn_counter = 0
            time_counter = 0
        prev_o = o
        prev_s = s
    return thread_length_counter, time_length_counter


def structure_data(list1, list2, list3, list4):
    structure_lst = []    
    l1 = iter(list1)
    l2 = iter(list2)
    l3 = iter(list3)
    l4 = iter(list4)
    for i in range(len(list1)):
        structure_lst.append((next(l1), next(l2), next(l3), next(l4)))
    return structure_lst


def main():
    matrix_lst = []
    matrix_lst2 = []
    t_list = []
    s_list = []
    w_list = []
    o_list = []
    for filename in glob.glob('data/*/*'):
        if filename.endswith("_t_.txt"):
            t_list.append(filename)
        elif filename.endswith("_s_.txt"):
            s_list.append(filename)
        elif filename.endswith("_w_.txt"):
            w_list.append(filename)
        else:
            o_list.append(filename)
    for o, s, t, w in zip(o_list, s_list, t_list, w_list):
        with open(o, 'r') as f1, open(s, 'r') as f2, open(t, 'r') as f3, open(w, 'r') as f4:
            f1 = f1.read()
            f2 = f2.read()
            f3 = f3.read()
            f4 = f4.read()
            e1 = np.array(f1.split('¦'))
            e2 = np.array(f2.split('¦'))
            e3 = np.array(f3.split('¦'))
            e4 = np.array(f4.split('¦'))
            # print(len(e1), len(e2), len(e3[2:]), len(e4))
            m = np.column_stack((e1, e2, e3[2:], e4))
            matrix_lst.append(m)
            matrix_lst2.append((e1, e2, e3[2:], e4))
            # for i in range(len(m)):
            #     print(m[i][0], m[i][1], m[i][2], m[i][3]) # , m[i][4], m[i][5], m[i][6], m[i][7])

    # means = retrieve_means(matrix_lst)
            # for i in m:
            #     print(i)


    conversations = []
    thr_means1 = []
    thr_means2 = []
    time_means1 = []
    time_means2 = []

    for i in matrix_lst2:
        interface1, interface2 = interface_splitter(i)
        thr_length1, time_length1 = thread_length_time(interface1)
        thr_length2, time_length2 = thread_length_time(interface2)
        conversations.append((thr_length1, thr_length2))
        mean1 = calculate_mean(thr_length1)
        mean2 = calculate_mean(thr_length2)

        mean3 = calculate_mean(time_length1)
        mean4 = calculate_mean(time_length2)


        thr_means1.append(mean1)
        thr_means2.append(mean2)
        time_means1.append(mean3)
        time_means2.append(mean4)

    turns_lst1 = []
    turns_lst2 = []


    for i1, i2 in conversations:
        turns_lst1.append(sum(i1))
        turns_lst2.append(sum(i2))


    q1 = structure_data(thr_means1, turns_lst1, thr_means2, turns_lst2)
    q2 = structure_data(thr_means1, time_means1, thr_means2, time_means2)


    scatter_plot(q1, 'Mean thread length (log)', 'Conversation length')
    scatter_plot(q2, 'Mean thread length (log)', 'Mean time length (log)')


if __name__ == '__main__':
    main()
