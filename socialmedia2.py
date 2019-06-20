#!/usr/bin/python3
# student: J.F.P. (Richard) Scholtens
# studentnr.: s2956586
# datum: 231/05/2019


from contextlib import ExitStack
import numpy as np
import glob
# import plotly.plotly as py
# import plotly.tools as tls
from matplotlib import pyplot as plt
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


def calculate_means(lst):
    all_means = []
    for i in lst:
        mean = i / sum(lst)
        all_means.append(mean)
    return all_means


def check_time(time, counter):
    try:
        time = time.astype(np.float)
        if time >= 100:
            counter += 1
    except ValueError:
        pass
    return counter


def scatter_plot(means):
    """Sequential coherence"""
    for x1, y1, x2, y2 in means:
        plt.scatter(x1, y1)
        plt.scatter(x2, y2)
    plt.show()


def retrieve_means(matrices):
    """Retrieves the mean thread length and mean conversation length
    per conversation and per interfaces. It returns a list with tuples.
    Every tuple is structured as following:
    (mean tread length interface 1, mean conversation length interface1,
    mean tread length interface 2, mean conversation length interface2)"""
    thread_lst1 = []
    thread_lst2 = []
    conversations_lst1 = []
    conversations_lst2 = []
    for m in matrices:
        c1 = c2 = c3 = 0
        conv1 = ""
        conv2 = ""
        check = False
        sec_m = np.flip(m)  # This is needed in order to determine the range of the double interpreter.
        for i in range(len(sec_m)):
            if sec_m[i][0] != "_" and check is False:
                c1 += 1
            else:
                check = True

        for i in range(len(m[:-c1])):
            conv1 += m[i][0]
            c2 = check_time(m[i][2], c2)
        conversations_lst1.append(c2)

        for i in range(c1):
            conv2 += sec_m[i][3]
            c3 = check_time(sec_m[i][1], c3)
        conversations_lst2.append(c3)

        threads1 = re.split('/[sd]', conv1)
        threads2 = re.split('/[sd]', conv2)

        thread_lst1.append(threads1)
        thread_lst2.append(threads2)

    thr_all1 = get_thread_lengths(thread_lst1)
    thr_all2 = get_thread_lengths(thread_lst1)

    thr_means1 = calculate_means(thr_all1)
    thr_means2 = calculate_means(thr_all2)

    conv_means1 = calculate_means(conversations_lst1)
    conv_means2 = calculate_means(conversations_lst2)

    means_lst = []
    l1 = iter(thr_means1)
    l2 = iter(conv_means1)
    l3 = iter(thr_means2)
    l4 = iter(conv_means2)
    for i in range(len(thr_means1)):
        means_lst.append((next(l1), next(l2), next(l3), next(l4)))
    return means_lst


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
            print(len(e1), len(e2), len(e3), len(e4))
            m = np.column_stack((e1, e2, e3[2:], e4))
            matrix_lst.append(m)
            matrix_lst2.append((e1, e2, e3, e4))
            # for i in range(len(m)):
            #     print(m[i][0], m[i][1], m[i][2], m[i][3]) # , m[i][4], m[i][5], m[i][6], m[i][7])

    means = retrieve_means(matrix_lst)

    scatter_plot(means)


if __name__ == '__main__':
    main()
