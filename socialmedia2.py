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
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


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


def calculate_log_mean(lst):
    return np.log(sum(lst)/len(lst))


def calculate_mean(lst):
    return sum(lst)/len(lst)


def check_time(time, counter):
    try:
        time = time.astype(np.float)
        if time >= 100:
            counter += 1
    except ValueError:
        pass
    return counter


def scatter_plot(info, xlabel, ylabel, title):
    """Sequential coherence"""
    interface1_thread, interface1_conv, interface2_thread, interface2_conv = zip(*info)
    plt.scatter(interface1_thread, interface1_conv, color='r')
    plt.scatter(interface2_thread, interface2_conv, color='g')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    z1 = np.polyfit(interface1_thread, interface1_conv, 1)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(interface2_thread, interface2_conv, 1)
    p2 = np.poly1d(z2)
    plb.plot(interface1_thread, p1(interface1_thread), 'm-', color='r', label='Single interface 1')
    plb.plot(interface2_thread, p2(interface2_thread), 'm-', color='g', label='Double interface 2')
    plt.title(title)
    plt.legend()
    plt.show()


def bar_plot(bars1, bars2, ylabel1, xlabels, title):

    # width of the bars
    barWidth = 0.3

    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]


    # Create blue bars
    plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='Interface 1')

    # Create cyan bars
    plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='Interface 2')

    # general layout
    plt.xticks([r + barWidth for r in range(len(bars1))], xlabels)
    plt.ylabel('height')
    plt.legend()
    plt.title(title)
    # Show graphic
    plt.show()


def underscore_index(w_list):
    for i in range(len(w_list)):
        if w_list[i] == '_':
            return i


def interface_splitter(matrix):
    """ Splits the matrix for each interface. This is done by looking at the
        file that ends with _w_.txt, or the 4th column in our matrix. It checks
        for a sequence of 20 dashes, then it is the single line interface, or
        for underscores, which means the double line interface was used. This
        is to check with which interface the conversation started.
        Then it goes through the list of dashes and underscores again to
        determine the index where to split. If it started with the double line
        interface, we go through the reversed list until an underscore is
        found, which is where to split. If it started with the single line
        interface, we go through the normal list.
        We split each of the arrays in the matrix, append them to a new matrix
        and return that one."""
    interface_chars = matrix[3]
    underscore_counter = 0
    dash_counter = 0
    index_single_to_double = None
    index_double_to_single = None
    for i in range(len(interface_chars)):
        if interface_chars[i] == '_':
            underscore_counter += 1
        if interface_chars[i] == '-':
            dash_counter += 1
        if dash_counter == 20 and underscore_counter == 0:
            break
        if underscore_counter == 20:
            break
    if underscore_counter == 20:
        reversed_chars = np.flip(interface_chars)
        index_double_to_single = underscore_index(reversed_chars)
    if dash_counter == 20:
        index_single_to_double = underscore_index(interface_chars)
    interface_matrix_single = []
    interface_matrix_double = []
    if index_single_to_double:
        for i in range(len(matrix)):
            interface_matrix_single.append(matrix[i][:index_single_to_double])
            interface_matrix_double.append(matrix[i][index_single_to_double:])
    if index_double_to_single:
        for i in range(len(matrix)):
            interface_matrix_double.append(matrix[i][:index_double_to_single])
            interface_matrix_single.append(matrix[i][index_double_to_single:])
    return interface_matrix_single, interface_matrix_double


def thread_length_time(matrix):
    """ Counts the number of turns in each thread, and time each thread takes.
        It goes through the s and o file and """
    thread_length_counter = []
    time_length_counter = []
    turn_counter = 0
    time_counter = 0
    prev_o = None
    prev_s = None
    sd = ['s', 'd']
    for s, o, t in zip(matrix[0], matrix[1], matrix[2]):
        if s and prev_o or o and prev_s:
            turn_counter += 1
            time_counter += t.astype(np.float)
        if prev_o == '/' and o in sd or prev_s == '/' and s in sd:
            thread_length_counter.append(turn_counter)
            time_length_counter.append(time_counter)
            turn_counter = 0
            time_counter = 0
        prev_o = o
        prev_s = s
    return thread_length_counter, time_length_counter

#
# def thread_splitter(matrix):
#     for s, o, t,


def structure_data(list1, list2, list3, list4):
    structure_lst = []
    l1 = iter(list1)
    l2 = iter(list2)
    l3 = iter(list3)
    l4 = iter(list4)
    for i in range(len(list1)):
        structure_lst.append((next(l1), next(l2), next(l3), next(l4)))
    return structure_lst


def get_text(matrix):
    s_conv = ""
    o_conv = ""
    for s, o in zip(matrix[0], matrix[1]):
        if not s:
            s = " "
        if not o:
            o = " "
        s_conv += s
        o_conv += o
    s_conv = s_conv.lower()
    o_conv = o_conv.lower()
    return s_conv, o_conv


def cosine_similarity_and_sentiment(matrix):
    s_conv, o_conv = get_text(matrix)
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform([s_conv, o_conv])
    lst = (tfidf * tfidf.T).A
    all_comp, all_pos, all_neu, all_neg = sentimentfinder(s_conv + o_conv)
    compound1, pos1, neu1, neg1 = sentimentfinder(s_conv)
    compound2, pos2, neu2, neg2 = sentimentfinder(o_conv)

    all_tup = (all_comp, all_pos, all_neu, all_neg)
    tup = (compound1, pos1, neu1, neg1, compound2, pos2, neu2, neg2)


    return lst[0][1], all_tup, tup



def sentimentfinder(string):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(string)
    return scores['compound'], scores['pos'], scores['neu'], scores['neg']


def sentiment_threads(matrix, ):


    conv = ""
    for i in range(len(matrix[0])):
        prev_s = matrix[0][i]
        prev_o = matrix[1][i]

        # if not matrix[0][i] and prev_s:
        #     print("1")
        #     conv = conv +  " " + matrix[1][i]


        if not matrix[0][i] and prev_o:

            print("2")
            conv = conv +  matrix[1][i]


        # if not matrix[1][i] and prev_o:

        #     print("3")
        #     conv = conv +  " " + matrix[0][i]

        if not matrix[1][i] and prev_s:

            print("4")
            conv = conv +  matrix[0][i]


        print(conv)
        threads = re.split('/[sd]', conv)
        sent_lst = []
        pos_counter = 0
        for t in threads:
            comp, pos, neu, neg = sentimentfinder(t)

            if pos > neg:
                pos_counter += 1
        #print(pos_counter, len(threads))
        return (pos_counter, len(threads))


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
            #matrix_lst.append(m)
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
    sim_lst1 = []
    sim_lst2 = []
    sent_lst1 = []
    sent_lst2 = []
    sent_lst3 = []
    sent_lst4 = []
    for i in matrix_lst2:
        interface1, interface2 = interface_splitter(i)
        # print("Len i:", len(i[0]), " len interface1: ", len(interface1[0]), " len interface2: ", len(interface2[0]), " len sum: ", (len(interface1[0]) + len(interface2[0])))
        thr_length1, time_length1 = thread_length_time(interface1)
        thr_length2, time_length2 = thread_length_time(interface2)
        conversations.append((thr_length1, thr_length2))
        similiarity1, all_sent1, sent_so1 = cosine_similarity_and_sentiment(interface1)
        similiarity2, all_sent2, sent_so2 = cosine_similarity_and_sentiment(interface2)


        mean1 = calculate_log_mean(thr_length1)
        mean2 = calculate_log_mean(thr_length2)
        mean3 = calculate_log_mean(time_length1)
        mean4 = calculate_log_mean(time_length2)

        thr_means1.append(mean1)
        thr_means2.append(mean2)
        time_means1.append(mean3)
        time_means2.append(mean4)
        sim_lst1.append(similiarity1)
        sim_lst2.append(similiarity2)
        sent_lst1.append(sent_so1)
        sent_lst2.append(sent_so2)
        sent_lst3.append(all_sent1)
        sent_lst4.append(all_sent2)
        sent_tup1 = sentiment_threads(interface1)
        sent_tup2 = sentiment_threads(interface2)
    turns_lst1 = []
    turns_lst2 = []

    compound1s, pos1s, neu1s, neg1s, compound1o, pos1o, neu1o, neg1o = zip(*sent_lst1)
    compound2s, pos2s, neu2s, neg2s, compound2o, pos2o, neu2o, neg2o = zip(*sent_lst2)
    compound3, pos3, neu3, neg3 = zip(*sent_lst3)
    compound4, pos4, neu4, neg4 = zip(*sent_lst4)


    compound_mean1s = calculate_mean(compound1s)
    pos_means1s = calculate_mean(pos1s)
    neu_means1s = calculate_mean(neu1s)
    neg_means1s = calculate_mean(neg1s)
    compound_means1o = calculate_mean(compound1o)
    pos_means1o = calculate_mean(pos1o)
    neu_means1o = calculate_mean(neu1o)
    neg_means1o = calculate_mean(neg1o)

    q4_interface1 = (compound_mean1s, pos_means1s, neu_means1s, neg_means1s, compound_means1o, pos_means1o, neu_means1o, neg_means1o)

    compound_mean2s = calculate_mean(compound1s)
    pos_means2s = calculate_mean(pos2s)
    neu_means2s = calculate_mean(neu2s)
    neg_means2s = calculate_mean(neg2s)
    compound_means2o = calculate_mean(compound2o)
    pos_means2o = calculate_mean(pos2o)
    neu_means2o = calculate_mean(neu2o)
    neg_means2o = calculate_mean(neg2o)

    q4_interface2 = (compound_mean2s, pos_means2s, neu_means2s, neg_means2s, compound_means2o, pos_means2o, neu_means2o, neg_means2o)

    compound_means3 = calculate_mean(compound3)
    pos_means3 = calculate_mean(pos3)
    neu_means3 = calculate_mean(neu3)
    neg_means3 = calculate_mean(neg3)
    q5_interface1 = (compound_means3, pos_means3, neu_means3, neg_means3)

    compound_means4 = calculate_mean(compound4)
    pos_means4 = calculate_mean(pos4)
    neu_means4 = calculate_mean(neu4)
    neg_means4 = calculate_mean(neg4)
    q5_interface2 = (compound_means4, pos_means4, neu_means4, neg_means4)





    for i1, i2 in conversations:
        turns_lst1.append(sum(i1))
        turns_lst2.append(sum(i2))





    q1 = structure_data(thr_means1, turns_lst1, thr_means2, turns_lst2)
    q2 = structure_data(thr_means1, time_means1, thr_means2, time_means2)
    q3 = structure_data(thr_means1, sim_lst1, thr_means2, sim_lst2)


    # scatter_plot(q1, 'Mean thread length (log)', 'Conversation length', 'Sequence coherence')
    # scatter_plot(q2, 'Mean thread length (log)', 'Mean time length (log)', 'Thread length vs Time')
    # scatter_plot(q3, 'Mean thread length (log)', 'Cosine similiarity', 'Cosine similiarity vs Thread length')
    # bar_plot(q4_interface1, q4_interface2, "Mean", ['Compound_self', 'Positive_self', 'Neutral_self',
    #                                                 'Negative_self', 'Compound_other', 'Positive_other',
    #                                                 'Neutral_other', 'Negative_other'], "Happier interface.")
    # bar_plot(q5_interface1, q5_interface2, "Mean", ['Compound', 'Positive', 'Neutral', 'Negative'], "Emotional Synchrony.")
    # bar_plot(sent_tup1, sent_tup2, "Correct answers", ['Check', 'Total threads'], "Predicting correct answers.")

if __name__ == '__main__':
    main()
