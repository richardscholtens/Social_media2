#!/usr/bin/python3
# student: J.F.P. (Richard) Scholtens & R.P. (Rolf) Daling
# studentnr.: s2956586 & s2344343
# datum: 21/06/2019


import numpy as np
import glob
from matplotlib import pyplot as plt
import matplotlib.pylab as plb
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def calculate_log_mean(lst):
    """ Helper function to calculate the logarithmic mean. """
    return np.log(sum(lst)/len(lst))


def calculate_mean(lst):
    """ Helper function to calculate the mean. """
    return sum(lst)/len(lst)


def structure_data(list1, list2, list3, list4):
    """ Helper function to structure the data. """
    structure_lst = []
    l1 = iter(list1)
    l2 = iter(list2)
    l3 = iter(list3)
    l4 = iter(list4)
    for i in range(len(list1)):
        structure_lst.append((next(l1), next(l2), next(l3), next(l4)))
    return structure_lst


def scatter_plot(info, xlabel, ylabel, title):
    """ Plots a scatter plot"""
    interface1_thread, interface1_conv, interface2_thread, interface2_conv = zip(*info)
    plt.scatter(interface1_thread, interface1_conv, color='r')
    plt.scatter(interface2_thread, interface2_conv, color='g')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    z1 = np.polyfit(interface1_thread, interface1_conv, 1)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(interface2_thread, interface2_conv, 1)
    p2 = np.poly1d(z2)
    plb.plot(interface1_thread, p1(interface1_thread), 'm-', color='r', label='Single line interface')
    plb.plot(interface2_thread, p2(interface2_thread), 'm-', color='g', label='Double line interface')
    plt.title(title)
    plt.legend()
    plt.show()


def bar_plot(bars1, bars2, ylabel1, xlabels, title):
    """ Plots a bar graph with 2 bars. """
    # width of the bars
    barWidth = 0.3

    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    # Create blue bars
    plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='Single line interface')

    # Create cyan bars
    plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='Double line interface')

    # general layout
    plt.xticks([r + barWidth for r in range(len(bars1))], xlabels, rotation='vertical')
    plt.ylabel(ylabel1)
    plt.legend()
    plt.title(title)
    plt.subplots_adjust(bottom=0.3)
    # Show graphic
    plt.show()


def underscore_index(w_list):
    """ Returns the index of the first underscore that occurs in the list."""
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
        It goes through the splitted thread matrix and checks for both parties
        if someone outputted a character, and if the other party outputted the
        the previous character.
        The result is a list with the number of turns for each thread, and a
        list with the time that each thread takes. """
    thread_length_counter = []
    time_length_counter = []
    turn_counter = 0
    time_counter = 0
    prev_o = None
    prev_s = None
    thread_splitted_matrix = thread_splitter(matrix)
    for thread in thread_splitted_matrix:
        for s, o, t in zip(thread[0], thread[1], thread[2]):
            if s and prev_o or o and prev_s:
                turn_counter += 1
                time_counter += t.astype(np.float)
            prev_o = o
            prev_s = s
        thread_length_counter.append(turn_counter)
        time_length_counter.append(time_counter)
        turn_counter = 0
        time_counter = 0
    return thread_length_counter, time_length_counter


def thread_splitter(matrix):
    """ Splits the conversation into single threads. Each thread is seperated
        by a slash followed by either a 'd' or a 's'. So we look at the
        previous and current character of both conversationists, and find the
        index. Then we split every array of the matrix at this index and append
        it to the list.
        The result is a list of matrices for every thread. """
    thread_matrix = []
    thread = []
    thread_start = 0
    prev_s = None
    prev_o = None
    sd = ['s', 'd']
    for i in range(len(matrix[0])):
        s = matrix[0][i]
        o = matrix[1][i]
        if prev_o == '/' and o in sd or prev_s == '/' and s in sd:
            thread_end = i
            for j in range(len(matrix)):
                thread.append(matrix[j][thread_start+1:thread_end-1])
            thread_matrix.append(thread)
            thread = []
            thread_start = thread_end
        prev_s = s
        prev_o = o
    for i in range(len(matrix)):
        thread.append(matrix[i][thread_start:])
    thread_matrix.append(thread)
    return thread_matrix


def get_text(matrix):
    """ Takes the s and o lists of the matrix and returns the string of each
        both parts of the conversation. """
    s_conv = ""
    o_conv = ""
    prev_s = None
    prev_o = None
    for s, o in zip(matrix[0], matrix[1]):
        if not s and prev_s and prev_s != " ":
            s = " "
        if not o and prev_o and prev_o != " ":
            o = " "
        s_conv += s
        o_conv += o
        prev_s = s
        prev_o = o
    s_conv = s_conv.lower()
    o_conv = o_conv.lower()
    return s_conv, o_conv


def cosine_similarity_and_sentiment(matrix):
    """ Finds the cosine similiarity of both parties of the conversation using
        the TfidfVectorizer from sklearn. Also finds the sentiment for the
        conversation as a whole, and for both parties separately. """
    s_conv, o_conv = get_text(matrix)
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform([s_conv, o_conv])
    lst = (tfidf * tfidf.T).A
    all_comp, all_pos, all_neu, all_neg = sentimentfinder(s_conv + o_conv)
    compound1, pos1, neu1, neg1 = sentimentfinder(s_conv)
    compound2, pos2, neu2, neg2 = sentimentfinder(o_conv)

    sent_all = (all_comp, all_pos, all_neu, all_neg)
    sent_per_conv = (compound1, pos1, neu1, neg1, compound2, pos2, neu2, neg2)

    return lst[0][1], sent_all, sent_per_conv


def sentimentfinder(string):
    """ Finds the sentiment using the SentimentIntensityAnalyzer from NLTK. """
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(string)
    return scores['compound'], scores['pos'], scores['neu'], scores['neg']


def sentiment_threads(matrix):
    """ Splits the text into threads, gets the text for each thread of both
        parties, finds the sentiment for each thread and counts how often the
        positive value is higher than the negative value.
        Returns the number of times it was positive, and the number of threads.
        """
    thread_matrix = thread_splitter(matrix)
    s_conv = ""
    o_conv = ""
    pos_counter = 0
    for thread in thread_matrix:
        s_conv, o_conv = get_text(thread)
        conv = s_conv + " " + o_conv
        comp, pos, neu, neg = sentimentfinder(conv)
        if pos > neg:
            pos_counter += 1
    return (pos_counter, len(thread_matrix))


def get_filenames():
    """ Gets the filenames using glob, and returns a list for each type of
        file. """
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
    return s_list, o_list, t_list, w_list


def open_files(s_list, o_list, t_list, w_list):
    """ Opens the files from each list and splits them on the split characters.
        Returns the list of tuples with each of the 4 lists."""
    matrix_lst = []
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
            matrix_lst.append((e1, e2, e3[2:], e4))
    return matrix_lst


def main():
    s_list, o_list, t_list, w_list = get_filenames()
    matrix_lst = open_files(s_list, o_list, t_list, w_list)

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
    check_answers1 = []
    check_answers2 = []
    for i in matrix_lst:
        interface1, interface2 = interface_splitter(i)
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
        check_answers1.append(sentiment_threads(interface1))
        check_answers2.append(sentiment_threads(interface2))
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

    q4_interface1 = (compound_mean1s, compound_means1o, pos_means1s, pos_means1o, neu_means1s, neu_means1o, neg_means1s, neg_means1o)

    compound_mean2s = calculate_mean(compound1s)
    pos_means2s = calculate_mean(pos2s)
    neu_means2s = calculate_mean(neu2s)
    neg_means2s = calculate_mean(neg2s)
    compound_means2o = calculate_mean(compound2o)
    pos_means2o = calculate_mean(pos2o)
    neu_means2o = calculate_mean(neu2o)
    neg_means2o = calculate_mean(neg2o)

    q4_interface2 = (compound_mean2s, compound_means2o, pos_means2s, pos_means2o, neu_means2s, neu_means2o, neg_means2s, neg_means2o)

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
        turns_lst1.append(np.log(sum(i1)))
        turns_lst2.append(np.log(sum(i2)))

    q1 = structure_data(turns_lst1, thr_means1, turns_lst2, thr_means2)
    q2 = structure_data(thr_means1, time_means1, thr_means2, time_means2)
    q3 = structure_data(thr_means1, sim_lst1, thr_means2, sim_lst2)

    ch_ans1 = []
    ch_ans2 = []

    for i, j in zip(check_answers1, check_answers2):
        ch_ans1.append(i[0]/i[1])
        ch_ans2.append(j[0]/j[1])
    ch_ans1 = calculate_mean(ch_ans1)
    ch_ans2 = calculate_mean(ch_ans2)

    scatter_plot(q1, 'Conversation length in turns (log)', 'Mean thread length in turns (log)', 'Sequence coherence')
    scatter_plot(q2, 'Mean thread length in turns(log)', 'Mean length in seconds (log)', 'Thread length in turns vs time')
    scatter_plot(q3, 'Mean thread length (log)', 'Cosine similiarity', 'Cosine similiarity vs Thread length')
    bar_plot(q5_interface1, q5_interface2, "Sentiment mean", ['Compound', 'Positive', 'Neutral', 'Negative'], "Happier interface")
    bar_plot(q4_interface1, q4_interface2, "Sentiment mean", ['Compound_self', 'Compound_other', 'Positive_self', 'Positive_other',
                                                    'Neutral_self', 'Neutral_other', 'Negative_self',
                                                    'Negative_other'], "Emotional Synchrony")
    bar_plot([ch_ans1], [ch_ans2], "Mean correct answers over all conversations", "", "Predicting correct answers")


if __name__ == '__main__':
    main()
