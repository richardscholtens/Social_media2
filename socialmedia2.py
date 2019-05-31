#!/usr/bin/python3
# student: J.F.P. (Richard) Scholtens
# studentnr.: s2956586
# datum: 231/05/2019

import csv
import os
from contextlib import ExitStack
import numpy as np

def main():
    filenames = [txt_file for txt_file in os.listdir("data/s3492338")]
    with ExitStack() as stack:
        files = [stack.enter_context(open('data/s3492338/' + fname)) for fname in filenames]
        for f1, f2, f3, f4 in zip(*files):
            e1 = np.array(f1.split('¦'))
            e2 = np.array(f2.split('¦'))
            e3 = np.array(f3.split('¦'))
            e4 = np.array(f4.split('¦'))
            print(len(e1), len(e2), len(e3), len(e4))
            m = np.column_stack((e1, e2, e3[2:], e4))
            for i in range(len(m)):
                print(m[i][0], m[i][1], m[i][2], m[i][3]) # , m[i][4], m[i][5], m[i][6], m[i][7])
if __name__ == '__main__':
    main()