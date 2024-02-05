from SLINK import SLINK
import numpy as np
import pickle
import time
import math

UITE_NAME = "printtoken2"
MAT_PATH = '../spec_matrix/v'
CLS_PATH = '../spec_classif/v'
CASE_NUM = 4115     # total counted to 4115 cases
VERSION_NUM = 10    # total counted to 10 versions


def f1(x):
    return math.log(x+1,2)

def f2(x):
    return 2*x -(x ** 2)

def f3(x):
    return x**0.5

def main():

    F = [f1,f2,f3]
    nF = len(F) + 1

    total_coinc = 0
    total_positive = [0]*nF
    total_TP = [0]*nF
    total_FP = [0]*nF
    total_TN = [0]*nF
    total_FN = [0]*nF

    for i in range(nF):
        with open('result_ratio' + str(i) + '.txt', 'w') as file:
            file.write('V' + '\t' + 'P' + '\t' + 'R' + '\t' + 'F' + '\n')
        with open('result' + str(i) + '.txt', 'w') as file:
            file.write('V' + '\t' 'TP' + '\t' + 'FP'  + '\t' + 'TN'  + '\t' + 'FN' + '\n')

    for v in range(1, VERSION_NUM + 1):
        spectrMatrix = []
        matrixfile = open(MAT_PATH + str(v) , 'r')
        for line in matrixfile:
            line = [int(i) for i in line.strip('\t\n').split('\t')]
            spectrMatrix.append(line)
        isPass = spectrMatrix[0][1:]
        spectrMatrix = np.delete(spectrMatrix, 0, 0)
        spectrMatrix = np.transpose(spectrMatrix)
        row = spectrMatrix[0]
        spectrMatrix = np.delete(spectrMatrix, 0, 0)

        with open(CLS_PATH + str(v) + '.pkl', 'rb+') as file:
            passed = pickle.load(file)
            failed = pickle.load(file)
            coinc = pickle.load(file)

        total_coinc += len(coinc)
        c = len(failed) / (len(passed) + len(coinc))

        hc = SLINK(spectrMatrix)

        isStop = False
        det_coinc = [set()] + [set() for i in F]

        while not isStop:
            hc.clustering(times = 1)
            isStop = True
            for cluster in hc.clusters:
                elements = cluster.elements
                fail_elem = elements & set(failed)
                prop = len(fail_elem) / cluster.size
                if prop > c:
                    coinc_elem = elements - fail_elem
                    det_coinc[0] |= coinc_elem
                    isStop = False
                    for i in range(nF-1):
                        if prop > F[i](c):
                            det_coinc[i+1] |= coinc_elem

        for i in range(nF):
            positive = len(det_coinc[i])
            total_positive[i] += positive
            if not positive and not len(coinc):
                TP, FP, TN, FN = 0, 0, len(passed), 0
                precision, recall, F1 = 1.0, 1.0, 1.0
            elif not positive:
                TP, FP, TN, FN = 0, 0, len(passed), len(coinc)
                precision, recall, F1 = 'null', 0, 0
            elif not len(coinc):
                TP, FP, TN, FN = 0, positive, len(passed)-positive, 0
                precision, recall, F1 = 0, 'null', 0
            else:

                TP = len(det_coinc[i] & set(coinc))
                FP = positive - TP
                TN = len(passed) - FP
                FN = len(coinc) - TP

                total_TP[i] += TP
                total_FP[i] += FP
                total_TN[i] += TN
                total_FN[i] += FN

                precision = round(TP / positive,3)
                recall = round(TP / len(coinc),3)
                F1 = round((2*TP / (positive+len(coinc))),3)

            with open('result'+ str(i) + '.txt', 'a') as file:
                file.write(str(v) + '\t' + str(TP) + '\t' + str(FP) + '\t' + str(TN) + '\t' + str(FN)+ '\n')

            with open('result_ratio'+ str(i) + '.txt', 'a') as file:
                file.write(str(v) + '\t' + str(precision) + '\t' + str(recall) + '\t' + str(F1) + '\n')


        print('v' + str(v) + ' is finished')

    for i in range(nF):
        precision = round(total_TP[i] / total_positive[i], 3)
        recall = round(total_TP[i] / total_coinc, 3)
        F1 = round(2*total_TP[i] / (total_positive[i]+total_coinc), 3)

        with open('result'+ str(i) + '.txt', 'a') as file:
            file.write('Tot' + '\t' + str(total_TP[i]) + '\t' + str(total_FP[i]) + '\t' + str(total_TN[i]) + '\t' + str(total_FN[i])+ '\n')
        with open('result_ratio'+ str(i) + '.txt', 'a') as file:
            file.write('Tot' + '\t' + str(precision) + '\t' + str(recall) + '\t' + str(F1) + '\n')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('time: ', end - start)
