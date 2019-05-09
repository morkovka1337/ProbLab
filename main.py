
# -*- coding: utf-8 -*-
import sys
import collections
from numpy import random
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWidgets, QtGui, QtCore
from ui_for_prob_lab import *
import numpy as np
import math
import matplotlib.pyplot as plt

def numInegrateFunc(x, r):
    return x ** (r/2) / (2 ** (r/2) * math.gamma(r/2) * x * math.exp(x/2)) if x > 0 else 0
    

def distrFunc(n, q):
    return (1-q**n)
def selDistrFunc(collection, x, n):
    res = 0
    for k, val in collection:
        if k < x:
            res += val
        else: break
    return res/n
def theoreticProb(n, p):
    return p**n*(1-p)

class MyWin(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None, *args, **kwargs):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.MyFunction)
    def MyFunction(self):
        
        # def selDistrFunc(self, x, y, m):
        #     return 

        p = float(self.textEdit.toPlainText())
        n = int(self.spinBox.value())
        answers = dict()
        for i in range(n):
            numAnswers = 0
            while random.uniform(0, 1) < p:
                numAnswers += 1
            if numAnswers in answers:
                answers[numAnswers] += 1
            else:
                answers[numAnswers] = 1
        i = 0
        odAnswers = collections.OrderedDict(sorted(answers.items()))
        self.tableWidget.setColumnCount(odAnswers.__len__())
        for key in odAnswers.keys():
            self.tableWidget.setItem(0, i, QtWidgets.QTableWidgetItem(str(key)))
            self.tableWidget.setItem(1, i, QtWidgets.QTableWidgetItem(str(odAnswers[key])))
            self.tableWidget.setItem(2, i, QtWidgets.QTableWidgetItem(str(odAnswers[key]/n)))
            i+= 1
        y = list(odAnswers.keys())
        E_ksi = p/(1-p)
        x_ksi = 0
        for k, val in odAnswers.items():
            x_ksi += k*val
        x_ksi /= n
        D_ksi = p/((1-p)**2)
        s_ksi = 0
        for k, val in odAnswers.items():
            s_ksi += ((k - x_ksi)**2)*val
        s_ksi /= n
        R = y[-1] - y[0]
        x = []
        for key, val in odAnswers.items():
            for i in range(val):
                x.append(key)
        Me = x[x.__len__()//2] if x.__len__() % 2 == 1 else (
            x[x.__len__()//2-1] + x[x.__len__()//2])/2
        self.tableWidget_2.setRowCount(1)
        self.tableWidget_2.setItem(0, 0, QtWidgets.QTableWidgetItem(str(E_ksi)))
        self.tableWidget_2.setItem(
        0, 1, QtWidgets.QTableWidgetItem(str(x_ksi)))
        self.tableWidget_2.setItem(
        0, 2, QtWidgets.QTableWidgetItem(str(abs(E_ksi - x_ksi))))
        self.tableWidget_2.setItem(0, 3, QtWidgets.QTableWidgetItem(str(D_ksi)))
        self.tableWidget_2.setItem(
                                       0, 4, QtWidgets.QTableWidgetItem(str(s_ksi)))
        self.tableWidget_2.setItem(
                                       0, 5, QtWidgets.QTableWidgetItem(str(abs(D_ksi - s_ksi))))
        self.tableWidget_2.setItem(0, 6, QtWidgets.QTableWidgetItem(str(Me)))
        self.tableWidget_2.setItem(0, 7, QtWidgets.QTableWidgetItem(str(R)))
        
        plt.subplot(111)
        
        if np.max(y) < math.ceil(math.log(0.0001, p)):
            theorDistrFunc = [distrFunc(i, p) for i in range(math.ceil(math.log(0.001, p)) + 1)]
            plt.step(list(range(math.ceil(math.log(0.001, p)) + 1)), theorDistrFunc)
            expDistrFunc =  []
            for i in range(math.ceil(math.log(0.001, p))):
                value = 0
                for j in odAnswers.keys():
                    if j < i:
                        value += odAnswers[j]
                    else:
                        break
                expDistrFunc.append(value/n)
            expDistrFunc.append(1)
            
            plt.step(list(range(math.ceil(math.log(0.001, p)) + 1)), expDistrFunc)
        else:
            theorDistrFunc = [distrFunc(i, p) for i in range(np.max(y)+1)]
            plt.step(list(range(np.max(y)+1)), theorDistrFunc)
            expDistrFunc =  []
            for i in range(np.max(y)+1):
                value = 0
                for j in odAnswers.keys():
                    if j < i:
                        value += odAnswers[j]
                    else:
                        break
                expDistrFunc.append(value/n)
            plt.step(list(range(np.max(y)+1)), expDistrFunc)
        
        
        probs = np.array([theoreticProb(x, p) for x in range(np.max(y)+1)])
        i = 0
        expProbs = []
        for x in range(np.max(y)+1):
            if x in odAnswers.keys():
                expProbs.append(odAnswers[x]/n)
            else:
                expProbs.append(0)
        self.tableWidget_3.setColumnCount(odAnswers.__len__())
        i = 0
        prob = 0
        for key in odAnswers.keys():
            prob = odAnswers[key]/n
            self.tableWidget_3.setItem(2, i, QtWidgets.QTableWidgetItem(str(prob)))
            self.tableWidget_3.setItem(0, i, QtWidgets.QTableWidgetItem(str(key)))
            self.tableWidget_3.setItem(1, i, QtWidgets.QTableWidgetItem(str(probs[i])))
            i+= 1
        err = max(abs(probs - expProbs))
        self.label_3.setText(QtCore.QCoreApplication.translate("MainWindow", 
            "Максимальное отклонение \nвероятности:" + str(err) + 
                "\nРасхождение графиков: " + str(max(abs(np.array(theorDistrFunc)- np.array(expDistrFunc))))))
        plt.legend(("Истинная функция распределения", "Выборочная функция распределения"))
        plt.show()
        kIntervals = int(input("Введите число интервалов: \n"))
        zEdges = []
        for i in range(kIntervals-1):
            zEdges.append(float(input("Введите границу интервала " + str(len(zEdges)+1) + "\n")))
        print("Отображение интервалов и соответствующих им теоретических вероятностей")
        print("-inf", zEdges[0], sep = '...', end = '|\t')
        for i in range(len(zEdges)-1):
            print(zEdges[i], zEdges[i+1], sep = '...', end = '|\t')
        print(zEdges[-1], '+inf', sep = '...', end = '|\t')
        sumProb = 0
        j = 0
        while j < (zEdges[0]):
            sumProb += theoreticProb(j, p)
            j+= 1
        print()
        qProbs = []
        qProbs.append(sumProb)
        for i in range(0, len(zEdges)-1):
            sumProb= 0
            j = math.ceil(zEdges[i])
            while j < math.ceil(zEdges[i+1]):
                sumProb += theoreticProb(j, p)
                j += 1
            qProbs.append(sumProb)
        sumProb = 0
        for i in range(math.ceil(zEdges[-1]), 1000):
            sumProb += theoreticProb(i, p)
        qProbs.append(sumProb)
        for prob in qProbs:
            print('{:06.5f}'.format(prob), end = '\t')
        print()
        alpha = float(input("Введите уровнь значимости альфа \n"))
        R = 0
        for i in range(len(zEdges)-1):
            nj = 0
            j = zEdges[i]
            while j < (zEdges[i + 1]):
                try:
                    nj += odAnswers[j]
                except KeyError:
                    nj += 0
                j+= 1
            R += (nj - n * qProbs[i]) ** 2 / (n * qProbs[i])


        nInt = int(R * 10000)
        intChiSquare = 0
        for i in range(1, nInt+1):
            intChiSquare += ((numInegrateFunc(R * (i - 1) / nInt, kIntervals - 1)) + 
                (numInegrateFunc(R * i / nInt, kIntervals - 1))) * R / (2*nInt)
        print("F(R0) = ", intChiSquare)
        if (1 - intChiSquare) < alpha:
            print("Гипотеза принята")
        else:
            print("Гипотеза не принята")




            
if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass
