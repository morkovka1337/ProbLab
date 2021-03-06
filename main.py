
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
        
        if len(y) < 15:
            theorDistrFunc = [distrFunc(i, p) for i in range(15)]
            plt.step(list(range(15)), theorDistrFunc)
            expDistrFunc =  []
            for i in range(15):
                value = 0
                for j in odAnswers.keys():
                    if j < i:
                        value += odAnswers[j]
                    else:
                        break
                expDistrFunc.append(value/n)
            
            plt.step(list(range(15)), expDistrFunc)
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
            if x in odAnswers.values():
                expProbs.append(x/n)
            else:
                expProbs.append(0)
        self.tableWidget_3.setColumnCount(odAnswers.__len__())
        for key in odAnswers.keys():
            self.tableWidget_3.setItem(0, i, QtWidgets.QTableWidgetItem(str(key)))
            self.tableWidget_3.setItem(1, i, QtWidgets.QTableWidgetItem(str(probs[i])))
            self.tableWidget_3.setItem(2, i, QtWidgets.QTableWidgetItem(str(expProbs[i])))
            i+= 1
        err = max(abs(probs - expProbs))
        self.label_3.setText(QtCore.QCoreApplication.translate("MainWindow", 
            "Максимальное отклонение \nвероятности:" + str(err)))
        plt.legend(("Истинная функция распределения", "Выборочная функция распределения"))
        plt.show()
        kIntervals = int(input("Введите число интервалов: \n"))
        zEdges = []
        for i in range(min(kIntervals, len(odAnswers))):
            zEdges.append(float(input("Введите границу интервала " + str(len(zEdges)+1) + "\n")))
        print("Отображение интервалов и соответствующих им теоретических вероятностей")
        print(0, zEdges[0], sep = '...', end = '|\t')
        for i in range(len(zEdges)-1):
            print(zEdges[i], zEdges[i+1], sep = '...', end = '|\t')
        sum = 0
        j = 0
        while j < (zEdges[0]):
            sum += theoreticProb(j, p)
            j+= 1
        print()
        qProbs = []
        qProbs.append(sum)
        for i in range(0, len(zEdges)-1):
            sum = 0
            j = math.ceil(zEdges[i])
            while j < math.ceil(zEdges[i+1]):
                sum += theoreticProb(j, p)
                j += 1
            qProbs.append(sum)
        print(qProbs, sep = '|\t')
        # R = 0
        # for j in range(len(qProbs)):
        #     R += 
        alpha = float(input("Введите уровнь значимости альфа"))


        




            
if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass
