import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR




def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def myfunc(x):
    return


if __name__ == '__main__':

    x = []
    y = []
    i = 0

    #test = ["a","b","c"]
    #print(test)
    with open("data.txt") as obj:
        for line in obj:

            row = line.split()
            y.append(int(row[1]))
            #x.append(i)
            #i += 1;

    size = len(y)


    for i in range(size):
        x.append(i)


    #plt.loglog(x,y,label="Dataset")
    #plt.plot()
    plt.scatter(x,y,label="Dataset")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("rank")
    plt.ylabel("frequency")
    plt.title("loglog diagram of frequency of word vs rank of the word")

    reg = LinearRegression()
    x1 = np.array(x).reshape(-1,1)
    y1 = np.array(y) #.reshape(-1,1)
    reg.fit(x1,y1)
    print(reg.coef_[0])
    print(reg.intercept_)

    svr = SVR(kernel="linear")
    svr.fit(x1,y)
    y_pred2 = svr.predict(x1)

    #y_pred = reg.predict(x1)


    plt.plot(x, y_pred2, label="Fited curve", color='red')
    plt.legend()

    plt.show()

    print("end")