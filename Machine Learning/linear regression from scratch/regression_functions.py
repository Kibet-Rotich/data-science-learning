import pandas as pd
import numpy as np

def lossfunction(m,b,points):

    loss = 0

    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y

        loss += (y - ((m*x) + b))**2

    return loss/len(points)


def gradient_descent(mnow,bnow,points,L):
    mgradient = 0
    bgradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        mgradient += (-2/n) * (y-((mnow*x)+bnow))*x
        bgradient += (-2/n) * (y-((mnow*x)+bnow))

    m = mnow - L*mgradient
    b = bnow - L*bgradient

    return m,b