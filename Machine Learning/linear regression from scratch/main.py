import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from regression_functions import gradient_descent

df = pd.read_csv("xy dataset.csv")
m,b = 0,0
L = 0.0001
epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m,b = gradient_descent(m,b,df,L)


linex = np.arange(50)
liney = []
for i in linex:
    liney.append((m*i)+b)

print(m,b)

linedf = pd.DataFrame(linex)
linedf['y'] = liney
linedf.columns = ['x','y']
linedf.to_csv('line.csv',index=False)

plt.scatter(df.x,df.y)
plt.plot(linex,liney)
plt.show()
