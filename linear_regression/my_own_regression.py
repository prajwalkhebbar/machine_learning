from statistics import mean
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def create_dataset(how_much,variance,step = 2,correlation = False):
    val=1
    ys=[]
    for _ in range(how_much):
        y=val+random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs=[i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)

xs,ys=create_dataset(40,10,2,correlation='pos')
# xs=np.array([i for i in range(1,6)], dtype=np.float64)
# ys=np.array([5,4,6,5,6], dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m=(((mean(xs)*mean(ys))-(mean(xs*ys)))/
    ((mean(xs)**2)-(mean(xs*xs))))

    b=mean(ys) - m*mean(xs)
    return m,b

def squared_error(ys_line,ys_orig):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_line,ys_orig):
    y_mean_line=[mean(ys_orig) for y in ys_orig]
    squared_error_reg=squared_error(ys_line,ys_orig)
    squared_error_mean_y_line=squared_error(y_mean_line,ys_orig)
    return 1-(squared_error_reg/squared_error_mean_y_line)
m,b=best_fit_slope_and_intercept(xs,ys)
# print(m,b)
#constructing the line
regression_line=[(m*x)+b for x in xs]

#predicting
predict_x=7
predict_y=m*predict_x+b
# print(predict_y)

r2=coefficient_of_determination(regression_line,ys)
print(r2)
#ploting
plt.scatter(xs,ys,color='#003F45',label='data')
plt.scatter(predict_x,predict_y,color="g",label='prediction')
plt.plot(xs,regression_line,label='regression_line')
plt.legend(loc=4)
plt.show()
