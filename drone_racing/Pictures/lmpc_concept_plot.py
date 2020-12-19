import matplotlib.pyplot as plt
import numpy as np

x0 = 0
xf = 2.5
N = 15

x = np.linspace(x0,xf,N)

ss = np.sin(x)
ssl = np.sin(xf) * x / xf

y = np.sin(x) - x*(xf-x) * 0.2

y0 = np.sin(x0)


plt.plot([x0],[y0],'o',color = 'green',markersize = 12)
plt.plot(x,ss,'o',color = 'orange')
plt.fill_between(x,ss,ssl,color = 'orange',alpha = 0.5)
plt.plot(x,y,color = 'blue')


plt.legend(('Current Position','Safe Set','Planned Trajectory','Safe Set Convex Hull'))

plt.show()
