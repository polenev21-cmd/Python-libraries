import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl



#grid=plt.GridSpec(1, 2)
#
#ax1=plt.subplot(grid[0, 0])
#ax1.set_xscale("log")
#ax1.set_xlim(1, 1000)
#ax1.grid(True, which="major")
#
#
#ax2=plt.subplot(grid[0, 1])
#ax2.set_yscale("log")
#ax2.set(ylim=(1, 1000))
#ax2.grid(True, which="major", axis="y")
#
#print(ax1.xaxis.get_major_locator())
#print(ax1.xaxis.get_major_formatter())
#print(ax1.xaxis.get_minor_locator())
#print(ax1.xaxis.get_minor_formatter())
#
#print(ax1.yaxis.get_major_locator())
#print(ax1.yaxis.get_major_formatter())
#print(ax1.yaxis.get_minor_locator())
#print(ax1.yaxis.get_minor_formatter())
#
#
#ax1.xaxis.set_major_formatter(plt.NullFormatter())
#ax2.xaxis.set_major_locator(plt.NullLocator())


#from sklearn.datasets import fetch_olivetti_faces
#
#faces=fetch_olivetti_faces().images
#
#fig, ax=plt.subplots(7, 7)
#fig.subplots_adjust(hspace=0, wspace=0)
#
#
#for i in range(7):
#    for j in range (7):
#        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
#        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
#        ax[i, j].imshow(faces[7*i+j], cmap="magma")

def ff(value, tick_number):
    N=int(np.round(2*value/np.pi))
    if N==0:
        return 0
    elif N==1:
        return r"$\frac{\pi}{2}$"
    elif N==2:
        return r"$\pi$"
    elif N%2==0:
        t=int(N/2)
        return f"{t}"+r"$\pi$"
    else:
        return f"{N}"+r"$\frac{\pi}{2}$"
    return value


#x=np.linspace(0, 4*np.pi, 1000)
#
#fig, ax=plt.subplots()
#
#ax.plot(x, np.sin(x), label="Sinus")
#ax.plot(x, np.cos(x), label="Cosinus")
#
#ax.legend()
#ax.set_xlim(0, 4*np.pi)
#
#ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
#ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi/4))
#
#ax.xaxis.set_major_formatter(plt.FuncFormatter(ff))

#ig, ax=plt.subplots(8, 1)
#
#=np.linspace(0, 10, 20)
#
#or i in ax.flat:
#   i.plot(x, x*0+2)
#
#x[0].xaxis.set_major_locator(plt.NullLocator())
#x[1].xaxis.set_major_locator(plt.MultipleLocator(0.8))
#x[2].xaxis.set_major_locator(plt.FixedLocator([1, 3, 8, 9]))
#x[3].xaxis.set_major_locator(plt.LinearLocator(numticks=4))
#x[4].xaxis.set_major_locator(plt.IndexLocator(base=1.5, offset=0.3))
#x[5].xaxis.set_major_locator(plt.AutoLocator())
#x[6].xaxis.set_major_locator(plt.MaxNLocator(8))
#x[7].xaxis.set_major_locator(plt.LogLocator(base=3))
#
#mport matplotlib.ticker as mtick
#
#x[1].xaxis.set_major_formatter(plt.NullFormatter())
#x[2].xaxis.set_major_formatter(plt.FixedFormatter(['a', 'b', 'c', "d"]))
#x[3].xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f $m^2$"))
#x[4].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=4))

#plt.style.use("lec_13.style")
#print(plt.style.available)
#exit()
#plt.style.use("grayscale")

#x=np.random.randn(1000)


#plt.axes(facecolor="#adadad")
#plt.figure(facecolor="#921212")


#plt.rc("figure", facecolor="#921212")
#plt.rc("axes", facecolor="#adadad")
#
#plt.hist(x)

from mpl_toolkits import mplot3d

def f(x, y):
    return np.sin(np.sqrt(x**2+y**2))

x=np.linspace(-6, 6, 30)
y=np.linspace(-10, 10, 50)

print(x.shape)
print(y.shape)


X, Y=np.meshgrid(x, y)

print(X.shape)
print(Y.shape)

print(X)
print(Y)

Z=f(x, y)

print(Z.shape)
print(Z)


fig=plt.figure()
ax=plt.axes(projection="3d")
ax.contour3D(X, Y, Z, 100)
ax.set_xlabel("x")
ax.set_xlabel("y")
ax.set_xlabel("z")

ax.viev_init(0, 90,)


plt.show()


#ig=plt.figure()
#x=plt.axes(projection="3d")
#
#
#
#
#lt.savefig("plt.png")