import numpy as np
import pandas as pd
#print(pd.date_range("2026-01-01", periods=4, freq="ME"))
#
#print(pd.date_range("2026-01-01", periods=4, freq="MS"))
#
#print(pd.date_range("2026-01-01", periods=4, freq="QE"))
#
#print(pd.date_range("2026-01-01", periods=4, freq="QS"))
#
#print(pd.date_range("2026-01-01", periods=4, freq="W"))
#
#print(pd.date_range("2026-01-01", periods=4, freq="W-MON"))
#
#print(pd.date_range("2026-01-01", periods=4, freq="4W-MON"))
#
#ind=pd.read_csv("/mnt/c/Users/repository/lec_09/base1.csv", sep=";",encoding='windows-1251')
#
#print(ind.head())
#
#print(type(ind))
#print(ind.dtypes)
#
#index=pd.DatetimeIndex(ind["Date"])
#
#ind.index=index
#print(ind.head())
#
#import matplotlib.pyplot as plt
#
#ind.plot()
#plt.savefig('plot.png')

df=pd.read_csv(
    "/mnt/c/Users/repository/lec_09/FremontBridge.csv",
    index_col="Date",
    parse_dates=True,
    date_format="%m/%d/%Y %I:%M:%S %p"
    )
print(df.head())
print(df.dtypes)

print(df.columns)
df.columns=["Total", "East", "West"]
print(df.head())

print(df.describe())

import matplotlib.pyplot as plt

weekly=df.resample("W").sum()
weekly.plot(style=["-",":"])
plt.ylabel("Количество велосипидистов")
plt.savefig('bicycle1.png')

daily=df.resample("D").sum()
# center = False -> прошлые значения от выбранного
# center = True -> прошлые и будущие значения от выбранного
daily.rolling (30, center=True)
daily.plot(style=["-", ":", "--"])
plt.ylabel("Количество велосипедистов (по неделям)")
plt.savefig('bicycle2.png')

daily=df.resample("D").sum()
daily.rolling (30, center=True).mean().plot(style=["-", ":", "--"])
plt.ylabel("Среднемесячное количество велосипедистов")
plt.savefig('bicycle3.png')

timely=df.groupby(df.index.time).mean()
ticks=60*60*4*np.arange(6)
timely.plot(xticks=ticks)
plt.savefig("timely.png")

weekly=df.groupby(df.index.dayofweek).mean()
weekly.plot()
plt.savefig("weekly.png")

timely=df.groupby(df.index.time).mean()
ticks=60*60*4*np.arange(6)
timely.plot(xticks=ticks)
plt.savefig("timely.png")

w1=np.where(df.index.weekday<5, "Будни", "Выходные")
t1=df.groupby([w1, df.index.time]).mean()

fig,ax=plt.subplots(1, 2)
ax[0].set_ylim(0,600)
t1.loc["Будни"].plot(ax=ax[0],title="Будни")
ax[1].set_ylim(0,600)
t1.loc["Выходные"].plot(ax=ax [1], title="Выходные")
plt.savefig("compare.png")

# matplotlib
# pyqt5

plt.style.use("classic")

#plt.show() - один раз

#fig.savefig("fig.png")

from IPython.display import Image

Image("ttt.png")

print(fig.canvas)