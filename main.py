
import pandas as pd

pd.set_option("display.float_format",lambda x:"%.2f" %x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

ds=pd.read_csv("advertising.csv")
#print(ds.shape)   # satir,sutun  (200, 4)

x=ds[["TV"]]
y=ds[["sales"]]

reg_model=LinearRegression().fit(x,y)         # y = a0 + a1*x

a0 = reg_model.intercept_[0]  # a0 sabitini buldu
a1 = reg_model.coef_[0][0]    # a1'i bulduk

y_pred = reg_model.predict(x)
mse = mean_squared_error(y,y_pred)    # mse = ortalama hata
y_ort=y.mean()                        # y'nin ortalaması
ssapma=y.std()                        # standart sapma

xi=ds[["TV"]].sample()
yi=a0+a1*xi

print(yi)
