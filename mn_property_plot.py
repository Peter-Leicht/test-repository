import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv('MN.csv')
dn = df.dropna(axis=1, how='all')
dn_new = dn[(dn['XCoord']>0) & (dn['YCoord']>0)]

def show():
	dn_new.plot(kind='scatter',figsize=(12,6),alpha=0.6,x="XCoord",y="YCoord",c="ZipCode",colorbar=True,cmap=plt.get_cmap("jet"))
	plt.show()

show()
