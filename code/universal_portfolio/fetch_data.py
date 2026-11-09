"""Download the two public datasets used in the paper into data/ next to this script.

Shiller's monthly SP500 series (price, dividend, CPI, long interest rate) and
daily closing prices for the SP500 constituents, 8 February 2013 to 7 February
2018. Both are public mirrors; neither is redistributed with this package.
"""
import os, urllib.request

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SOURCES = {
    "shiller.csv":
        "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
    "sp500_5yr.csv":
        "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv",
}

os.makedirs(DATA, exist_ok=True)
for name, url in SOURCES.items():
    target = os.path.join(DATA, name)
    if os.path.exists(target):
        print("have", target); continue
    print("fetching", name)
    urllib.request.urlretrieve(url, target)

# build the price relative matrix used by realpair.py and sp470.py
import pandas as pd, numpy as np
d = pd.read_csv(os.path.join(DATA, "sp500_5yr.csv"), usecols=["date", "close", "Name"])
p = d.pivot(index="date", columns="Name", values="close").sort_index().dropna(axis=1)
np.save(os.path.join(DATA, "X_sp.npy"), p.values[1:] / p.values[:-1])
np.save(os.path.join(DATA, "names_sp.npy"), p.columns.values)
print("built data/X_sp.npy", p.shape[1], "names,", p.shape[0] - 1, "periods")
