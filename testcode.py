import pandas as pd

df = pd.DataFrame([[10, 11, 12], [100, 110, 120]])


for index, row in df.iterrows():
    if row.__contains__(10):
        print("oh no")