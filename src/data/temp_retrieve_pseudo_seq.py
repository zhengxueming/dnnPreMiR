import pandas as pd

df = pd.read_csv("pseudo.csv")

for index, row in df.iterrows():
    print (">{}".format(row["Accession"]))
    print (row["HairpinSequence"])
