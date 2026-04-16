import pandas as pd

csv_path = "dsm/csv/DR16Q_v4.csv"
df = pd.read_csv(csv_path)
print(f"样本量: {len(df)}")