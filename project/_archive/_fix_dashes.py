import pandas as pd, glob

files = glob.glob(r'C:\Users\Inann\Desktop\**\previsions_2025*.xlsx', recursive=True)
for f in files:
    df = pd.read_excel(f)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.replace('\u2014', ' - ', regex=False).str.replace('\u2013', ' - ', regex=False)
    df.to_excel(f, index=False)
    print('Fixed:', f)
