import json
import pandas as pd

FEATURES = [
    "IAA","IEG","IPS","IDA","IPP","IPV","IAN",
    "DEFASAGEM","NOTA_PORT","NOTA_MAT","NOTA_ING",
    "ANO_INGRESSO","QTDE_AVAL","FASE","TURMA"
]

rows = []

with open("logs/predictions.json") as f:
    
    for line in f:
        
        record = json.loads(line)
        
        input_data = record["input"]
        
        row = {k: input_data.get(k) for k in FEATURES}
        
        rows.append(row)

df = pd.DataFrame(rows)

df.to_csv("monitoring/production.csv", index=False)

print("production.csv criado com", len(df), "linhas")