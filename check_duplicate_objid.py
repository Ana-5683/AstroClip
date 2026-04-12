import csv
from collections import defaultdict

csv_path = r"/mnt/d/SoftWare/PycharmProjects/AstroCLIP-main/dsm/fits/DR16Q_multimodal_dataset.csv"

objid_rows = defaultdict(list)

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row_idx, row in enumerate(reader):
        objid = row['OBJID']
        objid_rows[objid].append(row_idx + 2)

duplicates = {objid: rows for objid, rows in objid_rows.items() if len(rows) > 1}

if not duplicates:
    print("没有发现重复的OBJID字段")
else:
    print(f"发现 {len(duplicates)} 个重复的OBJID字段:\n")
    
    for objid, rows in duplicates.items():
        print(f"OBJID: {objid}")
        print(f"  重复次数: {len(rows)}")
        print(f"  所在行号 (从1开始,含表头): {rows}")
        print()
    
    print(f"总计: {len(duplicates)} 个OBJID存在重复")
    print(f"涉及总行数: {sum(len(rows) for rows in duplicates.values())}")
