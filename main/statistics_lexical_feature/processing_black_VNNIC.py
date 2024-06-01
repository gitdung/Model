import sys
# ae seƯt lại thư mục gốc của project
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import pandas as pd
sys.path.insert(1, 'D:/Workspace/Project_VNNIC')
from main.featureURL.lexical_feature import *
file_path = "Data/black_VNNIC.xlsx"
df = pd.read_excel(file_path, sheet_name=0)
df = df.dropna(subset=['Tên miền'])
urls = df.iloc[:, 1]
array_blacklist_x = []
array_blacklist_y = []
i = 0
unique_domain = set()
for url in df['Tên miền']:
    a = url.split(".")
    unique_domain.add(a[0])
for domain in unique_domain:
    lexical_feature = LexicalURLFeature(domain)
    entropy = round(lexical_feature.url_string_entropy(), 2)
    length = lexical_feature.url_length()
    if entropy > 3.6:
        i += 1
        print(domain)
        array_blacklist_x.append(domain)
        array_blacklist_y.append(entropy)
print(f"Có tổng cộng {i} URL xấu trong tổng số {len(unique_domain)} link trong blacklist VNNIC")
