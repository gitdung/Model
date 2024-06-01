import sys
# ae seƯt lại thư mục gốc của project
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import pandas as pd
sys.path.insert(1, 'D:/Workspace/Project_VNNIC')
from main.featureURL.lexical_feature import *
file_path = "Data/black_list.xlsx"
df = pd.read_excel(file_path)
urls = df.iloc[:, 1]
array_blacklist_x = []
array_blacklist_y = []
i = 0
unique_domain = set()
# length = set()
for url in urls:
    domain = urlparse(url).netloc
    domain = domain.replace("*.", "")
    a = domain.split(".")
    unique_domain.add(a[0])
for domain in unique_domain:
    lexical_feature = LexicalURLFeature(domain)
    entropy = round(lexical_feature.url_string_entropy(), 2)
    length = lexical_feature.url_length()
    if entropy > 3.6:
        i += 1

        # length.add(length)
    array_blacklist_x.append(domain)
    array_blacklist_y.append(length)
# print(length)
print(f"Có tổng cộng {i} URL xấu trong tổng số {len(unique_domain)} link trong blacklist CLD")
# x = np.array(array_blacklist_x)
# y = np.array(array_blacklist_y)
# sorted_indices = np.argsort(y)
# sorted_y = y[sorted_indices]
# sorted_x = x[sorted_indices]
# plt.axhline(y=20, color='r', linestyle='--', label='y=20')
# plt.title("Black list length")
# plt.bar(sorted_x, sorted_y)
# plt.show()
