from main.featureURL.lexical_feature import *
import sys
# ae seƯt lại thư mục gốc của project
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import pandas as pd
sys.path.insert(1, 'D:/Workspace/Project_VNNIC')
file_path = "Data\\white_list.xlsx"
df = pd.read_excel(file_path)
df['link'] = df['link'].fillna('')
links = []
for index, row in df.iterrows():
    text = row['link']
    # Replace 'link' with the actual column name containing the text
    if isinstance(text, str):  # Check if text is a string
        extracted_links = re.findall(r'www\.[^. ]+\.[a-z]+', text)
        # Replace 'www.' with an empty string in each extracted link
        extracted_links = [link.replace('www.', '') for link in extracted_links]
        links.extend(extracted_links)

i = 0
array_blacklist_x = []
array_blacklist_y = []
unique_domain = set()
for url in links:
    a = url.split(".")
    unique_domain.add(a[0])
for domain in unique_domain:
    lexical_feature = LexicalURLFeature(domain)
    entropy = lexical_feature.url_string_entropy()
    length = lexical_feature.url_length()
    if  entropy > 3.6:
        i += 1
    array_blacklist_x.append(domain)
    array_blacklist_y.append(entropy)
print(f"Có tổng cộng {i}  URL xấu trong tổng số {len(links)} link trong whitelist của VNNIC")
