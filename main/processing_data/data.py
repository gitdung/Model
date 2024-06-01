from getLexicalFeature import *
import pandas as pd
import time

start_time = time.time()


def normalize_url(urls):
    urls_without_vn = [str(url).replace('.vn', '') for url in urls]
    url_cleaned = [str(url).replace('.', ' ') for url in urls_without_vn]
    return url_cleaned


path = 'main/processing_data/33k_URL_VNNIC_CLD.csv'
df = pd.read_csv(path)
urls = df.iloc[:100, 0].tolist()
print(urls)
length, entropy, percent, num_special, bao_chi, chinh_tri, co_bac, khieu_dam, tpmt, con_lai = [
    [] for _ in range(10)]
for url in urls:
    features = getLexicalInputNN(url)
    [l.append(f) for l, f in zip([length, entropy, percent, num_special, bao_chi,
                                  chinh_tri, co_bac, khieu_dam, tpmt, con_lai], features)]
df = pd.DataFrame({
    'url': normalize_url(df.iloc[:100, 0].tolist()),
    'length': length,
    'entropy': entropy,
    'percent_num': percent,
    'num_special': num_special,
    'bao_chi': bao_chi,
    'chinh_tri': chinh_tri,
    'co_bac': co_bac,
    'khieu_dam': khieu_dam,
    'tpmt': tpmt,
    'con_lai': con_lai,
    'label': 1
})


xlsx_file = 'data_train.csv'
df.to_csv(xlsx_file, index=False)
print(f'Data has been written to {xlsx_file}')

end_time = time.time()
# Calculate execution time
execution_time = end_time - start_time
print("Execution time:", execution_time/60, "seconds")
