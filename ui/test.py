import os 
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.insert(1, os.getenv('PATH_ROOT'))

from main.processing_data.getLexicalFeature import *
from main.featureURL.lexical_feature import *
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import streamlit as st


tokenizer = AutoTokenizer.from_pretrained('gechim/phobert-base-v2-finetuned')
phoBert = AutoModel.from_pretrained("gechim/phobert-base-v2-finetuned")

# mang no ron


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.phoBert = phoBert  # (batchsize , 1 , 768)
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 768)  # (batchsize , 1 , 768)
        self.dropout_nn = nn.Dropout(0.1)
        self.dropout_lm = nn.Dropout(0.1)

        # self.out = nn.Linear(768, num_classes)
        self.out = nn.Linear(1536, num_classes)

    def forward(self, features, input_ids, token_type_ids, attention_mask, labels):
        # output bên sang
        x_nn = F.relu(self.fc1(features))
        x_nn = F.relu(self.fc2(x_nn))

        # output bên bảo
        x_phoBert = self.phoBert(input_ids=input_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask).last_hidden_state[:, 0, :]

        # drop out trước khi concat
        x_nn = self.dropout_nn(x_nn)
        x_phoBert = self.dropout_lm(x_phoBert)

        # print(x_phoBert.shape)
        logits = self.out(torch.cat((x_nn, x_phoBert), dim=1)
                          )  # self.out( x_nn + x_phoBert)

        # tính loss cái này chỉ để hiện kq loss tập valid
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        # hàm trainer cần cái này nó mới chịu train
        return SequenceClassifierOutput(loss=loss, logits=logits)


def preprocess_url(url):
    url = str(url)
    if url.startswith("http://"):
        url = url[7:]
        url = url.replace("www.", "")
    if url.startswith("https://"):
        url = url[8:]
        url = url.replace("www.", "")
        url = url.replace(".", " ")
    url = url.replace(".", " ")
    url = url.replace("/", "")
    return url


def normalize_url(url):
    url = str(url)
    if url.startswith("http://"):
        url = url[7:]
        url = url.replace("www.", "")
    if url.startswith("https://"):
        url = url[8:]
        url = url.replace("www.", "")
        url = url.replace(".", " ")
    url = url.replace(".", " ")
    url = url.replace("/", "")
    url = url.replace("edu vn", "")
    url = url.replace("com vn", "")
    url = url.replace("net vn", "")
    url = url.replace("org vn", "")
    url = url.replace("gov vn", "")
    url = url.replace("vn", "")
    return url


model = torch.load('D:\Workspace\Project_VNNIC\models\model_concat_dataV2.pt', map_location=torch.device('cpu'))

def detect_toxic_website(url):
    url = normalize_url(url)
    url_tokenize = tokenizer(url, return_tensors='pt')
    x_feature = torch.tensor([getLexicalInputNN(url)], dtype=torch.float32)
    y = model(features=x_feature, input_ids=url_tokenize['input_ids'], token_type_ids=url_tokenize['token_type_ids'],
              attention_mask=url_tokenize['attention_mask'], labels=torch.tensor([1])).logits
    if torch.argmax(y).item() == 0:
        return "Bình thường"
    if torch.argmax(y).item() == 1:
        return "Có tín nhiệm thấp"


def main():
    st.sidebar.title('Website Detector')
    app_mode = st.sidebar.selectbox('Menu', ['Website Detector'])
    if (app_mode == "Website Detector"):
        st.subheader('Input Single URL')
        url_input = st.text_input('Enter URL:')
        lexical = LexicalURLFeature(url_input)
        if st.button('Detect url'):
            results = []
            lengths = []
            entropys = []
            num_percents = []
            number_of_special_characters = []
           
            result = detect_toxic_website(url_input)
            lexical = LexicalURLFeature(url_input)
            lengths.append(lexical.get_length_url())
            entropys.append(lexical.get_entropy())
            num_percents.append(lexical.get_percentage_digits())
            number_of_special_characters.append(lexical.get_count_special_characters())
            results.append(result)
            st.write('Results:')
            data = {
                'URL': [url_input],
                'Entropy': entropys,
                'Phần trăm chữ số': num_percents,
                'Độ dài domain': lengths,
                'Số tự đặt biệt': number_of_special_characters,
                'Result': results
            }
            st.table(data)
        st.subheader('Input URLs from Excel File')
        excel_file = st.file_uploader('Upload Excel file', type=['xlsx'])
        if excel_file is not None:
            df = pd.read_excel(excel_file)
            if st.button('Detect file'):
                results = []
                lengths = []
                entropys = []
                num_percents = []
                number_of_special_characters = []
                for row in df.iterrows():
                    result = detect_toxic_website(row[1]['url'])
                    lexical = LexicalURLFeature(row[1]['url'])
                    lengths.append(lexical.get_length_url())
                    entropys.append(lexical.get_entropy())
                    num_percents.append(lexical.get_percentage_digits())
                    number_of_special_characters.append(
                        lexical.get_count_special_characters())
                    results.append(result)
                df['Entropy'] = entropys
                df['Phần trăm chữ số'] = num_percents
                df['Độ dài domain'] = lengths
                df['Số ký tự đặt biệt'] = number_of_special_characters
                df['Result'] = results

                st.write('Results:')
                st.write(df[['url', 'Entropy', 'Phần trăm chữ số',
                         'Độ dài domain', 'Số ký tự đặt biệt', 'Result']])


if __name__ == '__main__':
    main()
