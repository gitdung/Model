import sys
sys.path.insert(1, 'D:\Workspace\Project_VNNIC')
from main.processing_data.getLexicalFeature import *
from main.featureURL.lexical_feature import *
import torch
import pandas as pd
import streamlit as st
from main.embeddingURL.modelXmlRoberta import *

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



def detect_toxic_website(url):
    url = normalize_url(url)
    url_tokenize = tokenizer(url, return_tensors='pt')
    y = model(input_ids=url_tokenize['input_ids'], attention_mask=url_tokenize['attention_mask']).logits
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
        model_choice = st.selectbox('Select Model', ['Model PhoBert', 'Model XML Roberta'])
        
        
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
                    result = detect_toxic_website(str(row[1]['url']))
                    lexical = LexicalURLFeature(str(row[1]['url']))
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
