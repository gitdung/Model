import math
import re
from urllib.parse import urlparse
from socket import gethostbyname
import pandas as pd


def levenshtein_distance_string(str1, str2):
    # Get the lengths of the input strings
    str1 = str(str1).lower()
    str2 = str(str2).lower()
    m = len(str1)
    n = len(str2)

    # Initialize two rows for dynamic programming
    prev_row = [j for j in range(n + 1)]
    curr_row = [0] * (n + 1)

    # Dynamic programming to fill the matrix
    for i in range(1, m + 1):
        # Initialize the first element of the current row
        curr_row[0] = i

        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                # Characters match, no operation needed
                curr_row[j] = prev_row[j - 1]
            else:
                # Choose the minimum cost operation
                curr_row[j] = 1 + min(
                    curr_row[j - 1],  # Insert
                    prev_row[j],	 # Remove
                    prev_row[j - 1]  # Replace
                )

        # Update the previous row with the current row
        prev_row = curr_row.copy()

    # The final element in the last row contains the Levenshtein distance
    return curr_row[n]


def get_substrings_of_length(s, length):
    substrings = []
    for start in range(len(s) - length + 1):
        substrings.append(s[start:start+length])
    return substrings


class LexicalURLFeature:
    def __init__(self, url):
        self.description = 'blah'
        self.url = url
        self.domain = self.extract_domain()
        self.url_parse = urlparse(self.url)
        self.dict_word = self.load_dict_word()

    def load_dict_word(self):
        file_path = "D:/Workspace/Project_VNNIC/main/featureURL/dict_check_type.xlsx"
        try:
            df = pd.read_excel(file_path)
            return dict(zip(df['word'].tolist(), df['type'].tolist()))
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            return {}

    def extract_domain(self):
        return self.url.split('.')[0]

    def get_entropy(self):
        probs = [self.domain.count(c) / len(self.domain)
                 for c in set(self.domain)]
        entropy = -sum(p * math.log(p) / math.log(2.0) for p in probs)
        return round(entropy, 3)

    def get_length(self):
        return len(self.domain)
    def get_length_url(self):
        return len(self.url)
    def get_type_url(self):
        for word in self.dict_word.keys():
            if str(word) in self.domain:
                return self.dict_word.get(str(word), "con_lai")
            for sub_word in get_substrings_of_length(self.domain, len(str(word))):
                if levenshtein_distance_string(sub_word, word) == 1:
                    return self.dict_word.get(str(word), "con_lai")
        return "con_lai"

    def get_percentage_digits(self):
        num_digits = sum(1 for char in self.domain if char.isdigit())
        total_chars = len(self.domain)
        if total_chars == 0:
            return 0
        return round((num_digits / total_chars) * 100, 3)

    def get_count_special_characters(self):
        regex = r'[!@#$%^&*()_+\-=\[\]{};\'\\:"|<,/<>?]'
        special_characters = re.findall(regex, self.domain)
        return len(special_characters)
