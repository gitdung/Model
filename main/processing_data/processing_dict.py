import pandas as pd

file = "dict_word.xlsx"
df = pd.read_excel(file)
column1_values = df['word'].tolist()
column2_values = df['type'].tolist()
dict_word = dict(zip(column1_values, column2_values))

# Create an empty list to store types
types = []
unique_word = set()
# Iterate over each word in the DataFrame
for i, word_i in enumerate(df['word']):
    found_substring = False
    # Iterate over each word again to compare with other words
    a = ''
    for j, word_j in enumerate(df['word']):
        if i != j and str(word_i) in str(word_j) and dict_word[word_i] == dict_word[word_j]:
            # If the word is a substring of another word and types are similar, mark it as found and break the loop
            found_substring = True
            print(dict_word[word_i])
            break
    if found_substring:
        # If a substring is found and types are similar, add it to the set
        unique_word.add(word_i)
    else:
        unique_word.add(a)

# Iterate over unique words and get their types from dict_word
for word in unique_word:
    word_type = dict_word.get(word, "0")
    types.append(word_type)

# Add the types list as a new column to the DataFrame
df1 = pd.DataFrame({
    "word": list(unique_word),
    "type": types
})
csv_file = 'dict_check_type1.xlsx'
df1.to_excel(csv_file, index=False)
print(f'Data has been written to {csv_file}')
