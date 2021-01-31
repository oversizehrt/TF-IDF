# TF-IDF stands for
# "Term Frequency, Inverse Document Frequency."


import math
from textblob import TextBlob


def term_frequency(word, text):
    return text.words.count(word) / len(text.words)


def n_containing(word, text_list):
    return sum(1 for text in text_list if word in text)


def idf(word, text_list):
    return math.log(len(text_list) / (1 + n_containing(word, text_list)))


def tf_idf(word, text, text_list):
    return term_frequency(word, text) * idf(word, text_list)


file_num = 1
text_list = []

while True:
    try:
        with open('./doc-res/doc_' + str(file_num) + ".txt", 'r') as doc:
            file_num += 1
            text_list.append(TextBlob(doc.read()))
    except FileNotFoundError:
        break


for i, text in enumerate(text_list):
    print("Top words in document {}".format(i + 1))
    ratings = {word: tf_idf(word, text, text_list) for word in text.words}
    sorted_words = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for word, rating in sorted_words[:4]:
        print(f"Word: {word}, TF-IDF: {round(rating, 5)}")
