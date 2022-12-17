import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import requests
import sys
import optparse
import string
from bs4 import BeautifulSoup as bs # For webscraping 
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import spacy 
import csv
from pathlib import Path
from stop_words import get_stop_words

nltk_stopwords_english = list(stopwords.words('english'))
stopwords_english = list(get_stop_words('en'))
stopwords_english.extend(nltk_stopwords_english)

def fetchTextURL(url):
    paragraph_list = []
    #sentences = []
    html = requests.get(url)
    html_content = html.content 
    soup_html = bs(html_content, 'html.parser')
    paragraph = " "
    ptext= soup_html.find_all('p', _class=None)
    # Process text line by line.
    for paragraph in ptext:
        paragraph = paragraph.get_text()
        paragraph = paragraph.lower()
        paragraph = paragraph.replace("Read more:", "") # Useless phrase for model
        paragraph = paragraph.replace("iii", "") # replace edge cases of names such as John Doe III
        paragraph = paragraph.replace("ii", "") # replace edge cases of names such as John Doe II
        paragraph = paragraph.replace("\xa0", " ")
        # replace hyphenated words with spaces to dileniate words
        paragraph = paragraph.replace("—", " ")
        paragraph = paragraph.replace("–", " ")
        paragraph = paragraph.replace("-", " ")
        paragraph = paragraph.replace("-", " ") 
        paragraph = re.sub(r"[^\w\s]", "", paragraph) # Remove punctuation
        paragraph = re.sub(r"[0-9]+", "", paragraph) # Remove digits
        paragraph = re.sub(r'\s+', ' ', paragraph) # remove new-line characters
        paragraph = re.sub(r"\'", "", paragraph) # remove single quotes
        
        """
        paragraph = re.sub(r"\[\d+\]", "", paragraph)
        paragraph = re.sub(r"\[\w+\]", "", paragraph)
        paragraph = re.sub(r"[0-9]+", "", paragraph)
        paragraph = paragraph.replace("*", "")
        paragraph = paragraph.replace("%", "")
        paragraph = paragraph.replace("$", "")
        paragraph = paragraph.replace("@", "")
        paragraph = paragraph.replace("(", "")
        paragraph = paragraph.replace(")", "")
        paragraph = paragraph.replace("—", " ")
        paragraph = paragraph.replace("–", " ")
        paragraph = paragraph.replace("-", " ") # replace hyphenated words with just a space between
        paragraph = paragraph.replace("\n", "")
        paragraph = paragraph.replace("-", " ")
        paragraph = paragraph.replace("\t", "")
        paragraph = paragraph.replace("“", "")
        paragraph = paragraph.replace("”", "")
        paragraph = paragraph.replace(",", "")
        paragraph = paragraph.replace("'", "")
        paragraph = paragraph.replace("’", "")
        paragraph = paragraph.replace(".", "")
        paragraph = paragraph.replace(",", "")
        paragraph = paragraph.replace("?", "")
        paragraph = paragraph.replace("!", "")
        paragraph = paragraph.replace(":", "")
        paragraph = paragraph.replace(";", "")
        paragraph = paragraph.replace("#", "")
        paragraph = paragraph.replace('"', "")
        paragraph = paragraph.replace("/", "")
        paragraph = paragraph.replace("\xa0", " ")
        """
        paragraph_list.append(paragraph)
    paragraph_list_noEmpty = [c for c in paragraph_list if c != ' ' and c != '']
    cleaned_text = [word for word in paragraph_list_noEmpty if word not in stopwords_english] #stopwords removed
    #cleaned_sentences = " ".join(word for word in paragraph if word not in paragraph_list_noEmpty)
    return cleaned_text

def readLinksCSV(linksFile):
    text_data = []
    df = pd.read_csv(linksFile)
    df = df.dropna() # remove N/A rows in dataframe
    for link in df['urls']:
        link_text = fetchTextURL(link)
        text_data += link_text
    
    return text_data

def writeToCSV(text):
    df = pd.DataFrame(text)
    df.to_csv('scraped_data.csv')
        
if __name__ == '__main__':
    optparser = optparse.OptionParser()

    (opts, _) = optparser.parse_args()
    
    scraped_text = readLinksCSV("./urls.csv")
    writeToCSV(scraped_text)
    print(f"# of sentences collected: {len(scraped_text)}")

