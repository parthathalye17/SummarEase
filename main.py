from flask import Flask, render_template, request
import nltk
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
app.secret_key = 'arif9353'
LANGUAGE = "english"


@app.route('/', methods=['GET','POST'])
def home():
    answer1=""
    #answer2=""
    webscraped=[]
    keywords=[]
    if request.method == 'POST':
        text = request.form.get('textInput')
        print(text)
        link = request.form.get('urlInput')
        print(link)
        if text:
            answer1 = txxt(text,5)
            keywords = perform_keyword_analysis(answer1)
            for kw in keywords:
                wikipedia_url = 'https://en.wikipedia.org/wiki/'+kw
                web_scrape = scrape_wikipedia(wikipedia_url)
                if web_scrape:
                    webscraped.append(web_scrape)
        else:
            answer1 = "No text provided"
        #if link:
        #    answer2 = url(link,5)
        #else:
        #    answer2 = "No URL provided"
        #if inputkaisa == 'text':
         #   option = request.form.get('text')
          #  if option:
          #      answer = txxt(option, 5)
          #  else:
          #      answer = "Text option selected, but no text provided."
        #elif inputkaisa == 'link':
        #    option = request.form.get('link')
         #   answer = url(option,5)
        #else:
        #    option = request.form.get('file')'''
    return render_template("home.html",summarized_text=answer1,webscrape = webscraped,keyword=keywords)


def url(url,SENTENCES_COUNT):
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    ans =""
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        ans+=str(sentence)
    return ans


def txxt(letters,SENTENCES_COUNT):
    parser = PlaintextParser.from_string(letters, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarized_text = ""
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        summarized_text += str(sentence)
    return summarized_text


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)


def perform_keyword_analysis(text):
    preprocessed_text = preprocess_text(text)
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    # Get the TF-IDF scores for each feature
    tfidf_scores = tfidf_matrix.toarray()[0]
    # Create a list of tuples (word, TF-IDF score) and sort it by the score
    keyword_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
    keyword_scores = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
    # Extract the top keywords (adjust the number based on your needs)
    top_keywords = [keyword for keyword, score in keyword_scores[:7]]
    return top_keywords


def scrape_wikipedia(url):
    # Send a GET request to the URL
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find all the section headers (assumes they are in p tags)
        section_paras = soup.find_all('p')
        # Print the titles of the sections
        # for header in section_headers:
        #    print(header.text.strip())
        text= """"""
        for header in section_paras:
            #print(header.text.strip())
            text += header.text.strip()
        #for header in section_lists:
        #    print(header.text.strip())
# Split the text into lines
        lines = text.split('\n')
# Select the first 15 lines
        first_15_lines = lines[:15]
# Join the selected lines into a single string
        result = '\n'.join(first_15_lines)
        final = txxt(result,5)
        return final
    else:
        return ""

if __name__ == '__main__':
    app.run(debug=True)