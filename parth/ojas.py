from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import fitz
import nltk
import sys

nltk.download('punkt')

LANGUAGE = "english"
SENTENCES_COUNT = 5

#pdf_path = r"C:\Users\ma782\Desktop\AI project\Internship-AI\parth\ARIF_RESUME.pdf"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text("text")
    return text

pdf_text = extract_text_from_pdf(pdf_path)

# Tokenize the text using nltk
sentences = nltk.sent_tokenize(pdf_text)

# Combine the sentences into a single string for summarization
text_for_summarization = " ".join(sentences)

# Change the default encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

parser = PlaintextParser.from_string(text_for_summarization, Tokenizer(LANGUAGE))

stemmer = Stemmer(LANGUAGE)
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)

# Perform summarization and print the results
for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)
