# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


LANGUAGE = "english"


def url(url,SENTENCES_COUNT):
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)

def txxt(txxt,SENTENCES_COUNT):
    parser = PlaintextParser.from_file(txxt, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)

def letters(letters,SENTENCES_COUNT):
    parser = PlaintextParser.from_string(letters, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)


letters("Cristiano Ronaldo, often hailed as one of the greatest footballers of all time, is a Portuguese icon whose remarkable career has left an indelible mark on the sport. Born on February 5, 1985, in Funchal, Madeira, Ronaldo began his professional journey with Sporting Lisbon before catching the eye of Manchester United in 2003. His six seasons with the Red Devils were transformative, yielding three Premier League titles and a UEFA Champions League triumph in 2008. Ronaldo's unparalleled athleticism, prolific goal-scoring prowess, and incredible work ethic earned him the first of his five Ballon d'Or awards in 2008. His subsequent move to Real Madrid in 2009 saw him achieve unprecedented success, clinching four Champions League titles and becoming Real Madrid's all-time leading scorer. In 2018, Ronaldo embarked on a new chapter with Juventus in Serie A, securing two league titles. The prodigious forward returned to Manchester United in 2021, igniting a wave of enthusiasm among fans. Off the pitch, Ronaldo's philanthropy and dedication to charitable causes underscore his commitment to making a positive impact beyond football. As he continues to defy age with his goal-scoring exploits, Cristiano Ronaldo's legacy is firmly cemented in the annals of football history, leaving an enduring inspiration for aspiring athletes worldwide.",5)

url("https://www.mirror.co.uk/sport/football/news/manutd-transfer-news-live-todibo-31713310",5)