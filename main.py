from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
import nltk, requests, torch, fitz, os, logging, re, time
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from flask_session import Session
from gtts import gTTS



load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
nltk.download('stopwords')
nltk.download('punkt')


app = Flask(__name__)
app.config['SECRET_KEY'] = 'arif'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
LANGUAGE = "english"


@app.route('/')
def home():
    return render_template("opening.html")


@app.route('/developers')
def developers():
    return render_template("dev.html")


def txxt(letters,SENTENCES_COUNT):
    summ = "Summarize the above sentences in " +str(SENTENCES_COUNT)+" lines"
    question = letters + summ
    response =  genai.GenerativeModel("gemini-pro").generate_content(question)
    return response.text


def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)


def perform_keyword_analysis(text):
    preprocessed_text = preprocess_text(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    keyword_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
    keyword_scores = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
    top_keywords = [keyword for keyword, score in keyword_scores[:8]]
    return top_keywords


def english_text_to_mp3(text):
    print("Generating audio...")
    language = 'en'
    speed = False
    output_path = os.path.join(app.root_path, 'static', 'output.mp3')
    tts = gTTS(text=text, lang=language, slow=speed)
    tts.save(output_path)
    return "Done"

@app.route('/get_audio', methods=['GET'])
def serve_audio():
    return send_from_directory('static','output.mp3')


@app.route('/text', methods=['POST','GET'])
def text():
    answer1=""
    keywords=[]
    if request.method == 'POST':
        text = request.form.get('input_text')
        print(text)
        number = request.form.get('number_text')
        print(number)
        if text:
            answer1 = txxt(text,int(number))
            keywords = perform_keyword_analysis(answer1)
            session['keywords'] = keywords   
        else:
            answer1 = "No text provided"
            session['keywords'] = keywords
        session['summary'] = answer1
        return redirect(url_for('final_text'))
    return render_template("text.html")


@app.route('/final_text', methods=['GET','POST'])
def final_text():
    webscraped=[]
    keywords = session['keywords']
    summary = session['summary']    
    res = english_text_to_mp3(summary)
    print(res)
    if request.method=='POST':
        key_words = request.form.get('keywordz_text')
        keyword_list = key_words.split(',')
        for kw in keyword_list:
            if kw:
                web_scrape = scrape_wikipedia(kw,summary)
                if web_scrape:
                    webscraped.append(web_scrape)
                else:
                    empty_message = f"Sorry, couldn't find much information regarding '{kw}' try using differnet word or relevant a combination of this word!"
                    webscraped.append(empty_message)
            else:
                no_keyword_message = "Enter a keyword!"
                webscraped.append(no_keyword_message)
    return render_template("final_text.html", summarized_text=summary, keyword=keywords, webscrape=webscraped)


def pdf_xyz(pdf_path,SENTENCES_COUNT):
    pdf_text = extract_text_from_pdf(pdf_path)
    summ = "Summarizve the above sentences in " +str(SENTENCES_COUNT)+" lines"
    question = pdf_text +" "+ summ
    response =  genai.GenerativeModel("gemini-pro").generate_content(question)
    return response.text

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(stream=pdf_path.read(), filetype='pdf')
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text("text")
    print("\nExtracted Text:", text)
    return text


@app.route('/pdf', methods=['GET','POST'])
def pdf():
    answer1=""
    keywords=[]
    if request.method == 'POST':
        pdff = request.files['pdf']
        print(pdff)
        number = request.form.get('number_pdf')
        if pdff:
            answer1 = pdf_xyz(pdff,int(number))
            keywords = perform_keyword_analysis(answer1)
            session['keywords'] = keywords   
        else:
            answer1 = "No PDF provided"
            session['keywords'] = keywords
        session['summary'] = answer1
        return redirect(url_for('final_pdf'))
    return render_template("pdf.html")


@app.route('/final_pdf', methods=['GET','POST'])
def final_pdf():
    webscraped=[]
    keywords = session['keywords']
    summary = session['summary']    
    res = english_text_to_mp3(summary)
    if request.method=='POST':
        key_words = request.form.get('keywordz_pdf')
        keyword_list = key_words.split(',')
        print(keyword_list)
        for kw in keyword_list:
            if kw:
                web_scrape = scrape_wikipedia(kw,summary)
                if web_scrape:
                    webscraped.append(web_scrape)
                else:
                    empty_message = f"Sorry, couldn't find much information regarding '{kw}' try using differnet word or relevant a combination of this word!"
                    webscraped.append(empty_message)
            else:
                no_keyword_message = "Enter a keyword!"
                webscraped.append(no_keyword_message)
        
    return render_template("final_pdf.html", summarized_text=summary, keyword=keywords, webscrape=webscraped)


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


def sentiment_score(review):
    tokens =  tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


def sentiment_comments(urlll):
    link = urlll
    data = []
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    driver = webdriver.Chrome(options=chrome_options)
    with driver:
        wait = WebDriverWait(driver, 15)
        driver.get(f"{link}")
        for _ in range(12):  
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(5)

        for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content-text"))):
            data.append(comment.text)
    print(data)
    import pandas as pd   
    frame= pd.DataFrame(data, columns=['comment'])
    frame = pd.DataFrame(np.array(data), columns=['review'])
    frame['sentiment'] = frame['review'].apply(lambda x: sentiment_score(x))
    average_sentiment =frame['sentiment'].mean()
    return average_sentiment


def get_video_id(video_url):

    match = re.search(r"(?<=v=)[\w-]+", video_url)
    return match.group(0) if match else None


def get_video_transcript(video_url):
    video_id = get_video_id(video_url)   
    if video_id:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            return None
        if transcript:
            video_text = '\n'.join(entry['text'] for entry in transcript)
            try:
                model_name = "facebook/bart-large-cnn"
                model = BartForConditionalGeneration.from_pretrained(model_name)
                tokenizer = BartTokenizer.from_pretrained(model_name)
            except Exception as e:
                return None
            tokenized_text = tokenizer(video_text, return_tensors="pt")
            max_tokens_per_chunk = 1000
            chunks = [
                video_text[i:i + max_tokens_per_chunk]
                for i in range(0, len(video_text), max_tokens_per_chunk)
            ]
            summaries = []
            for chunk in chunks:
                try:
                    summarized_chunk = pipeline("summarization", model=model, tokenizer=tokenizer)(chunk, max_length=500, min_length=100, length_penalty=2.0, num_beams=4)
                    summaries.append(summarized_chunk[0]['summary_text'])
                except Exception as e:
                    return None           
            final_summary = ' '.join(summaries)
            return final_summary
        else:
            return None
    else:
        return None


@app.route('/youtube', methods=['GET', 'POST'])
def youtube():
    answer1=""
    keywords=[]
    if request.method == 'POST':
        urll = request.form.get('url_youtube')
        if urll:
            answer1 = str(get_video_transcript(urll))
            keywords = perform_keyword_analysis(answer1)
            sentiment = sentiment_comments(urll)
            session['sentiments'] = sentiment
            session['keywords'] = keywords   
        else:
            answer1 = "No text provided"
            session['keywords'] = keywords
        session['summary'] = answer1
        return redirect(url_for('final_youtube'))
    return render_template("youtube.html")


@app.route('/final_youtube', methods=['GET','POST'])
def final_youtube():
    webscraped=[]
    keywords = session['keywords']
    summary = session['summary']   
    res = english_text_to_mp3(summary)
    sentiment = session['sentiments'] 
    if request.method=='POST':
        key_words = request.form.get('keywordz_youtube')
        keyword_list = key_words.split(',')
        for kw in keyword_list:
            if kw:
                web_scrape = scrape_wikipedia(kw,summary)
                if web_scrape:
                    webscraped.append(web_scrape)
                else:
                    empty_message = f"Sorry, couldn't find much information regarding '{kw}' try using differnet word or relevant a combination of this word!"
                    webscraped.append(empty_message)
            else:
                no_keyword_message = "Enter a keyword!"
                webscraped.append(no_keyword_message)
        
    return render_template("final_youtube.html", summarized_text=summary, keyword=keywords, webscrape=webscraped, rating=sentiment)


def find_sentences_with_word(paragraph, target_word):
    sentences = nltk.sent_tokenize(paragraph)
    result_sentences = []
    for sentence in sentences:
        if target_word.lower() in sentence.lower():
            result_sentences.append(sentence)
    return result_sentences


def scrape_ndtv(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    target_class = 'pst-by_ul'
    elements_with_class = soup.find_all(class_=target_class)
    for element in elements_with_class:
        publishers = element.find_all(itemprop="name")
        update_date =  element.find_all(itemprop="dateModified")
    author_names = [author.text.strip() for author in publishers]
    update = [date.text.strip() for date in update_date]
    author_names = ', '.join(author_names)
    update= ''.join(update)
    result_publish = 'By '+ author_names + ' / '+  update
    target_class2 = 'sp-cn ins_storybody'
    elements_with_class2 = soup.find_all(class_=target_class2)
    for elements in elements_with_class2:
        paragraphs = elements.find_all('p')
    paragraph = [para.text.strip() for para in paragraphs]
    result_para = ' '.join(paragraph)
    results=[]
    results.append(result_publish)
    results.append(result_para)
    return results


def scrape_cnn(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    target_class = 'byline__names'
    elements_with_class = soup.find_all(class_=target_class)
    publisher_name = [para.text.strip() for para in elements_with_class]
    publisher_name='. '.join(publisher_name)
    target_class2 = 'timestamp'
    elements_with_class2 = soup.find_all(class_=target_class2)
    publish_time = [para.text.strip() for para in elements_with_class2]
    publish_time=' '.join(publish_time)
    publishdetail = []
    publishdetail.append(publisher_name)
    publishdetail.append(publish_time)
    result_publish = ' / '.join(publishdetail)
    target_class3 = 'article__content'
    elements_with_class3 = soup.find_all(class_=target_class3)
    paragraphs = [link.text.strip() for link in elements_with_class3]
    paragraphs = ' '.join(paragraphs)
    result_para = paragraphs.replace('\n','')
    results=[]
    results.append(result_publish)
    results.append(result_para)
    return results


def scrape_bbc(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    target_class = 'ssrcss-1if1g9v-MetadataText e4wm5bw1'
    elements_with_class = soup.find(class_=target_class)
    time_stamp = [author.text.strip() for author in elements_with_class]
    time_stamp = '.'.join(time_stamp)
    target_class2 = 'ssrcss-68pt20-Text-TextContributorName e8mq1e96'
    elements_with_class2 = soup.find(class_=target_class2)
    author_name = [author.text.strip() for author in elements_with_class2]
    author_name = ''.join(author_name)
    target_class3 = 'ssrcss-11r1m41-RichTextComponentWrapper ep2nwvo0'
    elements_with_class3 = soup.find_all(class_=target_class3)
    paragraphs = [author.text.strip() for author in elements_with_class3]
    result_para = ' '.join(paragraphs)
    result_pubish = author_name+" / "+time_stamp
    results=[]
    results.append(result_pubish)
    results.append(result_para)
    return results


def scrape_toi(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    target_class = 'xf8Pm byline'
    elements_with_class = soup.find_all(class_=target_class)
    publishdetails = [para.text.strip() for para in elements_with_class]
    result_publish = ''.join(publishdetails)
    result_publish = "By "+result_publish
    target_class2 = '_s30J clearfix'
    elements_with_class2 = soup.find_all(class_=target_class2)
    paragraphs = [link.text.strip() for link in elements_with_class2]
    result_para=' '.join(paragraphs)
    results=[]
    results.append(result_publish)
    results.append(result_para)
    return results


def split_url(url):
    empty=[]
    url_parts = url.split('/')
    for part in url_parts:
        part2 = part.split('.')
        for part3 in part2:
            if part3=="ndtv":
                answer=scrape_ndtv(url)
                return answer
            elif part3=="timesofindia":
                answer=scrape_toi(url)
                return answer
            elif part3=="cnn":
                answer=scrape_cnn(url)
                return answer
            elif part3=="bbc":
                answer=scrape_bbc(url)
                return answer
    return empty


@app.route('/url', methods=['GET','POST'])
def url():
    answer1=""
    if request.method == 'POST':
        urll = request.form.get('url')
        if urll:
            answer = split_url(urll)
            session['publishnews'] = answer[0]
            answer1 = answer[1]
        else:
            answer1 = "No text provided"
        session['webscrapednews'] = answer1
        return redirect(url_for('second_url'))
    return render_template("url.html")


@app.route('/second_url', methods=['GET','POST'])
def second_url():
    result_publish = session['publishnews']
    result_para = session['webscrapednews']
    final_para=""
    summarized_news=""""""
    if result_para:
        final_para = result_para
    else:
        final_para = "Please Enter a relevant link!"
    if request.method == 'POST':
        numberofline = request.form.get('number_url')
        summarized_news=txxt(final_para, int(numberofline))
        session['summary']=summarized_news
        return redirect(url_for('third_url'))
    return render_template("second_url.html", publish=result_publish, para=final_para)


@app.route('/third_url', methods=['GET','POST'])
def third_url():
    webscraped=[]
    searchednews=[]
    no_keyword_message="Enter a Keyword!"
    summarized_news = session['summary']
    res = english_text_to_mp3(summarized_news)
    actual_news = session['webscrapednews']
    keywords = perform_keyword_analysis(summarized_news)
    if request.method=='POST':
        wiki_keyword = request.form.get('wiki_keyword')
        news_keyword = request.form.get('news_keyword')
        keyword_list_wiki = wiki_keyword.split(',')
        keyword_list_news = news_keyword.split(',')
        for kw in keyword_list_wiki:
            if kw:
                web_scrape = scrape_wikipedia(kw,summarized_news)
                if web_scrape:
                    webscraped.append(web_scrape)
                else:
                    empty_message = f"Sorry, couldn't find much information regarding '{kw}' try using differnet word or relevant a combination of this word!"
                    webscraped.append(empty_message)
            else:
                webscraped.append(no_keyword_message)
        for kw in keyword_list_news:
            if kw:
                sentences= find_sentences_with_word(actual_news,kw)
                for sentence in sentences:
                    searchednews.append(sentence)
            else:
                searchednews.append(no_keyword_message)   
    return render_template('third_url.html', keyword=keywords, summarized_text=summarized_news, webscrape=webscraped, search=searchednews)


def scrape_wikipedia(kw,summary):
    summ = "Provide the definition or the meaning of "+kw
    response =  genai.GenerativeModel("gemini-pro").generate_content(summ)
    return response.text


if __name__ == '__main__':
    app.run(debug=True)