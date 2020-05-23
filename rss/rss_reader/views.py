from django.http import HttpResponse
import json
import requests
from bs4 import BeautifulSoup
import re
from googletrans import Translator
from summarize import summarize
from django.views.decorators.csrf import csrf_exempt
from google.cloud import translate_v2
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import feedparser
import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Petar/Downloads/Django rss-2958201af05c.json"

def compare_title_article(title,article):
    title_not_translated=title
    title=translate(title,'hr','en',formatT='text')
    article=translate(article,'hr','en',formatT='text')
    article_sent=[s.replace('\n','') for s in sent_tokenize(article)]


    maxi=0
    max_article=''
    for sent in article_sent:
        # tokenization 
        X_list = word_tokenize(title)  
        Y_list = word_tokenize(sent) 
        
        # sw contains the list of stopwords 
        sw = stopwords.words('english')  
        l1 =[];l2 =[] 
        
        # remove stop words from string 
        X_set = {w for w in X_list if not w in sw}  
        Y_set = {w for w in Y_list if not w in sw} 
        
        # form a set containing keywords of both strings  
        rvector = X_set.union(Y_set)  
        for w in rvector: 
            if w in X_set: l1.append(1) # create a vector 
            else: l1.append(0) 
            if w in Y_set: l2.append(1) 
            else: l2.append(0) 
        c = 0
        
        # cosine formula  
        for i in range(len(rvector)): 
                c+= l1[i]*l2[i] 
        cosine = c / float((sum(l1)*sum(l2))**0.5) 
        
        if cosine>maxi:
        
            maxi=cosine
            max_article=sent
    
    ##############################################
    ##############################################

    sent=summarize_article(article,1)
    # tokenization 
    X_list = word_tokenize(title)  
    Y_list = word_tokenize(sent) 
    
    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 
    
    # remove stop words from string 
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 
    
    # form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
    
    # cosine formula  
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5) 
    #print('Cosine za sazetak '+str(cosine))
    if cosine>maxi:
        maxi=cosine
        max_article=sent

    # print('*********************************')
    # print(title)
    # print(max_article)
    # print("similarity: ", maxi) 
    # print(summarize_article(article,1))
    # print('*********************************')

    return HttpResponse(json.dumps({'cosine':maxi,'title':title_not_translated,'sentence':translate(max_article,'en','hr','text')},ensure_ascii=False))


@csrf_exempt
def compare(request):
   
    link = json.loads(request.body)['link']
    title = json.loads(request.body)['title']
    

    if "index.hr" in link:
        parsed= parseIndex(link,None,False)
    elif "jutarnji.hr" in link:
        parsed=  parseJutarnji(link,None,False)
    elif "24sata.hr" in link:
        parsed=  parse24(link,None,False)
    elif "dnevnik.hr" in link:
        parsed=  parseDnevnik(link,None,False)
    elif "bug.hr" in link:
        parsed= parseBug(link,None,False)

    resp=compare_title_article(title,parsed)

    return HttpResponse(resp)

def getClickbaits(sent,titles,links):
    with open("C:\\Users\\Petar\\Desktop\\python\\DjangoEnv\\rss\\rss_reader\\clickbait.txt",encoding='utf-8') as f:
        lines = f.read().strip().split("\n")
        lines = [line.split("\t") for line in lines]
    headlines, labels = zip(*lines)

    train_headlines = headlines[:8000]
    test_headlines = headlines[8000:]

    train_labels = labels[:8000]
    test_labels = labels[8000:]
    vectorizer = TfidfVectorizer()
    svm = LinearSVC()
    train_vectors = vectorizer.fit_transform(train_headlines)
    
    test_vectors = vectorizer.transform(sent)
    svm.fit(train_vectors, train_labels)
    predictions = svm.predict(test_vectors)
    confidence=svm.decision_function(test_vectors)
    
    li=[]
    dic={'Title':'', 'Clickbait':'', 'Confidence':'','Link':''}
    for i in range(len(titles)):
        dic['Title']=(titles[i])
        dic['Clickbait']=predictions[i]
        dic['Confidence']=confidence[i]
        dic['Link']=links[i]
        li.append(dic.copy())
    #pd.DataFrame(dic).to_excel('All.xlsx')
    newlist = sorted(li, key=lambda k: k['Confidence'],reverse=True) 
    return HttpResponse(json.dumps(newlist))
   
def clickbaits(request):
    links=['https://www.bug.hr/rss','https://www.index.hr/rss','https://www.jutarnji.hr/rss','https://www.24sata.hr/feeds/aktualno.xml','https://dnevnik.hr/assets/feed/articles']
    
    titles=[]
    linkovi_na_clanke=[]
    for link in links:
        response = feedparser.parse(link)
        for entry in response.entries:
            titles.append(entry.title)
            linkovi_na_clanke.append(entry.link)
  
    print('poceo prijevod')
    spojeno='\n'.join(titles)
    prijevod=translate(spojeno,'hr','en',formatT='text')
    prijevod=prijevod.split('\n')
    print('preveo clanke idem u clickbait')

    return(getClickbaits(prijevod,titles,linkovi_na_clanke))

def translate(text, src, dest,formatT=None):
    # translator = Translator()
    # return translator.translate(text, src=src, dest=dest).text
    translate_client = translate_v2.Client()
    result = translate_client.translate(
        text, target_language=dest, model='nmt',format_=formatT)
    return result['translatedText'].replace("&quot;", '"').replace("&#39", '"').replace(';', '"')


def summarize_article(text,length):

    return summarize(text,length, 'english')


def translate_summarize(article,length):
    article = re.sub('\n\n*', '\n\n', article)
    translated = translate(article, 'hr', 'en')
    summarized = summarize_article(translated,length)
    article = translate(summarized, 'en', 'hr')
    print(length)
    return article


def parseJutarnji(link,length,summ=True):
    r = requests.get(link)
    bs = BeautifulSoup(r.text, features="html.parser")
    bs.find("div", {"class": "picture-author"}).extract()
    article = bs.find(id=re.compile("content-body-")
                      )
    for a in article.findAll('a'):
        a.decompose()
    for media in article.findAll("div", {"class": "media-body"}):
        media.decompose()
    # article = " ".join(article.split())
    article = article.text.replace(
        "Tablice omogućuje SofaScore LiveScore", '').replace('PROMO', '')

    if summ:
        return HttpResponse(translate_summarize(article,length))
    else:
        return article



def parseIndex(link,length,summ=True):
    r = requests.get(
        link)
    bs = BeautifulSoup(r.text, features="html.parser")
    article = bs.find("div", {"class": "content-holder"}
                      ).find("div", {"class": "text"})
    # article = " ".join(article.split())
    for a in article.findAll('a'):
        a.decompose()
    for img in article.findAll('img'):
        img.decompose()
    article = article.text.replace("Index.me aplikaciju za android besplatno možete preuzeti na , dok iPhone aplikaciju možete preuzeti .", '').replace(
        "Želite li momentalno primiti obavijest o svakom objavljenom članku vezanom uz koronavirus instalirajte Index.me aplikaciju i pretplatite se besplatno na tag: koronavirus", '').replace("Tekst se nastavlja ispod oglasa", '').replace("Standings provided by", '')

    if summ:
        return HttpResponse(translate_summarize(article,length))
    else:
        return article


def parse24(link,length,summ=True):
    r = requests.get(
        link)
    bs = BeautifulSoup(r.text, features="html.parser")

    article = bs.find("div", {"class": "article__text"})

    for a in article.findAll('a'):
        a.decompose()
    for media in article.findAll("span", {"class": "article__main_img"}):
        media.decompose()
    article = article.text.replace(
        "Vaš preglednik ne omogućava pregled ovog sadržaja.", '')
    if summ:
        return HttpResponse(translate_summarize(article,length))
    else:
        return article



def parseDnevnik(link,length,summ=True):
    r = requests.get(link)
    bs = BeautifulSoup(r.text, features="html.parser")

    article = bs.find("div", {"class": "article-body"})
    for media in article.findAll("blockquote", {"class": "twitter-tweet"}):
        media.decompose()
    for a in article.findAll('a'):
        a.decompose()
    article = article.text.replace("Vezani članci", "").replace(
        "Vijesti gledajte svakog dana na Novoj TV, a više o najvažnijim vijestima čitajte na portalu DNEVNIK.hr.", '').replace("Propustili ste vijesti?", '').replace("Pogledajte ih", '')
    if summ:
        return HttpResponse(translate_summarize(article,length))
    else:
        return article



def parseBug(link,length,summ=True):

    r = requests.get(link)
    bs = BeautifulSoup(r.text, features="html.parser")

    article = bs.find("div", {"class": "post-full__content"})

    for a in article.findAll('a'):
        a.decompose()
    for media in article.findAll("div", {"class": "fb-post"}):
        media.decompose()
    article = article.text
    if summ:
        return HttpResponse(translate_summarize(article,length))
    else:
        return article



@csrf_exempt
def index(request):

    link = json.loads(request.body)['link']
    length = int(json.loads(request.body)['length'])


    if "index.hr" in link:
        return parseIndex(link,length)
    elif "jutarnji.hr" in link:
        return parseJutarnji(link,length)
    elif "24sata.hr" in link:
        return parse24(link,length)
    elif "dnevnik.hr" in link:
        return parseDnevnik(link,length)
    elif "bug.hr" in link:
        return parseBug(link,length)
