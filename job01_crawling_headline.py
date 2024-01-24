from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime

category = ['Politics','Economic','Social','Culture','World','IT']
#url='https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100#&date=%2000:00:00&page=1'

headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
#크롤링일경우 서버에서 막을수 있음. 해결책->헤더에 정보를 보내면 우회 가능
# html 해당 페이지 검사 -> edit html 후에(data-useragent= " ") 쌍따움표 안에 있는 내용 가져오면 됨
# resp = requests.get(url,headers=headers)
#
# print(type(resp))
# print(list(resp))
#
# soup = BeautifulSoup(resp.text, 'html.parser')
# print(soup)
# title_tags = soup.select('.sh_text_headline')
# print(title_tags)
# print(len(title_tags))
# print(title_tags[0])
# titles=[]
# for title_tag in title_tags:
# #    print(title_tag.text)
#     titles.append(re.compile('[^가-힣 | a-z | A-Z]').sub(' ',title_tag.text)) #정규표현식이용
#
# print(titles)
#
df_titles = pd.DataFrame()
re_title = re.compile('[^가-힣 | a-z | A-Z]')

for i in range(6):
    url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}'.format(i)
    resp = requests.get(url,headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    title_tags = soup.select('.sh_text_headline')

    titles = []
    for title_tag in title_tags:
        titles.append(re_title.sub(' ',title_tag.text))
    df_section_titles = pd.DataFrame(titles, columns=['titles'])
    df_section_titles['category'] = category[i]
    df_titles = pd.concat([df_titles, df_section_titles],axis = 'rows',
                          ignore_index=True)
print(df_titles.head())
df_titles.info()
print(df_titles['category'].value_counts())
df_titles.to_csv('./crawling_data/naver_headline_news_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d')),index=False)


