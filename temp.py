# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 20:03:41 2018

@author: SurfaceBook2
"""
urlid = 39285983
url = "https://pantip.com/topic/" + (str)(urlid)

# display-post-wrapper with-top-border section-comment
import bs4 as BeautifulSoup
import re
from selenium import webdriver
import bleach

driver = webdriver.Chrome()
driver.get(url)
html = driver.page_source
soup = BeautifulSoup.BeautifulSoup(html, 'lxml')
driver.close()
mydivs = soup.find_all("div", class_="display-post-story")
re.sub
comment = []
for div in mydivs:
    text = bleach.clean(div.text.strip(), tags=[], attributes={}, styles=[], strip=True)
    clean_text = ' '.join(text.split())
    if len(clean_text) > 0:
        comment.append(clean_text)

with open('' + (str)(urlid) + '.txt', 'w+') as f:
    for item in comment:
        f.write("%s\n" % item)
f.close()
