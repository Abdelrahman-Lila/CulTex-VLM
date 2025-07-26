import pandas as pd
import wikipediaapi
import re
import requests
import glob
import os

from playwright.sync_api import sync_playwright, Playwright
from playwright._impl._errors import  rewrite_error
from playwright.sync_api import Error, TimeoutError
import html2text
from bs4 import BeautifulSoup
from rapidfuzz import fuzz



class Pipeline:
    def __init__(self, concept_db_uri, vector_db_uri=None, images_db_uri=None):
        
        self.concept_db_uri = concept_db_uri
        self.id_concept_df = pd.read_csv(concept_db_uri, index_col=0)
        self.images_db_uri = images_db_uri
        
        # client = chromadb.PersistentClient(path="/kaggle/working")
        # embedding_function = OpenCLIPEmbeddingFunction()
        # data_loader = ImageLoader()
        # self.collection = client.get_collection(name='concepts_images_knowledgeBase')

        self.__wiki_wiki = wikipediaapi.Wikipedia(user_agent='MyProjectName (merlin@example.com)', language='en') # Wikipedia Api setup
        #self.__gemini_client = genai.Client(api_key="AIzaSyD2-wBnIn3gV7awj9DocYbjS_5aIgPH6Ro") # Gemini Api setup


    def download_image(self, url, save_path):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(response.content)
            return save_path
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return None

    def getImages(self, id_concept_list, n=10):
        images = {}
        with sync_playwright() as playwright:
            firefox = playwright.firefox
            browser = firefox.launch(headless=True)
            for id, concept in id_concept_list:
                images[id] = []
                concept = concept.replace(" ", '+').replace("_", '+')
                base_url = f"https://duckduckgo.com/?q={concept}&t=h_&iar=images"
                page = browser.new_page()
                page.goto(base_url)
                page.wait_for_load_state("networkidle")
                html = page.content()
                pattern = r'<div class="SZ76bwIlqO8BBoqOLqYV"><img src="(.*?)" alt'
                for match in re.finditer(pattern, html):
                    images[id].append("https:"+match.group(1))
                    if len(images[id]) == n:
                        break
                page.close()
                
            browser.close()

        for id, img_urls in images.items():
            for i, img_url in enumerate(img_urls):
                self.download_image(img_url, f"{self.images_db_uri}/{id}/{i}.png")


    def searchWiki(self, id_concept_list, n=1):
        def get_wikipedia_titles(html):
        # Regex pattern to match the href and title attributes
            pattern = r'<li class="mw-search-result mw-search-result-ns-0">.*?<a href="(.*?)" title="(.*?)".*?>'
            matches = re.findall(pattern, html, re.DOTALL)
            return matches
        
        wikis = {}
        has_background = []
        for id, concept in id_concept_list:
            wikis[id] = []
            has_background.append(0)
            url = f"https://en.wikipedia.org/w/index.php?search={concept}&title=Special%3ASearch&profile=advanced&fulltext=1&ns0=1"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                titles = get_wikipedia_titles(response.text)
                if n == -1 or n > len(titles):
                    n = len(titles)
                for i in range(n):
                    if fuzz.WRatio(concept, titles[i][1]) > 75:
                        wikis[id].append(titles[i][1])
                        has_background[-1] = 1
        
        for id, wiki_list in wikis.items():
            with open(f"{self.images_db_uri}/{id}/background.md", 'a') as file:
                for wiki in wiki_list:
                    concept_wiki_page = self.__wiki_wiki.page(wiki)
                    file.write(f"\n# Wikipedia: {wiki}\n")
                    file.write(concept_wiki_page.text)
        
        return has_background

    
    def searchBritannica(self, id_concept_list):
        resources = {}
        answers = {}
        has_background = []
        with sync_playwright() as playwright:
            chrome = playwright.chromium
            browser = chrome.launch(headless=True)
            for id, concept in id_concept_list:
                resources[id] = []
                has_background.append(0)
                base_url = "https://www.britannica.com/chatbot"
                
                page = browser.new_page()
                page.goto(base_url, wait_until="domcontentloaded")
                page.wait_for_selector('input[placeholder="Ask a question"]', timeout=15000)
                page.wait_for_timeout(2000)

                prompt = page.get_by_placeholder('Ask a question')
                prompt.type(concept)
                prompt.press('Enter')
                #page.wait_for_selector('.answer-rating-and-retry', timeout=30000)
                try:
                    page.wait_for_function(
                        """(selector) => {
                            const el = document.querySelector(selector);
                            if (!el) return false;
                            if (!el.dataset.lastContentCheck) {
                                el.dataset.lastContentCheck = el.innerHTML;
                                return false;
                            }
                            const same = el.dataset.lastContentCheck === el.innerHTML;
                            el.dataset.lastContentCheck = el.innerHTML;
                            return same;
                        }""",
                        arg=".answer",
                        timeout=30000,
                        polling=500
                    )
                except TimeoutError as e:
                    print(f"Error waiting for answer: {e}")

                answer = page.locator('.answer').inner_html()
                if answer.startswith('Sorry'):
                    page.close()
                    continue
                elif answer.startswith('<span>'):
                    answer = page.locator('.answer > span').inner_html()
                
                answer = html2text.html2text(answer)
                if not answer:
                    page.close()
                    continue
                answers[id] = answer
                has_background[-1] = 1

                # try:
                #     if len(page.locator('.chatGPTSources').inner_html()):
                #         page.close()
                #         continue
                # except Error as e:
                #     print('No resources')
                try:
                    sources = page.locator('.britannicaSources')
                    sources.click()
                    page.wait_for_selector('ol.d-flex')

                    sources = page.locator('ol.d-flex').inner_html()
                    pattern = r'href\s*=\s*["\']?(https?://[^"\'>\s]+)'
                    links = re.findall(pattern, sources)
                    resources[id] = links
                except TimeoutError as e:
                    print("Error no resources")
                    with open(f"{self.images_db_uri}/{id}/background.md", 'a') as file:
                        file.write(f'\n# Britannica:\n')
                        file.write(answer + '\n')

                page.close()            
            browser.close()
        
        for id, links in resources.items():
            if len(links) == 0:
                continue
            with open(f"{self.images_db_uri}/{id}/background.md", 'a') as file:
                file.write(f'\n# Britannica:\n')
                file.write(answers[id] + '\n')
                for link in links:
                    response = requests.get(link)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    selector = 'article.article-content:nth-child(2) > div:nth-child(1) > div:nth-child(2) p.topic-paragraph'
                    paragraphs = soup.select(selector)

                    file.write(f"\n## Britannica: {soup.title.text}\n")
                    for para in paragraphs:
                        file.write(para.text + '\n')
        
        return has_background

    def addNewConcepts_pipeline(self, concepts_list, n_images=10, img_start=0, force_add=False):
        # def hasBackground(id):
        #     content = None
        #     with open(f"{self.images_db_uri}/{id}/background.md", 'r') as file:
        #         content = file.readline()
        #     return len(content) != 0

        id_concept_list = []
        count = len(self.id_concept_df)
        repeated = 0
        for i, concept in enumerate(concepts_list):
            similarity = self.id_concept_df['Concept'].apply(lambda x: fuzz.WRatio(x, concept))
            same_concept = self.id_concept_df[similarity > 88]
            if len(same_concept):
                repeated += 1
                with open("log.txt", 'a') as file:
                    file.write(f'{str(repeated)}. ')
                    file.write(f"Concept '{concept}' Already Exists as:\n")
                    file.write(str(same_concept))
                    print(f"Concept '{concept}' Already Exists as:\n")
                    print(same_concept)
                    file.write('\n------------------\n')
                if not force_add:
                    continue
            id_concept_list.append((str(count), concept))
            count+=1
            os.mkdir(f'{self.images_db_uri}/{id_concept_list[-1][0]}')     

        self.getImages(id_concept_list, n=n_images)
        has_background_wiki = self.searchWiki(id_concept_list)
        has_background_brit = self.searchBritannica(id_concept_list)
        has_background = [bool(x or y) for x,y in zip(has_background_wiki, has_background_brit)]
        new_df = pd.DataFrame(id_concept_list, columns=['id', 'Concept'])
        new_df['has_background'] = pd.Series(has_background)
        self.id_concept_df = pd.concat([self.id_concept_df, new_df], ignore_index=True)
        self.id_concept_df.to_csv(self.concept_db_uri)
    


# ------------------------------------------------- Database Formation ------------------------------------------------------#

# df = pd.read_csv("mapped_qa_data.csv")
# concepts = df['Concept'].drop_duplicates().apply(lambda x: str(x).replace("_", ' ')).values

# pipeline = Pipeline('concepts.csv', None, 'conceptImgDataset')

# batchSize = 1
# for i in range(0, len(concepts), batchSize):
#     batch = concepts[i:i+batchSize]
#     pipeline.addNewConcepts_pipeline(batch, n_images=20)
#     print(i)

# concepts_df = pd.read_csv('concepts.csv', index_col=0)
# more_concepts = list(filter(lambda x: x not in concepts_df["Concept"].values, concepts))
# print(more_concepts)
# batchSize = 1
# for i in range(0, len(more_concepts), batchSize):
#     batch = more_concepts[i:i+batchSize]
#     pipeline.addNewConcepts_pipeline(batch, n_images=20, force_add=True)
#     print(i)

