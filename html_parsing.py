# import libraries

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import json


class ArticleTable(object):
    """From medium page with listing of articles get links and then extract info from each article"""

    def __init__(self, list_of_urls):
        self.list_of_urls = list_of_urls

    def get_urls_from_listing(self):
        valid_links = []
        for item in self.list_of_urls:
            req = requests.get(item)
            soup = BeautifulSoup(req.text, features="html")
            for link in soup.find_all('a'):
                links = link['href']
                if links.startswith('https'):
                    valid_links.append(links)
        self.list_links = (dict.fromkeys(valid_links))
        return self.list_links

    def clean_links(self):
        """Deletes added text to article links"""
        self.clean_list_of_links = [re.sub(r"\?source.*", "", string) for string in self.list_links]
        return self.clean_list_of_links

    def main(self):
        """Execute pipeline
        For each link extracted from listing links it applies information extraction

        Returns:
            Pandas table
        """
        self.get_urls_from_listing()
        self.clean_links()

        lines = []
        titles_list = []
        for link in self.clean_list_of_links:
            ap = ArticleParsing(link)
            line = ap.main()
            if not line.empty:
                if line.title.values not in titles_list:
                    titles_list.append(line.title.values)
                    lines.append(line)
        self.df = pd.concat(lines, sort=False, axis=0).reset_index(drop=True)
        return self.df


class ArticleParsing(object):
    """From medium webpage url extract article info"""

    def __init__(self, input_url):
        """Takes as input an url with article
        Args:
            input_url (str)- url to a valid medium webpage
        """
        self.input_url = input_url
        self.text = ''
        self.dico = {}

    def extract_soup(self):
        """Extracts Beautifulsoup object from url"""
        req = requests.get(self.input_url)
        self.soup = BeautifulSoup(req.text, "lxml")
        return BeautifulSoup(req.text, "lxml")

    def clean(self, text):
        # removing paragraph numbers
        text = re.sub('[0-9]+.\t', '', str(text))
        # removing new line characters
        text = re.sub('\n ', '', str(text))
        text = re.sub('\n', ' ', str(text))
        # removing apostrophes
        text = re.sub("'s", '', str(text))
        # removing hyphens
        text = re.sub("-", ' ', str(text))
        text = re.sub("— ", '', str(text))
        # removing quotation marks
        text = re.sub('\"', '', str(text))
        # removing any reference to outside text
        text = re.sub("[\(\[].*?[\)\]]", "", str(text))

        return text

    def extract_meta(self):
        """Extracts article metadata such as tags, tipics, author, dates, title, subtitle, publisher"""
        TagsList = []
        TopicList = []
        try:
            script = self.soup.find("script", {"type": "application/ld+json"})
            y = json.loads(script.string)
            for entry in y["keywords"]:
                if entry.startswith("Tag:"):
                    TagsList.append(entry.replace("Tag:", ""))
                if entry.startswith("Topic:"):
                    TopicList.append(entry.replace("Topic:", ""))

            self.dico = {
                'tags': "; ".join(TagsList),
                'topic': "; ".join(TopicList),
                'date_created': pd.to_datetime(y['dateCreated']),
                'date_published': pd.to_datetime(y['dateCreated']),
                'date_modified': pd.to_datetime(y['dateModified']),
                'title': y["name"],
                'subtitle': y["description"],
                'url': y["url"],
                'publisher': y["publisher"]["name"],
                'author': y["author"]["name"]
            }
        except:
            self.dico = {}
        return self.dico

    def extract_text(self):
        "Extracts main text taking care putting spaces when lists are used"
        try:
            art = self.soup.find("article")  # <article> tag delimitates article
            uls = art.find_all(['ul', 'p', 'h1', 'h2', 'h3', 'h4'])  # get all interesing html tags
            lis = []
            text = []
            for ul in uls:
                if (ul.name == 'ul'):
                    for li in ul.findAll('li'):
                        lis.append(li)
                else:
                    lis.append(ul)
            for li in lis:
                x = li.text
                x = x.str.encode()
                x = self.clean(x)
                text.append(x)

            text_f = ' '.join(text[2:])  # join text list wirh spaces
            self.text = text_f
            self.dico["text"] = self.text
            return text_f
        except:
            text_f = "not found"
            return text_f

    def return_pandas(self):
        return pd.DataFrame([self.dico])

    def main(self):
        """Run the pipeline
        Retruns a pandas dataframe line
        """
        self.soup = self.extract_soup()
        self.dico = self.extract_meta()
        self.text = self.extract_text()
        return self.return_pandas()