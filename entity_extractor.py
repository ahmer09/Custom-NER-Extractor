import re
import json
import requests
from requests import get
from bs4 import BeautifulSoup
import spacy
from spacy import displacy
from collections import Counter
from string import punctuation
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import random

class NamedEntityExtractor:
    """
    Performs named entity recognition from texts
    """

    def extract(self, text: str):
        """
        Performs named entity recognition from text
        :param text: Text to extract
        """
        # load spacy nlp library
        spacy_nlp = spacy.load('C:\\Users\\Hammer\\PycharmProjects\\Webcrawler-Extractor\\output')

        # parse text into spacy document
        doc = spacy_nlp(text.strip())

        # create sets to hold words
        topic_entities = []


        for i in doc.ents:
            entry = str(i.lemma_).lower()
            text = text.replace(str(i).lower(), "")
            # organization entities detection
            if i.label_ in ["TOPIC"]:
                topic_entities.append(entry)
            # Geographical and Geographical entities detection
            elif i.label_ in ["ORG"]:
                topic_entities.append(entry)
            # extract artifacts, events and natural phenomenon from text
            elif i.label_ in ["ART", "EVE", "NAT", "PRODUCT", "GPE"]:
                topic_entities.append(entry.title())

        #print(f"topic entities - {topic_entities}")
        #keywrds = random.sample(topic_entities, 2)
        keywrds = topic_entities
        return keywrds



    def get_text_frm_url(self, url):
        headers = {"Accept-Language": "en-US, en;q=0.5"}
        req = requests.get(url, headers=headers)
        soup = BeautifulSoup(req.text, features="lxml")
        texts = soup.get_text()
        return texts

    def clean(self, text):
        text = re.sub('[0-9]+.\t', '', str(text))
        text = re.sub('\n ', '', str(text))
        text = re.sub('\n', ' ', str(text))
        text = re.sub("'s", '', str(text))
        text = re.sub("-", ' ', str(text))
        text = re.sub("— ", '', str(text))
        text = re.sub('\"', '', str(text))
        text = re.sub("^\x20-\x7E", "", str(text))
        # text = re.sub(text, "\\u00AE|\\u00a9|\\u2122", "")
        return text

if __name__ == '__main__':
    named_entity_extractor = NamedEntityExtractor()
    text = "eBay Bulk Product Upload/Listing, Amazon Bulk Upload/Listing, Data Entry Services                            ﻿ Home About Us Blog Contact Us 415 906 0457           eBay Product Based Services:  eBay Data Entry  eBay Product Listing eBay Product Description  eBay Product Uploading eBay Data Upload Services  eBay List Building eBay Bulk Product Listing   Outsource eBay Data Entry Services eBay Store Creation eBay Bulk Product Upload eBay Tools We Use:  eBay Auctiva Uploading eBay InkFrog Product Listing  Amazon        Amazon Bulk Data Upload Amazon Bulk Product Listing      Amazon Data Entry Amazon Data Upload Outsource Amazon Data Entry    Amazon List Building Amazon Product Listing     Amazon Store Creation Amazon Product Upload Amazon Product Descrption         Shopify        Shopify Bulk Data Upload Shopify Bulk Product Listing      Shopify Data Entry Shopify Data Upload    Shopify List Building Shopify Product Listing     Shopify Store Creation Shopify Product Upload Shopify Product Descrption         Magento        Magento Bulk Data Upload Magento Bulk Product Listing      Magento Data Entry Magento Data Upload    Magento List Building Magento Product Listing     Magento Store Creation Magento Product Upload Magento Product Descrption         Zen Cart        Zen Cart Bulk Data Upload Zen Cart Bulk Product Listing      Zen Cart Data Entry Zen Cart Data Upload    Zen Cart List Building Zen Cart Product Listing     Zen Cart Store Creation Zen Cart Product Upload Zen Cart Product Descrption         OpenCart        OpenCart Bulk Data Upload OpenCart Bulk Product Listing      OpenCart Data Entry OpenCart Data Upload    OpenCart List Building OpenCart Product Listing     OpenCart Store Creation OpenCart Product Upload OpenCart Product Descrption         BigCommerce        BigCommerce Bulk Data Upload BigCommerce Bulk Product Listing      BigCommerce Data Entry BigCommerce Data Upload    BigCommerce List Building BigCommerce Product Listing     BigCommerce Store Creation BigCommerce Product Upload BigCommerce Product Descrption        COMPLETE ECOMMERCE SOLUTIONSDATA ENTRY#1 Data Entry Experts & eCommerce Product Listers The fastest way to grow your business with the leader in Data Entry ServicesWe help in Streamline your Data for Better Business Performance.Skype Us Now!Data Entry AdriotsWe provide data entry services and Outsource Data Entry Services. We at Data Entry Adroits strive towards providing industry best practices of Online Data entry, Offline Data entry, Manual data entry, Image data entry, PDF data entry and book data entry services. Our clients attest to our team of highly trained professionals who diligently deliver excellent offshore services. We strictly adhere to the agreed upon guidelines and the suggested time frames. We accommodate the needs of a wide variety of clients with diverse needs thanks to our robust and dynamic technical support system.  Read MoreOnline & Offline Data Entry ServicesOnline Data Entry ServicesWe are experts in providing Online Data Entry Services with high standards of data entry quality controls.Offline Data Entry ServicesWe have highly qualified team specifically trained to handle offline data entry projects for our clients.PDF Data Entry ServicesWe offer a rich array of PDF Data Entry Services that streamline your business operations for imperative business activities.Image Data Entry ServicesOur Image Data Entry Services help you to convert scanned images into a any format of your choice for an easy access of data. We Love Your Ping!   Get in Touch Now!Contact us or give us a call to discover how we can help.Your name *Your email address *SubjectMessage *Message has been sent to us.Error sending your message.We Provide theAccuratePreciseQualityData Entry Services & Product Listing Services on eBay and Amazon StoresWe provide a wide range of offshore services all under one roof and in a one stop basis. You no longer have to worry about in house troubles because you are able to outsource data entry, back office support and entire ecommerce product management services to us. StrategyPlanningExecuteDeliver What More in Data Entry Services & eCommerce Product Uploading ServicesWe keep up to date with the latest technologies which help in the conversion of bulky physical papers to electronically compatible forms that can be used across multiple digital platforms.Outsource Data EntryCopy Paste ServicesBook Data EntryManual Data EntryeBay Data EntryeBay Listing ServiceseBay Product UploadeBay Store CreationAmazon Data EntryAmazon ListingAmazon Data UploadAmazon List BuildingImage EditingPhoto Clipping PathImage EnhancementPhoto ResizingContent WritingWebsite Blog WritingPress Release WritingDigital Marketingand so much more... Data Entry Services By Data Entry AdroitsAt Data Entry Adroits, our every single employee works really hard to give a definite competitive edge to all our clients and always ensure that our service offerings exceed their expectations. Hence, we continuously refine and filter our processes for the next level of customer support.Discuss Your Project With Us Now: Let us apply our capabilities in enhancing your business objectives. To learn more about our unique ideas or out of the box solutions, please write to us at info@dataentryadroits.com.FAQ   Data Entry AdroitsDo you offer a free trial for Data Entry and eCommerce Product Uploading Services?Yes, we offer free trials for all Data Entry Services and Product uploading services for eCommerce stores. We believe, it’s important for any client to first see the quality of work by assigning some sample work which we will do for completely free of cost. Once you satisfied with the quality of our work, you can assign us an entire project.What kind of clients and industries have you are Working for?We are working for wide industry verticals and great variety of business. Our main service industry verticals includes, eCommerce Industry, Law Firms, Education Institutions/Universities, Healthcare and Medical, Real Estate Firms, Online Stores, Finance Industry, Human Resource Industry, Job Portals, Marketing Firms, NGOs, Etc. How do I get started with Data Entry Adroits for my Data Entry or eCommerce project?Well, that’s pretty easy. All you need to send your requirements at info@dataentryadroits.com or you can get in touch with our top departments through Skype. After all the discussions for your project requirements and the deadlines you can share your files via Emails, FTP uploads, Dropbox or you can simply courier to us.How do I track the progress of my Project?By every EOD the concern project manager supervising your project will share the detailed work reports with all updates on the progress of your project. We maintain very easy open line of communication through email for all the work reposts or by any specific model preferred by you.How do you receive payments?All clients based Outside India can pay us through Paypal and for all Indian clients, you can pay us via Wire Transfer.Why Our Clients Choose UsWe have 5 years of experience providing the most affordable, flexible and high quality services to our esteemed clients.We Are #1 Data Entry Experts Data Entry Adroits is all you need for Data Entry Services!The #1 eCommerce Product Listing Service CompanySkype Now Success! Your message has been sent to us.Error! There was an error sending your message.Contact UsYour name *Your email address *SubjectMessage *Get in TouchOutsource all your data entry and back office tasks to us. The benefits of doing so include reduced costs, high quality services, and greater flexibility, wider market access, increased credibility by associating with reputed service providers, access to innovation, reduced dependence on internal resources and increased focus on core functions.The Office Headquater: F 75/B, Street No.3, Mangal Bazar, Laxmi Nagar, New Delhi   110092 USA Location: 548 Market St, San Francisco, CA 94104, United States Phone: +91 9958080618, +91 8802555230 Email: info@dataentryadroits.comBusiness Hours Monday   Friday   9am to 6pm Saturday Sunday   Closed﻿    Get in Touch    Site Usage PolicyThe information on this website is protected by copyright. Users of this website are not authorized to redistribute, reproduce, republish, modify, or make commercial use of the information without the written authorization of MysticDigi Pvt. Ltd. We are committed to the prevention of copyright infringement.    Our Services  Data Entry Services eBay Product Listing Services Amazon Product Listing Services Content Writing Services Digital Marketing Services     Contact Us Headquater: F 75/B, Street No.3, Mangal Bazar, Laxmi Nagar, New Delhi   110092USA Location: 548 Market St, San Francisco, CA 94104, United StatesPhone: +91 9958080618, +91 8802555230Email: info@dataentryadroits.com     Follow Us                  Amazon Listing Services Amazon Services: Amazon Bulk Product Upload Services  | Amazon Product List Building  | Outsource Amazon Data Entry  | Amazon Data Entry Services  | Amazon Product Listing Services  | Amazon Data Upload Services  | Amazon Bulk Product Listing  | Amazon Store Creation     eBay Listing Services eBay Services: eBay Bulk Product Uploading  | eBay Data Upload Services  | eBay Bulk Product Listing  |  | eBay Bulk Product Upload Services  | eBay Product Listing Services  | eBay List Building Services  | eBay Data Entry Services  | eBay Store Creation Services  | eBay InkFrog Product Listing  | Terapeak eBay Data Research      eCommerce Product Optimization Image Editings: Image Editing Services   | Photo Editing Services  | Photo Enhancement  | Photo Restoration Services  | Image Masking Services  | Photo Clipping Path  | Photo Cutout Services  | Image Background Removal  | Photo Retouching Services  | Photo Resizing Services  | Image Colorization              © Copyright 2017. All Rights Reserved.  | Developed By: SEO Service in India     About us Contact Blog XML SiteMap HTML SiteMap ROR SiteMap       DataEntryAdroits5.0/5 based on 7 Google My Business Rivews"
    urls = ["http://www.spracht.com/about/","https://www.dataentryadroits.com/","http://www.enmassenergy.com","https://www.docupile.com","https://www.fieldlevel.com","https://www.weblogic.com","http://www.opcodesolutions.com"]
    data = {}
    data['startup'] = []
    for url in urls:
        text = named_entity_extractor.get_text_frm_url(url)
        text = named_entity_extractor.clean(text)
        keywrds = named_entity_extractor.extract(text)
        #item = {"keywords": keywrds}
        data['startup'].append({
            'url': url,
            'keywords': keywrds
        })
    jsonData = json.dumps(data)

    # Writing to sample.json
    with open("sample_custm.json", "w") as outfile:
        outfile.write(jsonData)