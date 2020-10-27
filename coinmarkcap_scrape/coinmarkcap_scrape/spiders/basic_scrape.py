#%%

import scrapy
import time
import pandas as pd
import numpy as np

'''
response.css("span.prevnext")[1] gets you the next page button. [0] is prev.
next_button.getall()[0].split("href=")[1].split(">")[0].replace("\"", "")
gets you the actual link to next page. 

response.css("td.td_headerandpost") gets you the list of posts
[0].css("div::text").getall() gets you all of the text.

response.css("td.td_headerandpost")[0].css("div.smalltext::text").get()
gets you the actual date. 

str(pd.to_datetime("February 17, 2016, 10:25:37 AM"))
gets you: '2016-02-17 10:25:37'

'''

class TestSpider(scrapy.Spider):
    name = "bitcointalk"

    def start_requests(self):
        urls = [
        "https://bitcointalk.org/index.php?topic=178336.296000",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        def arr_to_str(arr):
            string = ""
            for f in arr:
                string = f"{f} "
            string = string[:-1]
            return string 
        
        topic = response.css("a.nav::text")[-1].get()
        for post in response.css("td.td_headerandpost"):
            # Introduce delay so hopefully they don't think it's DDOS
            text = post.css("div.post::text").getall()
            text = arr_to_str(text)

            date = post.css("div.smalltext::text").get()

            yield {'date': date, 'topic': topic, 'text': text,}

        next_page = response.css("span.prevnext")[1].css("a.navPages::attr(href)").get()
        # As long as there is a next page, keep searching.
        if len(response.css("span.prevnext")) == 4:
            time.sleep(1)
            yield response.follow(next_page, callback=self.parse)

        # for something in response.css('f6l7tu-0'):
        #     yield {
        #         'price': something.css("span.cmc-details-panel-price__price::text").get(),
                
        #     }

        # next_page = response.css('li.next a::attr(href)').get()
        # if next_page is not None:
        #     yield response.follow(next_page, callback=self.parse)
class TestSpider2(scrapy.Spider):

    def __init__(self):
        self.end_date = 'January 11, 2016, 07:02:21 PM'
        self.back_marker = "«"
        self.count = 0
        
    name = "bitcointalk_threadhopper"
    
    def start_requests(self):
        urls = [
        #"https://bitcointalk.org/index.php?topic=5280666.0;prev_next=prev#new",
        "https://bitcointalk.org/index.php?topic=3189768.40",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        def arr_to_str(arr):
            string = ""
            for f in arr:
                string = string +  f"{f} "
            string = string[:-1]
            return string 
        
        topic = response.css("a.nav::text")[-1].get()
        for post in response.css("td.td_headerandpost"):
            # Introduce delay so hopefully they don't think it's DDOS
            text = post.css("div.post::text").getall()
            text = arr_to_str(text)

            date = post.css("div.smalltext::text").get()
            
            yield {'date': date, 'topic': topic, 'text': text,}
        try:

            prev_page = response.css("span.prevnext")[0].css("a.navPages::attr(href)").get()
        
        except:
            next_topic = response.css("td").css("div.nav").css("a::attr(href)")[1].get()
            self.count = 0
            time.sleep(1)
            yield response.follow(next_topic, callback=self.parse)

        # As long as there is a next page, keep searching.
        if len(response.css("span.prevnext")) > 2:
            time.sleep(1)
            self.count = self.count + 1
            yield response.follow(prev_page, callback=self.parse)

        elif self.count == 0 and len(response.css("span.prevnext")) == 2:
            time.sleep(1)
            self.count = self.count + 1
            yield response.follow(prev_page, callback=self.parse)

        else:
            next_topic = response.css("td").css("div.nav").css("a::attr(href)")[1].get()
            self.count = 0
            time.sleep(1)
            yield response.follow(next_topic, callback=self.parse)

''' Altcoin outline:
1. Get list of pages
2. Set current page as 1 when starting out
3. For each thread in this forum page:
    a. Follow link, call a specific forum_parser function
    b. Get list of pages, scrape content for date, topic, text
    c. Go to next page and repeat until the current page == length of pages found.
4. When done, just go to next page. 
        
'''

class TestSpider3(scrapy.Spider):
    name = "altcointalk_threadhopper"

    def __init__(self):
        self.startCrawl = True
        self.page_list = None
        self.forum_max_page = None
        self.curr_forum_page = 1

        self.curr_thread_page = 1
        self.thread_max_page = None
        self.thread_page_list = None

        self.finished_forum_links = None
        self.finished_thread_links = None

        self.following_thread = False

        #https://www.altcoinstalks.com/index.php?board=15.0

    def start_requests(self):
        urls = [
            "https://www.altcoinstalks.com/index.php?board=15.0",
        # "https://www.altcoinstalks.com/index.php?topic=79189.0",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def thread_scraper(self, response):
        print("Scraping threads")

        if self.thread_page_list is None:
            self.curr_thread_page = 1
            self.thread_page_list = response.css("div.pagelinks")[0].css("a.navPages::text").getall()
            try:
                self.thread_max_page = int(self.thread_page_list[-1])
            except:
                self.thread_max_page = 1

            self.finished_thread_links = [response.request.url]
            

        # links = response.css("div.pagelinks")[0].css("a.navPages::attr(href)").getall()
            
        self.finished_thread_links = np.unique(self.finished_thread_links)
        if self.curr_thread_page <= self.thread_max_page:

        ########################
            print("individual posts on threads")

            def arr_to_str(arr):
                string = ""
                for f in arr:
                    string = string +  f"{f} "
                string = string[:-1]
                return string 

            topic = response.css('li.last').css("span::text").get()
            for post in response.css("div.postarea"):
                # Introduce delay so hopefully they don't think it's DDOS
                text = post.css("div.inner::text").getall()
                text = arr_to_str(text)

                date = post.css("div.smalltext::text").getall()[-1].split(" ")[1:-1]
                date = arr_to_str(date)

                yield {'date': date, 'topic': topic, 'text': text,}
            # self.get_threads(response)

        ########################

            hasSeen = True
            start_ind = 0
            try:
                for f in range(10):
                    if not hasSeen:
                        break

                    next_page = response.css("div.pagelinks")[0].css("a.navPages::attr(href)").getall()[start_ind]
                    start_ind = start_ind + 1
                    hasSeen = next_page in self.finished_thread_links
                
                if not hasSeen:
                    self.curr_thread_page += 1
                    self.finished_thread_links = list(self.finished_thread_links)
                    self.finished_thread_links.append(next_page)
                    self.finished_thread_links = np.unique(self.finished_thread_links)
                    time.sleep(1)
                    yield response.follow(next_page, callback=self.thread_scraper)
                    
            except:
                print("Found no more pages")


    # First function it will call. 
    def parse(self, response):
        
        # Get initial page list.
        if self.startCrawl:
            self.curr_forum_page = 1
            self.page_list = response.css("div.pagesection")[0].css("a.navPages::text").getall()
            self.forum_max_page = int(self.page_list[-1])
            self.startCrawl = False
            self.finished_forum_links = [response.request.url]

        # links = response.css("div.pagesection")[0].css("a.navPages::attr(href)").getall()
        self.finished_forum_links = np.unique(self.finished_forum_links)
        
        if self.curr_forum_page <= self.forum_max_page:
            print("Condition is true")

    ########################
            print("Parsing threads")
            threads = response.css("td.subject").css("span").css("a::attr(href)").getall()

            for thread in threads:
                self.thread_max_page = None
                self.thread_page_list = None

                self.following_thread = True
                time.sleep(1)
                yield response.follow(thread, callback=self.thread_scraper)



            # self.parse_threads(response)

    ########################
            hasSeen = True
            start_ind = 0
            # try:
            for f in range(10):
                if not hasSeen:
                    break

                next_page = response.css("div.pagelinks")[0].css("a.navPages::attr(href)").getall()[start_ind]
                start_ind = start_ind + 1
                hasSeen = next_page in self.finished_forum_links
                # print(hasSeen)

            if not hasSeen:
                print("New page")
                self.curr_forum_page += 1
                self.finished_forum_links = list(self.finished_forum_links)
                self.finished_forum_links.append(next_page)
                self.finished_forum_links = np.unique(self.finished_forum_links)

                time.sleep(1)
                yield response.follow(next_page, callback=self.parse)
                    
            # except:
            #     print("Found no more pages")


class TestSpider2(scrapy.Spider):

    def __init__(self):
        self.count = 0
        
    name = "coinprice_extractor"
    
    def start_requests(self):
        urls = [
        #"https://bitcointalk.org/index.php?topic=5280666.0;prev_next=prev#new",
        # "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20150922&end=20201022",
        "https://coinmarketcap.com/currencies/dogecoin/historical-data/?start=20150922&end=20201022",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        def arr_to_str(arr):
            string = ""
            for f in arr:
                string = string +  f"{f} "
            string = string[:-1]
            return string 
        
        table_rows = response.css("tr.cmc-table-row")

        for row in table_rows:
            # Date, Open, High, Low, Close, Volume, Market Cap
            data = row.css("div::text").getall()

            date, open_, high, low, close_, volume, market_cap = data

            yield {"date": date, "open": open_, "high": high, "low": low, "close": close_, "volume": volume, "market_cap": market_cap}
                  
            


# class TestSpider3(scrapy.Spider):
#     name = "altcointalk"


#     def __init__(self):
#         self.max = 0
#         self.isMax = False
#         self.start_at = 2
#         self.isSearchPage = False
#         #https://www.altcoinstalks.com/index.php?board=15.0

#     def start_requests(self):
#         urls = [
#         "https://www.altcoinstalks.com/index.php?topic=79189.0",
#         ]
#         for url in urls:
#             yield scrapy.Request(url=url, callback=self.parse)

#     def parse(self, response):
#         def arr_to_str(arr):
#             string = ""
#             for f in arr:
#                 string = string +  f"{f} "
#             string = string[:-1]
#             return string 
        
#         if self.isMax:
#             self.isMax = False


#         topic = response.css('li.last').css("span::text").get()
#         for post in response.css("div.postarea"):
#             # Introduce delay so hopefully they don't think it's DDOS
#             text = post.css("div.inner::text").getall()
#             text = arr_to_str(text)

#             date = post.css("div.smalltext::text").getall()[-1].split(" ")[:-1]
#             date = arr_to_str(date)[1:]

#             yield {'date': date, 'topic': topic, 'text': text,}

#         if not self.isMax:
#             page_list = response.css("a.navPages")#.getall()
#             page_list = page_list[:int(len(page_list)/2)]
#             next_page = page_list[self.start_at - 1].css("a.navPages::attr(href)").get()

#         # next_page = response.css("span.prevnext")[1].css("a.navPages::attr(href)").get()
#         # As long as there is a next page, keep searching.
#         if self.start_at != self.max:
#             time.sleep(1)
#             yield response.follow(next_page, callback=self.parse)

#         else:
#             self.isMax == True
#             yield response.follow(next_page, callback=self.parse)


# class QuotesSpider(scrapy.Spider):
#     name = "quotes"

#     def start_requests(self):
#         urls = [
#             'http://quotes.toscrape.com/page/1/',
#             'http://quotes.toscrape.com/page/2/',
#         ]
#         for url in urls:
#             yield scrapy.Request(url=url, callback=self.parse)

#     def parse(self, response):
#         page = response.url.split("/")[-2]
#         filename = 'quotes-%s.html' % page
#         with open(filename, 'wb') as f:
#             f.write(response.body)
#         self.log('Saved file %s' % filename)


# class TestSpider(scrapy.Spider):
#     name = "test"
#     start_urls = [
#         'https://coinmarketcap.com/'
#         'https://coinmarketcap.com/currencies/bitcoin/',
#     ]

#     def parse(self, response):
#         for something in response.css('f6l7tu-0'):
#             yield {
#                 'price': something.css("span.cmc-details-panel-price__price::text").get(),
                
#             }

#         next_page = response.css('li.next a::attr(href)').get()
#         if next_page is not None:
#             yield response.follow(next_page, callback=self.parse)

#get responses: a = response.css("div.rbayaw-0")[0]      

# get links:links = a.css("a.cmc-link")   

# get names: name = .links[8].css('[class^="Text-"]').get()
# start 0, iterate by 4. name.split(">")[-2].split("<")[0]
# 




# scrapy crawl quotes -o quotes.json
#rc-table-row rc-table-row-level-0 cmc-table-row
# rc-table-cell nameTHeader___1_bKM forced_name_font_size___3lG3U rc-table-cell-fix-left rc-table-cell-fix-left-last

#cmc-link
#Box-sc-16r8icm-0 CoinItem__Container-sc-1teo54s-1 oRQFi

# %%
# class QuotesSpider(scrapy.Spider):
#     name = "quotes"
#     start_urls = [
#         'http://quotes.toscrape.com/page/1/',
#         'http://quotes.toscrape.com/page/2/',
#     ]

#     def parse(self, response):
#         page = response.url.split("/")[-2]
#         filename = 'quotes-%s.html' % page
#         with open(filename, 'wb') as f:
#             f.write(response.body)


#response.css("div.quote")[0]

# quote = response.css("div.quote")[0]
# quote.get() shows you the different spans, like text.
# To get the text do: quote.css("span.text::text").get()

