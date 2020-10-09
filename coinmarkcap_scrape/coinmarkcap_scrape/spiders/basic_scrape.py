#%%

import scrapy


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


class TestSpider(scrapy.Spider):
    name = "test"
    start_urls = [
        'https://coinmarketcap.com/currencies/bitcoin/',
    ]

    def parse(self, response):
        for something in response.css('f6l7tu-0'):
            yield {
                'price': something.css("span.cmc-details-panel-price__price::text").get(),
                
            }

        next_page = response.css('li.next a::attr(href)').get()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)

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

