import re

from scrapy.crawler import CrawlerProcess
from scrapy.spiders import Spider


class MySpider(Spider):
    name = 'public.oed.com'
    allowed_domains = ['public.oed.com']
    start_urls = ['http://public.oed.com/how-to-use-the-oed/abbreviations/']

    def parse(self, response):
        for tr in response.xpath('//*[@id="content"]/div/table/tbody/tr[not(td/@colspan = \'2\')]'):
            abbreviation, phrase = tr.xpath('td/text()').extract()
            for phrase in phrase.split(', '):
                phrase_variants = MySpider.generate_phrase_variants(list(filter(None, re.split(r'(\()|\)', phrase))))
                for phrase_variant in phrase_variants:
                    yield {'Abbreviation': abbreviation, 'Phrase': phrase_variant.strip()}

    @staticmethod
    def generate_phrase_variants(tokens):
        phrase_variants = list()
        phrase = ''
        for i, token in enumerate(tokens):
            if token is '(':
                optional_token = tokens[i + 1]
                further_tokens = tokens[i + 2:]
                further_phrase_variants = MySpider.generate_phrase_variants(further_tokens)
                phrase_variants += [phrase + phrase_variant for phrase_variant in further_phrase_variants]
                phrase_variants += [phrase + optional_token + phrase_variant for phrase_variant in further_phrase_variants]
                break
            else:
                phrase += token
        else:
            phrase_variants += [phrase]
        return phrase_variants

process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    'FEED_FORMAT': 'csv',
    'FEED_URI': './data/external/abbreviations.csv',
    'FEED_EXPORT_FIELDS': ['Phrase', 'Abbreviation']
})

process.crawl(MySpider)
process.start()