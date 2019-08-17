from collections import namedtuple

from google_images_download import google_images_download
from multiprocessing import Pool

GIParams = namedtuple('GIParams', ['keyword', 'count', 'path'])


def load_images(params: GIParams):
    try:
        response = google_images_download.googleimagesdownload()
        arguments = {
            "keywords": params.keyword,
            "size": ">400*300",
            "format": "jpg",
            "limit": params.count,
            "output_directory": params.path,
            "no_directory": True
        }
        response.download(arguments)
    except Exception:
        pass


def load_images_multiprocessing(output_directory: str, process_count: int = 4, batch_size: int = 2):
    keywords = get_keywords()
    args = (
        GIParams(keyword, batch_size, output_directory)
        for keyword in keywords
    )
    with Pool(processes=process_count) as pool:
        pool.map(load_images, args)


def get_keywords():
    prefixes = 'chinese,asian,japanese,korean,thai,'.split(',')
    words = 'journal,journals,news,newspaper,newspapers,text,texts,sting,stings,content,' \
            'contents,document,documents,paragraph,paragraphs,characters,letter,letters,' \
            'alphabet,words,lines,manual,guide,gazette,magazine,note,paper,review,press'.split(',')
    words = list(set(words))
    return [
        ('%s %s' % (prefix, word)).strip()
        for prefix in prefixes
        for word in words
    ]


if __name__ == '__main__':
    load_images_multiprocessing(
        output_directory='../data/images',
        process_count=8,
        batch_size=100
    )
