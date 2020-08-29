import requests
import csv
import argparse
from tqdm import tqdm
import hashlib
from urllib import parse as urlparse
import urllib
import os
from PIL import Image
import aiohttp
import asyncio
from contextlib import closing

parser = argparse.ArgumentParser(description='Download data from WGA')
parser.add_argument('--csvfile', default="data/wga/catalog.csv", help='.csv from WGA with description')
parser.add_argument('--destination', default="data/wga")
args = parser.parse_args()

column2ind = {
    "url": -5,
    "date": 3
}

def format_date(date_str):
    date_str = date_str.lower()
    original = date_str
    if len(date_str) == 0 or date_str == '15%-72':
        return None
    date_str = date_str.replace(', etten', '').replace(', the hague', '').replace(', paris', '')
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
            'august', 'september', 'october', 'november', 'december', 'summer',
            'winter', 'autumn', 'spring', 'second half of ', 'febrauary']
    for el in months:
        date_str = date_str.replace(el, '')
    if date_str == '1838 (1862)':
        return 1838
    if date_str == '1505 (completed 1508)':
        return 1505
    if date_str == 'established in 1228':
        return 1228

    for i in range(4, 21):
        if date_str == f"{i}th century":
            return i * 100 - 50
        if date_str == f"late {i}th century":
            return i * 100 - 25
        if date_str == f"early {i}th century":
            return i * 100 - 75

    if date_str == '3rd century':
        return 250
    elif date_str == '2nd century':
        return 150
    elif date_str == 'late 3rd century':
        return 275
    if date_str == '11th-12th centuries':
        return 1101
    if date_str == '12th-13th centuries':
        return 1201
    elif date_str == '11th-13th centuries':
        return 1201
    if date_str == '-':
        return None

    date_str = date_str.replace('c.', '').replace('c,', '').replace('c-', '')
    sep = ['and', '-', '–', '/', ',']
    for el in sep:
        if el in date_str:
            date_str = list(date_str.split(el))[0]

    date_str = date_str.replace('s', '').replace(' ', '')
    date_str = date_str.replace('.', '').replace('before', '').replace('(completed)', '') \
                        .replace('(etup)', '').replace('after', '').replace('begun', '') \
                        .replace('began', '').replace('(retored)', '') \
                        .replace('founded', '').replace('conecrated', '') \
                        .replace('from', '').replace('around', '') \
                        .replace('completed', '').replace('rebuilt', '') \
                        .replace('oon', '').replace('converted', '') \
                        .replace('planned', '').replace('about', '') \
                        .replace('(model)', '').replace('(', '').replace('c', '') \
                        .replace('?)', '').replace('late', '').replace('onward', '')
    if 'or' in date_str:
        date_str = date_str.split('or')[0]
    if date_str == '':
        return None
    try:
        return int(date_str)
    except Exception as exp:
        print(original, '-' in date_str, list(date_str.split('-'))[0], len(original))
        return None # raise exp

table = []

with open(args.csvfile, newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    is_header = True
    for row in tqdm(spamreader):
        if is_header:
            is_header = False
            continue
        date = format_date(row[column2ind['date']])
        if date is None:
            continue
        table.append((row[column2ind['url']],
                      date,
                      row[column2ind['date']]
                     ))

print(f'Size: {len(table)}')
print(table[:3])

def url_converter(url):
    return url.replace('html/', 'detail/').replace('.html', '.jpg')

def get_filename(url):
    path = urlparse.urlparse(url).path
    ext = os.path.splitext(path)[1].split('.')[1]
    name = hashlib.md5(url.encode('utf-8')).hexdigest() + '.' + ext
    return name

cnt = 0

def retrieve_wrapper(data):
    urllib.request.urlretrieve(data[0], data[1])

async def download_image(url, name, pbar):
    global cnt
    path = os.path.join(args.destination, 'art', name)
    if not os.path.exists(path):
        print(url)
        await loop.run_in_executor(None, retrieve_wrapper, (url, path))
    pbar.update(1)
    cnt += 1
    print(cnt / len(urls_to_download), flush=True)

os.system(f'mkdir {os.path.join(args.destination, "art")}')

urls_to_download = []

with open(os.path.join(args.destination, 'prepared.csv'), 'w', newline='') as file:
    spamwriter = csv.writer(file)
    for row in tqdm(table):
        url = url_converter(row[0])
        name = get_filename(url)
        date = row[1]
        urls_to_download.append((url, name))
        spamwriter.writerow([name, date, row[2]])


'''async def download_file(url: str, name: str):
    path = os.path.join(args.destination, 'art', name)
    if os.path.exists(path):
        return

    async with session.get(url) as response:
        try:
            assert response.status == 200
        except Exception as exp:
            print(url, name)
            raise exp
        buff = await response.read()
        print(type(buff))
        img = Image.frombuffer(buff)
        img.save(path)'''

async def main():
    routines = []
    with tqdm(total=len(urls_to_download)) as pbar:
        for el in urls_to_download:
            routines.append(download_image(el[0], el[1], pbar))

    await asyncio.gather(*routines)
    #        download_image(*el) for el in urls_to_download
    #    )

'''async def factorial(name, number):
    f = 1
    for i in range(2, number + 1):
        print(f"Task {name}: Compute factorial({i})...")
        # await asyncio.sleep(1)
        f *= i
    print(f"Task {name}: factorial({number}) = {f}")

async def main():
    # Schedule three calls *concurrently*:
    await asyncio.gather(
        download_image(*urls_to_download[0]),
    )'''

loop = asyncio.get_event_loop()
# loop.run_until_complete(main())
loop.close()
