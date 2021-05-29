from bs4 import BeautifulSoup
import requests

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36',
        'Accept-Encoding': 'identity'
    }

datafile = open("dataSet.txt", "a")

for i in range(1,6):
    url = 'https://www.realtor.com/realestateandhomes-search/San-Antonio_TX/price-220000-1000000/sby-8/pg-'+str(i)

    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, 'lxml')

    for item in soup.select('.component_property-card'):
        try:
            datafile.writelines(item.select('[data-label=pc-price]')[0].get_text()+","+item.select('.prop-meta')[0].get_text()+"\n")

        except Exception as e:
            print("er")
datafile.close()


datafile = open("dataSet.txt", "rt")
fout1 = open("dataSet2.txt", "wt")
for line in datafile:
    fout1.write(line.replace('$', '')
                .replace(line[4], '')
                .replace('bed', ',')
                .replace('bath', ',')
                .replace('sqft lot', '')
                .replace('sqft', ',')
                .replace('acre lot', ''))

datafile.close()
fout1.close()
