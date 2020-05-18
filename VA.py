import xml.etree.ElementTree as ET
description = '/root/yangsen-data/LIRIS-ACCEDE-data/ACCEDEdescription.xml'
root = ET.parse(description).getroot()
def getRating():
    ranking_file = '../LIRIS-ACCEDE-annotations/annotations/ACCEDEranking.txt'
    import pandas as pd
    ranking = pd.read_csv(ranking_file, delimiter='\t')
    return ranking
rating = getRating()

def getAffectiveValue(name):
    return [rating[rating['name'] == name]['valenceValue'].values[0],
            rating[rating['name'] == name]['arousalValue'].values[0]]

# get clips of each move from description file
movies = {}
for media in root.findall('media'):
    idd = media.find('id')
    name = media.find('name')
    m = media.find('movie')
    if m.text not in movies:
        movies[m.text] = []
    movies[m.text].append(name.text)

def getVaSequence(movie_name: str) -> list:
    ans = []
    for clip in movies[movie_name]:
        ans.append(getAffectiveValue(clip))
    return ans

# get VA curve from label file
VAlines = {}
for m in movies:
    VAlines[m] = getVaSequence(m)
print(VAlines)
