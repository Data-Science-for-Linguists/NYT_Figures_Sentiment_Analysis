import xml.etree.ElementTree as Et
import glob


for file in glob.glob("../data/NYT Corpus/nyt_corpus/data/2007/01/01/*.xml"):
    tree = Et.parse(file)
    root = tree.getroot()
    for c in root.iter('person'):
        print(str(c.text).upper())



