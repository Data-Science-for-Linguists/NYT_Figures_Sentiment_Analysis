import xml.etree.ElementTree as et
import glob
import pandas as pd
import pickle


# create new dataframe with empty data
columns = ['DOCID', 'Date', 'Month', 'Year', 'Name', 'Text']
data = pd.DataFrame(columns=columns)

# create year variable
# use this to select the year you want to save
year = "2006"

# open each xml file in the specified folder, open it and print out the names of mentioned people
for file in glob.glob("../data/NYT Corpus/nyt_corpus/data/" + year + "/*/*/*.xml"):
    # parse the xml file into an element tree to extract data
    tree = et.parse(file)
    root = tree.getroot()

    # get document id information (not sure if I need this yet, seems like it could be helpful)
    docid = root.find('.//doc-id[@id-string]').attrib['id-string']

    # get publication date information
    date = root.find(".//meta[@name='publication_day_of_month']").attrib['content']
    month = root.find(".//meta[@name='publication_month']").attrib['content']
    year = root.find(".//meta[@name='publication_year']").attrib['content']

    # get article text information
    # some articles seem to lack text - this is caught and handled in the if/else
    article = root.find(".//block[@class='full_text']/p")
    if article is not None:
        text = article.text.lower()
    else:
        text = None

    # for each person mentioned, create a new row of data for them in the dataframe
    for c in root.iter('person'):
            name = str(c.text).upper()
            data = data.append([{'DOCID': docid, 'Date': date, 'Month': month, 'Year': year, 'Name': name, 'Text': text}])

    print("PROCESSED: ", file)

# PICKLE THE FILE TO SAVE IT
pickle.dump(data, open("nyt-" + year + ".p", "wb"))