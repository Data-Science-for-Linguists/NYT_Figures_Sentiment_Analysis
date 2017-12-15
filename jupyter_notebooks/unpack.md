
# Unpack

`unpack.ipynb` is a utility used to process the NYT Annotated Corpus' XML files to extract particular tags of information, and append it to a DataFrame. A file is generated from this script that will be used for further processing.  

  
  The files produced from the script are used in later ones ( `processing.ipynb` and `analysis.ipynb` )
  
  Running time for the script can be lengthy depending on the values entered for year, month, and date.
  
  ---
  
### Table of Contents  
- [Import Libraries](#Import-Libraries)
- [Create a List to Collect Data](#Create-a-List-to-Collect-Data)
- [Set the Dates of Desired Files](#Set-the-Dates-of-Desired-Files)
- [Begin Processing Files from NYT Annotated Corpus](#Begin-Processing-Files-from-NYT-Annotated-Corpus)
- [Create and Sort the DataFrame](#Create-and-Sort-the-DataFrame)
- [Verify that the DataFrame is Sorted at the Beginning](#Verify-that-the-DataFrame-is-Sorted-at-the-Beginning)
- [Verify that the DataFrame is Sorted at the End](#Verify-that-the-DataFrame-is-Sorted-at-the-End)
- [Write Out the Resulting DataFrame to a File](#Write-Out-the-Resulting-DataFrame-to-a-File)

### Import Libraries
The XML library is used to parse and traverse the .xml files provided in the corpus.  
The glob library is used to be able to find files using regular expressions to loop through multiple files.  
The pandas library is used to hold all of the information that is extracted from the corpus.  
The pickle library is used to serialize the DataFrame object into a file, to be loaded and used by another script.


```python
import xml.etree.ElementTree as Et
import glob
import pandas as pd
import pickle
```

### Create a List to Collect Data
Appending large data to a list and then converting it to a DataFrame has proven _MUCH_ faster than appending rows to a DataFrame directly.


```python
gather_data = []
```

### Set the Dates of Desired Files
The variables used below are able to be modified in order to determine what month, day, and year to extract and process files from.  
All values are in numerical format. Single digit values are expressed as `01`, `02`, `...`, `09`.  
If you wish to use all of a specific type of value, use the `*` instead of a number.


```python
year = "2007"
month = "*"
day = "*"
```

### Begin Processing Files from NYT Annotated Corpus
Using the values for year, month, day, the glob library is able to get the file names that match a particular path, represented as a regular expression.  
  
The data being extracted are:
- docid
- date
- month
- year
- identified name

  
Values are stored in a DataFrame called `data`.  


```python
# open each xml file in the specified folder, open it and print out the names of mentioned people
for file in glob.glob("../data/NYT Corpus/nyt_corpus/data/"+year+"/"+month+"/"+day+"/*.xml"):
    # parse the xml file into an element tree to extract data
    tree = Et.parse(file)
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
        text = (article.text).lower()
    else:
        text = None
        
    # get all of the classifer information
    doctypes = ""
    for d in root.iter('classifier'):
        doctypes += str(d.text).upper() + "|"
        
    # for each person mentioned, create a new row of data for them in the dataframe    
    for c in root.iter('person'):
        name = str(c.text).upper()
        cur = [docid, date, month, year, name, text, doctypes]
        gather_data.append(cur)
```

### Create and Sort the DataFrame
This creates a new, empty DataFrame to read in information from the NYT Annotated Corpus.  
For readability, the DataFrame is sorted below by Month and then by Date. The minimum preferred granularity for processing files is by year. Any larger than that and the script would take too long to execute.


```python
columns = ['DOCID', 'Date', 'Month', 'Year', 'Name', 'Text', 'Doctypes']
data = pd.DataFrame(gather_data, columns=columns)
data = data.sort_values(ascending=[True, True], by=['Month', 'Date'])
```

### Verify that the DataFrame is Sorted at the Beginning


```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DOCID</th>
      <th>Date</th>
      <th>Month</th>
      <th>Year</th>
      <th>Name</th>
      <th>Text</th>
      <th>Doctypes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47030</th>
      <td>1815832</td>
      <td>1</td>
      <td>1</td>
      <td>2007</td>
      <td>BROWN, CLIFTON</td>
      <td>while the green bay packers wonder about brett...</td>
      <td>FOOTBALL|TOP/NEWS|TOP/NEWS/SPORTS|TOP/NEWS/SPO...</td>
    </tr>
    <tr>
      <th>47031</th>
      <td>1815832</td>
      <td>1</td>
      <td>1</td>
      <td>2007</td>
      <td>FAVRE, BRETT</td>
      <td>while the green bay packers wonder about brett...</td>
      <td>FOOTBALL|TOP/NEWS|TOP/NEWS/SPORTS|TOP/NEWS/SPO...</td>
    </tr>
    <tr>
      <th>47032</th>
      <td>1815832</td>
      <td>1</td>
      <td>1</td>
      <td>2007</td>
      <td>GROSSMAN, REX</td>
      <td>while the green bay packers wonder about brett...</td>
      <td>FOOTBALL|TOP/NEWS|TOP/NEWS/SPORTS|TOP/NEWS/SPO...</td>
    </tr>
    <tr>
      <th>47033</th>
      <td>1815826</td>
      <td>1</td>
      <td>1</td>
      <td>2007</td>
      <td>BOOTY, JOHN DAVID</td>
      <td>None</td>
      <td>FOOTBALL|ROSE BOWL (FOOTBALL GAME)|CAPTION|FOO...</td>
    </tr>
    <tr>
      <th>47034</th>
      <td>1815826</td>
      <td>1</td>
      <td>1</td>
      <td>2007</td>
      <td>HENNE, CHAD</td>
      <td>None</td>
      <td>FOOTBALL|ROSE BOWL (FOOTBALL GAME)|CAPTION|FOO...</td>
    </tr>
  </tbody>
</table>
</div>



### Verify that the DataFrame is Sorted at the End


```python
data.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DOCID</th>
      <th>Date</th>
      <th>Month</th>
      <th>Year</th>
      <th>Name</th>
      <th>Text</th>
      <th>Doctypes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>54782</th>
      <td>1853117</td>
      <td>9</td>
      <td>6</td>
      <td>2007</td>
      <td>PERKINS, THOMAS</td>
      <td>to the editor:</td>
      <td>ROADS AND TRAFFIC|TOLLS|LETTER|TOP/OPINION/OPI...</td>
    </tr>
    <tr>
      <th>54783</th>
      <td>1853117</td>
      <td>9</td>
      <td>6</td>
      <td>2007</td>
      <td>BLOOMBERG, MICHAEL R (MAYOR)</td>
      <td>to the editor:</td>
      <td>ROADS AND TRAFFIC|TOLLS|LETTER|TOP/OPINION/OPI...</td>
    </tr>
    <tr>
      <th>54784</th>
      <td>1853103</td>
      <td>9</td>
      <td>6</td>
      <td>2007</td>
      <td>ERLANGER, STEVEN</td>
      <td>the palestinian prime minister, ismail haniya ...</td>
      <td>PALESTINIANS|TOP/NEWS|TOP/NEWS/WORLD/COUNTRIES...</td>
    </tr>
    <tr>
      <th>54785</th>
      <td>1853103</td>
      <td>9</td>
      <td>6</td>
      <td>2007</td>
      <td>HANIYA, ISMAIL</td>
      <td>the palestinian prime minister, ismail haniya ...</td>
      <td>PALESTINIANS|TOP/NEWS|TOP/NEWS/WORLD/COUNTRIES...</td>
    </tr>
    <tr>
      <th>54786</th>
      <td>1853103</td>
      <td>9</td>
      <td>6</td>
      <td>2007</td>
      <td>HANIYA, ISMAIL</td>
      <td>the palestinian prime minister, ismail haniya ...</td>
      <td>PALESTINIANS|TOP/NEWS|TOP/NEWS/WORLD/COUNTRIES...</td>
    </tr>
  </tbody>
</table>
</div>



### Write Out the Resulting DataFrame to a File
The DataFrame is serialized below using the pickle library. The filename is taken from the `year` variable. Pickle files from this script carry the `.p` extension.


```python
pickle.dump(data, open("nyt-" + year + ".p", "wb"))
```
