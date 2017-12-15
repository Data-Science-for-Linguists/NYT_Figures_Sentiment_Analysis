# Sentiment Analysis of Figures in the New York Times
Christopher Lagunilla
---

## Introduction  
This project seeks to find summary statistics and descriptive data related to sentiment. Sentiment is an interesting idea to explore since it can be closely related to time and historical events and thus can fluctuate throughout a year, or over many years. Additionally, different people will have different sentiments about them depending on what kinds of stories they happen to be mentioned in. By looking at editorial and opinion articles, we can look at what ways journalists are presenting particular figures.  

The code explores a number of questions discussed below regarding prominent figures from the early 2000s, using sentiment analysis tools to answer them.

## Data Source
The data source used for this project is the New York Times Annotated Corpus, licensed by the Linguistic Data Consortium. The files, having already been nicely cleaned and noted, made data collection more simple than to scrape the internet for similar data. Thus, my tasks related to the data involved getting data from a well-defined data structure using additional Python libraries.

The original plan of the project was to utilize Times magazine rather than the New York Times. However, the articles from Times magazine were not collected as a corpus, and to acquire them over time required multiple payments. Thus, the project plan was amended to use the NYT corpus. This change actually helped the project overall, in that it would be easier to define opinion based articles to use for analysis, due to the well-defined structure.

## Data Capturing
The data capturing process occurs in `unpack.ipynb`. The first task was to leverage Python libraries to parse the .xml files into a usable object which can be traversed. This made capturing particular fields of data quite simple using xpaths. Some of the important information that was captured from each file included: day, month, year, article text, names mentioned, and topic information. Each of these entries (one for each person mentioned) were appended to a regular Python dictionary, and later converted to a Pandas DataFrame for easier processing later.

> One of the issues faced in the beginning was that of efficiency and code execution time. The original implementation of the code created an empty DataFrame, and appended each row to it as they were captured from the XML files. This proved to be quite slow, and unreasonable to continue using. After looking into the implementation of the Pandas append() function, I learned that DataFrames are immutable, and executing `df = df.append(...)` would repeatedly create new DataFrames, wasting time and memory. Thus, the solution to store information as a dictionary object first and convert to a DataFrame later proved to be much more efficient.

Once the DataFrame is created, the file is serialized into a file using the `pickle` library, allowing for easier access.

The final step to capturing the appropriate data was to use the `glob` library in order to loop through _all_ the relevant XML files using regular expressions. Instead of having to run the `unpack.ipynb` code for each file (of which there are many), it can be run once, with defined parameters regarding what days, months, or years are to be captured.

> Another issue that came up during the capturing process is regarding the files contained within the year 2006. The code does not seem to recognize the path to the files contained in the 2006 folder, however the code works for every other year. Thus files from the year 2006 are not present in the analysis.


## Data Processing


## Analysis
__1. Who are the Top 10 mentioned people?__  
__2. What topics are associated with positive articles? What topics are negative articles?__  
__3. What words are associated with positive articles? What words are associated with negative words?__  
__4. Are there particular months that harbor a particular sentiment?__

## Future Exploration
