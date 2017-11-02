<!-- Progress Report Information Goes Here -->
# Progress Report 1
- I started to small-scale process the xml files in my data. Because the corpus is already well annotated XML files, it was more of a matter of reading in the data that I need and storing the relevant information in a DataFrame so that I can work with only the information I need. The DataFrame is still a WIP and more information will be appended to it. I could also create multiple DataFrames with separate information. They would be linked by their index?

# Progress Report 2
- I started looking into functions to help with sentiment analysis, and found that NLTK has a sentiment analysis package called VADER. It provides a polarity score for a given sentence along 4 variables: compound, negative, neutral, and positive. I think I can use this in my overall analysis since it provides numerical feedback.  

- I also had to stop what I was doing and rethink the direction I am going to take my project. Originally, I wanted to get polarity scores for particular people across a number of months and years. However, I realized that because my source is a newspaper, my texts are unlikely to be overwhelmingly positive or negative because of the medium that is a newspaper. I think I have to calculate my values based on the intensity from zero, compared to neutral as a baseline instead.

- I also scaled up the program to 1 month's worth of data to be able to start exploring looking at people over time. This is within my data_processing.ipynb file.

- I also pulled the code from data_processing into a new ipynb and modified it in order to take a year's worth of data and to pickle the resulting dataframe so that I dont have to do it every single time I run my program.
