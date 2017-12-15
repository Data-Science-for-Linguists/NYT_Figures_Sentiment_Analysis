# Progress Report 1
- I started to small-scale process the xml files in my data. Because the corpus is already well annotated XML files, it was more of a matter of reading in the data that I need and storing the relevant information in a DataFrame so that I can work with only the information I need. The DataFrame is still a WIP and more information will be appended to it. I could also create multiple DataFrames with separate information. They would be linked by their index?

# Progress Report 2
- I started looking into functions to help with sentiment analysis, and found that NLTK has a sentiment analysis package called VADER. It provides a polarity score for a given sentence along 4 variables: compound, negative, neutral, and positive. I think I can use this in my overall analysis since it provides numerical feedback.  

- I also had to stop what I was doing and rethink the direction I am going to take my project. Originally, I wanted to get polarity scores for particular people across a number of months and years. However, I realized that because my source is a newspaper, my texts are unlikely to be overwhelmingly positive or negative because of the medium that is a newspaper. I think I have to calculate my values based on the intensity from zero, compared to neutral as a baseline instead.

- I also scaled up the program to 1 month's worth of data to be able to start exploring looking at people over time. This is within my data_processing.ipynb file.

- I also pulled the code from data_processing into a new ipynb and modified it in order to take a year's worth of data and to pickle the resulting dataframe so that I dont have to do it every single time I run my program.

# Progress Report 3
- Now I am digging deeper and refining what exactly it is I want to do as far as analysis goes. I wrote a utility script to extract my data of interest from the NYT corpus for the year of 2007. I am using this as a start before I unpack the entire corpus in the interest of time and space.

- This also involved additional manipulation of my dataframes in order to make it more useful for what I want to do, as well as make it faster. In this process of refining the analysis process, it is also giving me a chance to figure out ~how~ exactly these structures should be saved in order to save time in file reading.

- The plan for the final submission is to have unpacked the entire corpus and scale the code and structures up to accommodate all of the data.

- ISSUE: One issue that I ran into was that my utility script, which worked to unpack the year 2007, no longer works on other years in the corpus due to xpathing in the xml files. I have to figure out what is going on with this in order to complete the code.

- I separated my code into two folders: jupyter_notebooks and src. This is for organization purposes to separate my utility scripts from the code I wish to present.

# Final Progress Report
- Having presented what I have so far, I am able to take the knowledge gained from that to be able to write the code for more pointed research questions.
- I ultimately decided to focus on the years from 2000-2007 for times sake. Having to unzip and process all of the years of the corpus would have increased the time to process and analyze, however doing so would be very interesting to see in the code.
