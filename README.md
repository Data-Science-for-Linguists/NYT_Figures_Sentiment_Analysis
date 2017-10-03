# Christopher Lagunilla LING 1340 Term Project 

### TIME Magazine: Measuring Sentiment Related to Historical Figures

#### SUMMARY
> I was looking to use the TIME Magazine archive (from recent years) to scan for proper nouns and also grab the text that 
surrounds specific proper nouns, namely famous figures. After parsing out the data, it should show the sentiment toward 
certain figures as being favorable or unfavorable.

#### DATA SOURCE  
> The data would be sourced from the TIME archive. I would only use ones from the past decade, to keep the corpus 
manageable. After scanning, I would maintain data structures that map the name of a famous figure to sentences that 
refer specifically to them. Additionally, each figure would receive a sentiment score. Additionally, there would be 
additional fields where a particular figure would receive sentiment scores across multiple years if they are mentioned 
multiple times across multiple issues.

#### ANALYSIS  
> I'm hoping to bring in other corpora using nltk to assist with the sentiment analysis. Additionally, I may have to 
manually take tallies of specific vocabulary as being particularly positive or negative to use. The end goal is to be 
able to produce output mapping the rise and fall of the sentiment for particular people across a number of years.

#### PRESENTATION
> I think the primary component of this would be the visuals since the analysis would look into the change of sentiment 
over time.