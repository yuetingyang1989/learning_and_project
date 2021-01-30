# Topic Detection from Customer Comments with LDA

LDA is a classical method for topic modelling, and it is widely applied due to the fast speed. 

The performance is LDA is evaluated by coherence score. 
* According to industry standard, the coherence score :
    * 0.3 is bad
    * 0.4 is low
    * 0.55 is okay
    * 0.65 might be as good as it is going to get
    * 0.7 is nice
    * 0.8 is unlikely
    

* I reached 0.56 coherence score, but the result is not very good, because:
    * the length of the text is too short, most of the comment is less than 50 words, while LDA works well with text >= 100 words
    * the topic clusters are very much overlapped
    * the keywords per topic are not forming a good story and they don't fit with the top 10 sentences per topic 
    
LDA is very dependent on how the user clean the data and interpreting the results. 
LDA is not suitable for text without much variaties and not for text without much named entities. 