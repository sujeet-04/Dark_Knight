######################################################################################
#                Analysis of The dark knight from imdb reviews                       #
######################################################################################

library(RCurl)
library(rvest)
library(wordcloud)
library(tm)
library(wordcloud2)
library(textstem)
library(RWeka)

aurl <- "https://www.imdb.com/title/tt0468569/reviews?ref_=tt_ql_3"
IMDB_reviews <- NULL
for (i in 1:20){
  murl <- read_html(as.character(paste(aurl,i,sep="=")))
  rev <- murl %>%
    html_nodes(".show-more__control") %>%
    html_text()
  IMDB_reviews <- c(IMDB_reviews,rev)
}
length(IMDB_reviews)
View(IMDB_reviews)

movie_rec <- iconv(IMDB_reviews,to="UTF-8")

movie_rec <- Corpus(VectorSource(movie_rec)) #Transform the text into vector source
class(movie_rec)
inspect(movie_rec)

movie_rec <- tm_map(movie_rec,tolower) # Transform the uppercase data into lower case
inspect(movie_rec)

movie_rec <- tm_map(movie_rec,removeNumbers) #Remove the numbers from the speech.
inspect(movie_rec)

movie_rec <- tm_map(movie_rec,removeWords,stopwords("english")) #Remove all the stopwords
inspect(movie_rec)

movie_rec <- tm_map(movie_rec,removeWords,c("the","that","was","have","there","dark","film",
                                          "knight","tdk","movi","can","its","could","has","this"))
movie_rec <- tm_map(movie_rec,stemDocument)#generate root from the inflected word
inspect(movie_rec)

movie_rec <- tm_map(movie_rec,lemmatize_strings)
inspect(movie_rec)

movie_rec <- tm_map(movie_rec,removePunctuation)
inspect(movie_rec)

movie_rec <- tm_map(movie_rec,stripWhitespace)
inspect(movie_rec)

#create a term document matrix

tdm_movie <- TermDocumentMatrix(movie_rec)
findFreqTerms(tdm_movie,lowfreq = 5) # To See all the terms that have frequency more than 5

movie_matrix <- as.matrix(tdm_movie) #Transform into matrix
View(movie_matrix)

word_freq <- rowSums(movie_matrix) #sum out the total frequency of a particluar words
View(word_freq)

word_freq2 <- subset(word_freq,word_freq>200) #subset the words that have frequency more than 10 
View(word_freq2)

barplot(word_freq2,las=2,col=rainbow(7))

movie_data <- data.frame(movie_matrix) #Transform the matrix into data frame.
View(movie_data)

word_freq3 <- sort(rowSums(movie_data),decreasing = TRUE) #sort out the words in decreasing order according to frequency.

wordcloud(words = names(word_freq3),freq = word_freq3,min.freq = 10,
          random.order = F,colors = rainbow(5))

word_freq4 <- data.frame(rownames(movie_matrix),rowSums(movie_matrix))
colnames(word_freq4) <- c("Word","Frequency")
View(word_freq4)

wordcloud2(word_freq4,size = 0.5,minSize =0.5,
           shape = "star",color = rainbow(50))

#In the wordcloud the most common words are batman,joker,harvey,heath,ledger,nolan,
#gotham,and bale which are the characters and name of the cast of movie apart from that
#good and great words are also used frequently which talk about the movie performance.
#there are so many other words are there to get deeper understanding lets build a bigram
#and trigram

#Try to build a matrix for combination of words and visualize it

#For 2 words :-
bigram <- NGramTokenizer(movie_rec,Weka_control(min=2,max=2)) #Create a bigram containing two words used together.
View(bigram)
bigram_data <- data.frame(table(bigram))#Transform into data frame
View(bigram_data)

bigram_2 <- bigram_data[order(bigram_data$Freq,decreasing = TRUE),]#sort it into decending order

wordcloud(bigram_2$bigram, bigram_2$Freq, scale = c(1,1),
          random.order = F,  min.freq =30,color = rainbow(10))

wordcloud2(bigram_data,size=1,minSize=1,shape="star",color = rainbow(10))

#For 3 words together
trigram <- NGramTokenizer(movie_rec,Weka_control(min=3,max=3))
trigram_data <- data.frame(table(trigram))

wordcloud(trigram_data$trigram,trigram_data$Freq,scale = c(1,0.5),random.color = F,
          min.freq = 20,color = rainbow(30))

wordcloud2(trigram_data,size=0.5,minSize=0.5,shape="star",color = rainbow(10))

############### Sentiment Analysis on the review ###############

#Here we try to extract the seniment from the review and analyze it.
#there are so many sentiment like anger,disgust,joy,fear and much more, we try
# to divide our review data into these category .

library(syuzhet)
library(lubridate)
library(scales)
library(reshape2)
library(dplyr)


sentment <- get_nrc_sentiment(IMDB_reviews)
head(sentment)

barplot(colSums(sentment),las=2,col = rainbow(10),ylab = 'count')
#From the barplot we can say that there are so many postive words are in the reviews 
#about the movie,apart from that all emotion have equal share it means the movie
#appreciate by the people and most of the people say good things about the movie.
# Now we can go further deep into our analysi and can see what are the most used positive and negative words.

sent <-  get_sentences(IMDB_reviews)#Get all the reviews into the sent as a character type
View(sent) #we got 9160 sentences
class(sent) #character type
str(sent)
sent[2]
#Analyzing the sentiment by bing method
#In this method we analyze each sentence and score it the bing lexicon transform
#the sentence into binary and score it,for negative values less than zero and for 
#positive value greater than 0 assigned.

sentiment_vector <- get_sentiment(sent, method = "bing")
sentiment_vector
range(sentiment_vector)
View(sentiment_vector)
head(sentiment_vector)
#Visualization of all the sentences and there scores.

plot(sentiment_vector, type = "l", main = "Plot Trajectory",
     xlab = " Sentences ", ylab = "Emotion scores")
abline(h = 0, col = "red")


#From the top 5 sentence 1st sentance have a score of 2 and the fourth one have a 
#score of -2 and all other are 0,zero means neutral.
#as we got the score for each sentences, now we can separately produce two
# set one from positive words and another for negative words ,and we can see from 
#wordplot the frequency of the positive and negative words.

#For positive sentences

positive_vector <- which(sentiment_vector>0)
head(positive_vector)#these are the sentence number that contains the positive sentence

positive_sentence <- sent[positive_vector] #collection of all positive sentence.
View(positive_sentence)
class(positive_sentence)

positive_sentence2 <- Corpus(VectorSource(positive_sentence)) #Transform into corpus

positive_tdm <- TermDocumentMatrix(positive_sentence2)#create a term doccument matrix
positive_tdm

positive_matrix <- as.matrix(positive_tdm)# Transform into matrix
head(positive_matrix)

positive_data <- data.frame(rownames(positive_matrix),rowSums(positive_matrix))#create a data set containing words and its frequency.
colnames(positive_data) <- c("Word","Frequency")

wordcloud(words = positive_data$Word,freq = positive_data$Frequency,random.order = F,
          colors = rainbow(10),min.freq = 30)

wordcloud2(positive_data,size = 1,minSize = 1,shape = "star",color = rainbow(7))

#For Negative sentences

Negative_vector <- which(sentiment_vector<0)
head(Negative_vector)#these are the sentence number that contains the positive sentence

Negative_sentence <- sent[Negative_vector] #collection of all positive sentence.
View(Negative_sentence)
class(Negative_sentence)

Negative_sentence2 <- Corpus(VectorSource(Negative_sentence)) #Transform into corpus

Negative_tdm <- TermDocumentMatrix(Negative_sentence2)#create a term doccument matrix
Negative_tdm

Negative_matrix <- as.matrix(Negative_tdm)# Transform into matrix
head(Negative_matrix)

Negative_data <- data.frame(rownames(Negative_matrix),rowSums(Negative_matrix))#create a data set containing words and its frequency.
colnames(Negative_data) <- c("Word","Frequency")

wordcloud(words = Negative_data$Word,freq = Negative_data$Frequency,random.order = F,
          colors = rainbow(10),min.freq = 2)

wordcloud2(Negative_data,size = 1,minSize = 1,shape = "star",color = rainbow(7))

#We can also fet the most positive and negative sentence
# To extract the sentence with the most negative emotional valence
negative <- sent[which.min(sentiment_vector)]
negative

# and to extract the most positive sentence
positive <- sent[which.max(sentiment_vector)]
positive

#### Afinn method
#in this method the word is rated between -5 and 5
afinn_s_v <- get_sentiment(sent, method = "afinn")
View(afinn_s_v)
head(afinn_s_v)
range(afinn_s_v)


