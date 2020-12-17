#------------------------------------------------------------
# Name:   Ng Jing Xun (Jonas)
# Purpose:  BC3409 Practical Assessment
# Date:     31st March 2020 Submission
#------------------------------------------------------------

library(data.table)
library(rpart)
library(car)
library(magrittr)

setwd('C:/Users/Jing Xun/Desktop/3409 Practical Assessment/Merged Normalised Data STI Price')

#------------------------------------------------------------
# Linear Regression

# Import data set with continuous Y
data <- fread("STI_Normalized.csv")

# Remove 'Date' column 
data$Date <- NULL

# Correlation plot
library(corrplot)
corrplot(cor(data), type = "upper")

# Train-test split
library(caTools)
set.seed(2004)
train <- sample.split(Y = data$STI_Y, SplitRatio = 0.7)
trainset <- subset(data, train == T)
testset <- subset(data, train == F)
m1 <- lm(STI_Y ~ ., data = trainset)
summary(m1)

# Forward stepwise regression
m1.start <- lm(STI_Y ~ 1, data = trainset)
m1.step.forward <- step(m1.start, direction='forward', scope = formula(m1))
summary(m1.step.forward)

# Column 'SGDUSD', SGDIDR', 'SGDHKD' removed
m1.5 <- lm(STI_Y ~ EAFE + SP500 + WILSHIRE5000 + SGDEURO + DOW + NASDAQ + 
             GOLD_PRICE + SGDJPY + BARCLAY + CRUDEPETROLEUM + SGDMYR, data = trainset)
vif(m1.5) #huge multicollinearity with EAFE, S&P500, WILSHIRE, DOW, NASDAQ

# Remaining columns
m1.5.5 <- lm(STI_Y ~ SGDEURO + GOLD_PRICE + SGDJPY + BARCLAY + CRUDEPETROLEUM + SGDMYR, data = trainset)
summary(m1.5.5)

# Diagnostic plot
par(mfrow = c(2,2))  
plot(m1.5.5)  

# Check for overfitting
RMSE.m1.5.5.train <- sqrt(mean(residuals(m1.5.5)^2))  # RMSE on trainset based on m5 model.
summary(abs(residuals(m1.5.5)))  # Check Min Abs Error and Max Abs Error.

predict.m1.5.5.test <- predict(m1.5.5, newdata = testset)
testset.error <- testset$STI_Y - predict.m1.5.5.test

RMSE.m1.5.5.test <- sqrt(mean(testset.error^2))
summary(abs(testset.error))

#------------------------------------------------------------
# Non-linear Regression
m2 <- lm(STI_Y ~ BARCLAY + I(BARCLAY^2) + SGDEURO + GOLD_PRICE + SGDJPY + CRUDEPETROLEUM + SGDMYR, data = trainset)

# Since column 'SGDMYR' has no star, it is removed
m2.5 <- lm(STI_Y ~ BARCLAY + I(BARCLAY^2) + SGDEURO + GOLD_PRICE + SGDJPY + CRUDEPETROLEUM + SGDMYR, data = trainset)
summary(m2.5)

#------------------------------------------------------------
# Logistic Regression
# Import data set with categorical Y
data1 <- fread("STI_Normalized_CategoricalY.csv")

# Remove 'Date' column 
data1$Date <- NULL

# Train-Test split
set.seed(2014)
train <- sample.split(Y = data1$STI_Y_CAT, SplitRatio = 0.7)
trainset <- subset(data1, train == T)
testset <- subset(data1, train == F)

# Only columns 'BARCLAY', 'DOW', 'EAFE', 'SGDEURO', 'SP500' and 'WILSHIRE5000' are statistically important
m3 <- glm(STI_Y_CAT ~ . , family = binomial, data = trainset)
summary(m3)

# Only retain significant columns
m3.5 <- glm(STI_Y_CAT ~ BARCLAY + DOW + EAFE + SGDEURO + SP500 + WILSHIRE5000, family = binomial, data = trainset)
summary(m3.5)

# Column 'BARCLAY' is removed because it is not statistically important
m3.5.5 <- glm(STI_Y_CAT ~ DOW + EAFE + SGDEURO + SP500 + WILSHIRE5000, family = binomial, data = trainset)
summary(m3.5.5)

# Odds Ratio
OR <- exp(coef(m3.5.5))
OR

# Odds Ratio Confidence Interval
OR.CI <- exp(confint(m3.5.5))
OR.CI

# Confusion Matrix on Trainset
prob.train <- predict(m3.5.5, type = 'response')
predict.default.train <- ifelse(prob.train > 0.5, "Yes", "No")
table3 <- table(trainset$STI_Y_CAT, predict.default.train)
table3

#overall trainset accuracy = 313+453 / 313+453+21+16 = 95.4%

# Confusion Matrix on Testset
prob.test <- predict(m3.5.5, newdata = testset, type = 'response')
predict.default.test <- ifelse(prob.test > 0.5, "Yes", "No")
table4 <- table(testset$STI_Y_CAT, predict.default.test)
table4


#overall testset accurcy = 134+195 / 134+195+9+6 = 95.6% 

#no signs of overfitting since both accuracies are quite similar 

#------------------------------------------------------------
#CART
library(rpart)
library(rpart.plot)
# Import data set with categorical Y
data2 <- fread("STI_Normalized_CategoricalY.csv")

# Remove 'Date' column 
data2$Date <- NULL

set.seed(2014)

m4 <- rpart(STI_Y_CAT ~ . , method = 'class', cp = 0, data = data2)

CVerror.cap <- m4$cptable[which.min(m4$cptable[,"xerror"]), "xerror"] + 
  m4$cptable[which.min(m4$cptable[,"xerror"]), "xstd"]

# Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree m4.
i <- 1; j<- 4
while (m4$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}

# Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
cp.opt = ifelse(i > 1, sqrt(m4$cptable[i,1] * m4$cptable[i-1,1]), 1)

# Use cp.opt to prune the maximal tree to the optimal tree
m4.5 <- prune(m4, cp = cp.opt)
printcp(m4.5)
plotcp(m4.5)

#EAFE is the most important variable
m4.5$variable.importance

rpart.plot(m4.5)

mod <- rpart(STI_Y_CAT ~ ., data = data2)
pred <- predict(mod)
table(pred,data2$STI_Y_CAT)
#accuracy = 95.4%
#------------------------------------------------------------
# Random Forest
library(earth)              # MARS
library(randomForest)       # Random Forest

# Import data set with continuous Y
data3 <- fread("STI_Normalized.csv")

# Remove 'Date' column 
data3$Date <- NULL

# Train-test split ---------------------------------------------------------
set.seed(2)
train <- sample.split(Y=data3$STI_Y, SplitRatio = 0.7)
trainset <- subset(data3, train==T)
testset <- subset(data3, train==F)

# RF at default settings of B & RSF size -------------------------------------
m.RF1 <- randomForest(STI_Y ~ . , data=trainset, importance=T)

m.RF1
sqrt(0.0007915688)
#RMSE of trainset = 0.028

plot(m.RF1)
## Confirms error stablised before 500 trees.

m.RF1.yhat <- predict(m.RF1, newdata = testset)

RMSE.test.RF1 <- round(sqrt(mean((testset$STI_Y - m.RF1.yhat)^2)),3)
RMSE.test.RF1
#RMSE of testset = 0.035

var.impt.RF <- importance(m.RF1)

varImpPlot(m.RF1)
## EAFE is clearly the most impt.

#------------------------------------------------------------
# Text Analytics
library(dplyr)
library(tidytext)
library(tidyr)
data(stop_words)

setwd('C:/Users/Jing Xun/Desktop/3409 Practical Assessment/Text Analytics/Articles')

sentiments # 3 sentiment lexicons avail in package tidytext
## positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, and trust.
get_sentiments("nrc")
## Scoring from - 5 to +5 for each word
get_sentiments("afinn")
## Neg or Pos for each word
get_sentiments("bing")
nrcjoy <- get_sentiments("nrc") %>% filter(sentiment == "joy")

txtList <- dir(pattern = "*.txt") # creates the list of all the csv files in the directory
overall <- data.frame(Sentiment.Bing = rep(0, length(txtList)), Sentiment.Afinn = rep(0, length(txtList)), Sentiment.Joy = rep(0, length(txtList)), row.names = txtList)

for (each in 1:length(txtList)){
  
  # load file
  article <- readLines(txtList[each])
  article <- data_frame(text = article)
  
  # tokenize
  article.t <- article %>% unnest_tokens(word, text)
  article.t <- article.t %>% anti_join(stop_words)
  
  article.joy <- article.t %>% inner_join(nrcjoy) %>% count(word, sort = T)
  article.sen <- article.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
    spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))
  article.sen.af <- article.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
  
  overall$Sentiment.Bing[each] <- article.sen %>% summarise(sum(sentiment))
  overall$Sentiment.Afinn[each] <- article.sen.af %>% summarise(sum(value*n))
  overall$Sentiment.Joy[each] <- article.joy %>% summarise(sum(n))
  
}

overall

data <- data.frame(cbind(overall))
data <- apply(data,2,as.character)
write.table(data, "textanalytics.csv", row.names=txtList, sep=",")

#------------------------------------------------------------
# Text Analytics - Linear Regression
setwd('C:/Users/Jing Xun/Desktop/3409 Practical Assessment/Text Analytics/Articles')

# Import data set with continuous Y
data_text <- fread("textanalytics_final.csv")

# Remove 'Date' column 
data_text$Date <- NULL

m_text <- lm(STI_Y ~. , data=data_text)
summary(m_text)

#------------------------------------------------------------
