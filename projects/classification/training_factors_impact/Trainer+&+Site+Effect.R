library(tidyverse)
library(bkrdata)
#library(magrittr)
library(dplyr)

##-------------- 1. Load data ---------##
d <- hive_query("SELECT * FROM yueyang.trainer_performance_emea", dsn = "hive-ams4")

d$weekly_cpi_log <- log1p(d$weekly_cpi)

### --------------transfer columns into factors -------------###
#d$trainer <- as.factor(d$trainer)
cols <- c("staff_id","trainer", "trainer_atrrition", "channel","trainer_site",
          "sitecode", "shift_hired_for", "executive_training_shift")
d %<>%
  mutate_at(cols, funs(factor(.)))
str(d)

### -------------transfer a column to date and get the month part ----------------###
library(lubridate)
class(d$exec_training_date)
d$exec_training_date <- ymd(d$exec_training_date)
d$month <- month(d$exec_training_date)
d$month_fac <- as.factor(d$month)

##---------------2. Train-test split --------------##
## 75% of the sample size  0.75
smp_size <- floor(1 * nrow(d))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(d)), size = smp_size)

train <- d[train_ind, ]
test <- d[-train_ind, ]

#summary(train)

##-------------3.Random Forest -------------------##
library(randomForest)
# Create a Random Forest model with default parameters
model1 <- randomForest(weekly_cpi ~ trainer + sitecode + month_fac + 
                         channel + shift_hired_for, 
                       data = train, importance = TRUE)

model1$importance
model1

##### ------default plot of variable importance -----------#####
varImpPlot(model1)

model1$variable
##### ------customrized plot of variable importance with ggplot2 -----------#####
library(ggplot2) 
ggplot(model1$variable.importance, aes(x=reorder(variable,importance), y=importance,fill=importance))+ 
  geom_bar(stat="identity", position="dodge")+ coord_flip()+
  ylab("Variable Importance")+
  xlab("")+
  ggtitle("Information Value Summary")+
  guides(fill=F)+
  scale_fill_gradient(low="red", high="blue")



##-----------predict cpi with log transformation ---------------##
model1p <- randomForest(weekly_cpi_log ~ trainer + sitecode  , 
                       data = train, importance = TRUE)

model1p$importance
model1p
#scatter.smooth(x=d$month_fac, y=d$weekly_cpi, main="Month ~ CPI")  # scatterplot

predict(test$weekly_cpi, model1)

## ------------------4. Liner Regression -------------------##

model_null <- lm(weekly_cpi ~ trainer, data=train)
summary(model_null)


model2 <- lm(weekly_cpi ~ trainer + sitecode + channel + month_fac + class_size, data=train)

summary(model2)

##-------------- get the impact of trainer --------##
ln.mod1 <- glm(weekly_cpi ~ trainer + sitecode + month_fac + channel,
               data=train) # with trainer
ln.mod2 <- glm(weekly_cpi ~ sitecode + month_fac + channel, 
               data=train) # without trainer
anova(ln.mod1, ln.mod2, test="LRT")

