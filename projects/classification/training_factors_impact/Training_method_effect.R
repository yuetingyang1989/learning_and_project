library(tidyverse)
library(bkrdata)
#library(magrittr)
library(dplyr)

##-------------- 1. Load data ---------##
d <- hive_query("SELECT * FROM cs_learning.csp_training_method_agent_cpi", dsn = "hive-ams4")
d$trainer_tenure_log <- log1p(d$trainer_tenure_training_in_days)
d$trainer_tenure_prior_log <- log1p(d$trainer_tenure_prior_trainer_in_days)

### --------------transfer columns into factors -------------###
#d$trainer_id <- as.factor(d$trainer_id)
cols <- c("staff_id","job_profile","sitecode","channel", "mp_team_name", "hired_language",
          "trainer_id","trainer_name", "trainer_site",
           "mp_team_name", "training_method")
d %<>%
  mutate_at(cols, funs(factor(.)))
str(d)


##---------------2. Train-test split --------------##
## 75% of the sample size  0.75
smp_size <- floor(1 * nrow(d))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(d)), size = smp_size)

train <- d[train_ind, ]
test <- d[-train_ind, ]


## ------------------3. Liner Regression -------------------##

model_null <- lm(cpi ~ training_method, data=train)
summary(model_null)

model_null_corrected <- lm(cpi_corrected ~ training_method, data=train)
summary(model_null_corrected)

### --------- 3.1 add job profile ---------------###

model_job_profile <- lm(cpi ~ training_method + job_profile, data=train)
summary(model_job_profile)

model_job_profile_corrected <- lm(cpi_corrected ~ training_method + job_profile, data=train)
summary(model_job_profile_corrected)


### --------- 3.2 add channel ---------------###
model_channel <- lm(cpi ~ training_method + channel, data=train)
summary(model_channel)

model_channel_corrected <- lm(cpi_corrected ~ training_method + channel, data=train)
summary(model_channel_corrected)


### --------- 3.3 add job profile + channel ---------------###
model_job_profile_channel <- lm(cpi ~ training_method + job_profile + channel, data=train)
summary(model_job_profile_channel)

model_job_profile_channel_corrected <- lm(cpi_corrected ~ training_method + job_profile + channel, data=train)
summary(model_job_profile_channel_corrected)



##-------------4.Random Forest -------------------##
library(randomForest)

###-------------4.1 Predict CPI -------------------##
# Create a Random Forest model with default parameters
model_cpi_basic <- randomForest(cpi ~ training_method + job_profile + channel , 
                       data = train, importance = TRUE)

model_cpi_basic$importance
model_cpi_basic

# Adding more predicor variable 
model_cpi_full <- randomForest(cpi ~ training_method +  job_profile + channel 
                       + trainer_name + sitecode + hired_language + class_size + trainer_site 
#                       + exp_cpi_quarter
#                       + mp_team_name
#                       + trainer_tenure_training_in_days
#                        +trainer_tenure_prior_training_in_days
                        , 
                       data = train, importance = TRUE)

model_cpi_full$importance
model_cpi_full



##### ------default plot of variable importance -----------#####
varImpPlot(model_cpi_full)

model_cpi_full$variable
##### ------customrized plot of variable importance with ggplot2 -----------#####
library(ggplot2) 
ggplot(model_cpi_full$variable.importance, aes(x=reorder(variable,importance), y=importance,fill=importance))+ 
  geom_bar(stat="identity", position="dodge")+ coord_flip()+
  ylab("Variable Importance")+
  xlab("")+
  ggtitle("Information Value Summary")+
  guides(fill=F)+
  scale_fill_gradient(low="red", high="blue")


###-------------4.2 Predict CPI Ajusted-------------------##

# Create a Random Forest model with default parameters
model_cpi_corrected_basic <- randomForest(cpi_corrected ~ training_method + job_profile + channel , 
                                data = train, importance = TRUE)

model_cpi_corrected_basic$importance
model_cpi_corrected_basic

# Adding more predicor variable 
model_cpi_corrected_full <- randomForest(cpi_corrected ~ training_method +  job_profile + channel 
                               + trainer_name + sitecode + hired_language + class_size + trainer_site 
                               #                       + exp_cpi_quarter
                               #                       + mp_team_name
                               #                       + trainer_tenure_training_in_days
                               #                        +trainer_tenure_prior_training_in_days
                               , 
                               data = train, importance = TRUE)

model_cpi_corrected_full$importance
model_cpi_corrected_full