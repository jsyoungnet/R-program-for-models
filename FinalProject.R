library(ggplot2)
library(lattice)
library(caret)
library("randomForest")
library(sandwich)
library(party)
library(grid)
library(mvtnorm)
library(zoo)
library(modeltools)
library(stats4)
library(strucchange)
library(kernlab)

#load the data from the files placing NA for all non-values

testing_data_raw <- read.csv('pml-testing.csv', header = TRUE, na.strings = c("NA","NaN","","#DIV/0!"))
training_data_raw <- read.csv('pml-training.csv', header = TRUE, na.strings = c("NA","NaN","","#DIV/0!"))

#remove first column (ID)
testing_data_raw <- testing_data_raw[,-1]
training_data_raw <- training_data_raw[,-1]

validation_partition <- createDataPartition(y=training_data_raw$classe, list = FALSE, p=4/5)

#split the dataset into a training and a validation set, "testing" is already taken

validation_data<- training_data_raw[-validation_partition,]
training_data <- training_data_raw[validation_partition,]

#find the number of NA values in a colum

na_values <- sapply(training_data, function(x) sum(is.na(x)))

keep_columns <- c()
discard_columns <- c()

#if the number of NA values in a column is greater than 1/3 of the total number
#delete that column

for (column_names in colnames(training_data_raw)){
        
        if ((na_values[column_names]) < (max(na_values)/3)) {
                keep_columns = append(keep_columns, column_names)
        }else{
                discard_columns = append(discard_columns, column_names)
                #print_arg <- c("COLUMN ", column_names, "CONTAINS ", na_values[column_names], "NA VALUES")
                #print (print_arg)
        }
}

print ("THESE COLUMNS CONTAIN MOSTLY NA ")
print (discard_columns)

print ("THESE COLUMNS CONTAIN VALID DATA")
print (keep_columns)

#print ("TESTING DATA")
#print (colnames(testing_data_raw))
#print ("INTERSECT")
#print (intersect(discard_columns, colnames(testing_data_raw)))

nzv_training_data <- training_data[,keep_columns]


#check for any remaining columns that have near zero variance

nzv_delete_columns <- nearZeroVar(nzv_training_data)
for (a_column in nzv_delete_columns){
  discard_columns <- append(discard_columns, nzv_training_data[colnames(nzv_training_data[a_column])])
}

correlation_training_data <- nzv_training_data[,-nzv_delete_columns]

print ("DROPPING COLUMNS FOR NEAR ZERO VARIANCE")
print (colnames(nzv_training_data[,nzv_delete_columns]))

#for the correlation to work, all columns must be integer
non_integer_columns <- c('user_name','cvtd_timestamp', 'new_window', 'classe')


training_data_integers_only <- correlation_training_data[,!names(correlation_training_data) %in% non_integer_columns]
#nzv_training_data <- nzv_training_data[,!names(nzv_training_data) %in% non_integer_columns]


#check for any highly-correlated combinations of columns and remove them
correlated_training_data <- cor(training_data_integers_only, use = 'pairwise.complete.obs' )
correlated_columns_to_delete <- findCorrelation(correlated_training_data, 0.90)

for (cor_column in correlated_columns_to_delete){
  discard_columns <- append(discard_columns, correlation_training_data[colnames(correlation_training_data[cor_column])])
}
#combine the nzv columns with the highly correlated columns
print ("DROPPING COLUMNS FOR HIGH CORRELATION")
print (colnames(correlated_training_data[,correlated_columns_to_delete]))

clean_training_data <- training_data[,!names(training_data) %in% discard_columns]
clean_testing_data <- testing_data_raw[,!names(testing_data_raw) %in% discard_columns]
clean_validation_data <- validation_data[,!names(validation_data) %in% discard_columns]


#wipe out any remaining NA values
clean_training_data[is.na(clean_training_data)] <- 0
clean_testing_data[is.na(clean_testing_data)] <- 0
clean_validation_data[is.na(clean_validation_data)] <- 0



clean_column_names <- colnames(clean_training_data)
clean_val_names <- colnames(clean_validation_data)

print ('the diff between training and validation columns:')
print (setdiff(clean_val_names, clean_column_names))

#some methods are going to use "y ~ ." and some aren't so get a df without the classe in it

#classe_num <- which( colnames(training_data)=="classe" )
#drop_classe <- append(discard_columns, 'classe')
#print(drop_classe)
#training_classe_excluded <- training_data[,!names(clean_training_data) %in% discard_columns]

print ("KEEPING THESE COLUMNS")
print (clean_column_names)

#preProcess the remaining columnstesting_data_raw <- read.csv('pml-testing.csv', header = TRUE, na.strings = c("NA","NaN","","#DIV/0!"))

#xTrans <- preProcess(clean_training_data)
#testing_data_processed <- predict(xTrans,clean_testing_data)
#training_data_processed <- predict(xTrans,clean_training_data)
#validation_data_processed <- predict(xTrans,clean_validation_data)



training_data_classe <- training_data[,'classe']
bootControl <- trainControl(number=3000)
preProcess_methods <- c("scale", 'center')

print ("STARTING PLAIN RANDOMFOREST")

set.seed(14387)

rf_Model <- randomForest(classe ~., data=clean_training_data)

print ("STARTING TRAIN RF WITH REPEATED CV")
#
t_control <- trainControl(method= "repeatedcv", number= 5, repeats= 1, verboseIter = TRUE)
mtry_def <- floor(sqrt(ncol(clean_training_data)))
t_grid <- expand.grid(mtry= c(mtry_def/2, mtry_def, 2 * mtry_def))


set.seed(14387)

## works without parallel
rf_RCV_Model <- train(classe ~ ., data= clean_training_data,
                      method= "cforest", 
                      trControl= t_control,
                      tuneGrid= t_grid) 

# remove verbose, importance, proximity

#print ("STARTING TRAIN RANDOM FOREST")

#rf_train_Model <- train(classe  ~., 
#               data = clean_training_data, 
#               #preProcess=preProcess_methods,  
#               method = 'rf', 
#               mtry = 10)

print ("RANDOM FOREST COMPLETE")
#svmFit <- train(classe ~., data = clean_training_data, preProcess=preProcess_methods, method='svmRadial', tuneLength = 3, trControl = bootControl, na.action = na.omit)

#print ("SUPPORT VECTOR MACHINE COMPLETE")

common <- intersect(names(clean_training_data), names(clean_testing_data)) 
for (p in common) { 
  if (class(clean_training_data[[p]]) == "factor") { 
    levels(clean_testing_data[[p]]) <- levels(clean_training_data[[p]]) 
  } 
}
