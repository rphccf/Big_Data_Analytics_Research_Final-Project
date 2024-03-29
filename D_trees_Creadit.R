### FGV/AESP - Final Project 
### Winter School Course: FGV Big Data Analytics Research
### Processor: Eric van Heck (Erasmos University)
### July/2017
### Nome do aluno: Raphael Castro da Costa Ferreira 




## Manter estas linhas
rm(list=ls());
set.seed(0);

library(rpart);
library(caret)
library(pROC)
library(ROCR)
library(randomForest)



## Carregar a base de dados do arquivo "../database/GermanCredit.tsv"

dataTree= read.table("../database/german_credit_2.csv", sep =";", header = TRUE, stringsAsFactors = FALSE);

## -- Tratamento dos dados

## Cria��o da classe nominal: Creditability 1="good" e 2="bad"

# dataTree$Creditability = ifelse(dataTree$Creditability==1,"good","bad");
# 
# 
# ## Categoriza��o da vari�vel 'CreditAmount' '0-|2500' '2500-|5000' '5000+'
# dataTree$CreditAmount = (ifelse(dataTree$CreditAmount<=2500,'0-|2500',ifelse(dataTree$CreditAmount<=5000,'2500-|5000','5000+')));
# 
# ## Categoriza��o da vari�vel 'AgeInYears' '0-|25' '25-|50' '50+'
# dataTree$AgeInYears = (ifelse(dataTree$AgeInYears<=25,'0-|25',ifelse(dataTree$AgeInYears<=50,'25-|50','50+')));
# 
# ## Categoriza��o da vari�vel 'DurationInmonth' '0-|24' '24-|48' '48+'
# dataTree$DurationInmonth = (ifelse(dataTree$DurationInmonth<=24,'0-|24',ifelse(dataTree$DurationInmonth<=48,'24-|48','48+')));


## -- Separa��o da base aleat�ria em base de treinamento e teste

## Train dataset  

ids = sort(sample(nrow(dataTree),nrow(dataTree)*0.6));

trainData = dataTree[ids,];

## Test dataset

testData = dataTree[-ids,];

## -- Constru��o do Modelo 01

## Monta a �rvore de treinamento por classifica��o

trainmod1 = rpart(Creditability ~.,data=trainData, method = "anova")


## Faz a previs�o
testmod1 = predict(trainmod1,testData)

## Verifica��o do resultado: Confusion Matrix

confusionMatrix(ifelse(testmod1>0.5,1,0), testData$Creditability);
### accuracy 0,736 e prevalence 0,3040 balanced Accuracy 

## C�lculo do custo

testmod1.prediction = prediction(testmod1,testData$Creditability)
testmod1.performance = performance(testmod1.prediction,"tpr","fpr")
plot(testmod1.performance)
abline(0, 1, lty = 2)
### accurancy accross cutoff
# plot(testmod1.prediction, "acc")

## ACU
auc1 = performance(testmod1.prediction, 'auc')
slot(auc1, 'y.values')

## -- Constru��o do Modelo 02

## Monta a �rvore de classifica��o
trainmod2 = rpart(Creditability ~ DurationInmonth+CreditAmount+ CreditHistory + PresentEmploymentSince + Job,data=trainData, method = "anova")

## Faz a previs�o
testmod2 = predict(trainmod2,testData)

## Verifica��o do resultado: Confusion Matrix
confusionMatrix(ifelse(testmod2>0.5,1,0), testData$Creditability);
### accuracy 0,7 e prevalence 0,3040

## C�lculo do custo

testmod2.prob = predict(trainmod2,testData);
testmod2.prediction = prediction(testmod2,testData$Creditability)
testmod2.performance = performance(testmod2.prediction,"tpr","fpr")
plot(testmod2.performance)
abline(0, 1, lty = 2)


###auc
auc2 = performance(testmod2.prediction, 'auc')
slot(auc2, 'y.values')


##### --- Modelo 3 (watsonanalytics)

###excel para watsonanalytics
library(xlsx)
write.xlsx(x = as.data.frame(trainData) , file="TrainTree.xlsx",sheetName = "TrainData",row.names = FALSE);


## Monta a �rvore de classifica��o
# trainmod3 = rpart(Creditability ~ CreditHistory+ ExistingCheckingAccount+OtherInstallmentPlans + PresentEmploymentSince,data=trainData, method = "class")
trainmod3 = rpart(Creditability ~ DurationInmonth+ InstallmentRate+CreditAmount +CreditHistory+ProvideMaintenanceFor,data=trainData, method = "anova")


# trainmod3 = rpart(Creditability ~ AgeInYears + Purpose+ CreditHistory +OtherInstallmentPlans + PresentEmploymentSince,data=trainData, method = "anova")


## Faz a previs�o
testmod3 = predict(trainmod3,testData)

## Verifica��o do resultado: Confusion Matrix
confusionMatrix(ifelse(testmod3>0.5,1,0), testData$Creditability);
### accuracy 0,7 e prevalence 0,3040

## C�lculo do custo

testmod3.prob = predict(trainmod3,testData);
testmod3.prediction = prediction(testmod3,testData$Creditability)
testmod3.performance = performance(testmod3.prediction,"tpr","fpr")
plot(testmod3.performance)
abline(0, 1, lty = 2)


###auc
auc3 = performance(testmod2.prediction, 'auc')
slot(auc3, 'y.values')

##### --- Modelo 4 Random Forest)



trainForest = randomForest(Creditability~., data=trainData, importance=TRUE, proximity=TRUE,ntree=500, keep.forest=TRUE);


## Faz a previs�o
testmod4 = predict(trainForest,testData)

## Verifica��o do resultado: Confusion Matrix
confusionMatrix(ifelse(testmod4>0.5,1,0), testData$Creditability);
### accuracy 0,7 e prevalence 0,3040

## C�lculo do custo

testmod4.prob = predict(trainForest,testData);
testmod4.prediction = prediction(testmod4,testData$Creditability)
testmod4.performance = performance(testmod4.prediction,"tpr","fpr")
plot(testmod4.performance)
abline(0, 1, lty = 2)


###auc
auc4 = performance(testmod4.prediction, 'auc')
slot(auc4, 'y.values')


## Plota: num.�rvores vs error 
plot(trainForest);

## Gr�fico de import�ncia das vari�vesi
varImpPlot(trainForest);


##plota todos os gr�ficos

plot(testmod1.performance, col="blue",lwd=2)
plot(testmod2.performance,add=TRUE, col="red",lwd=2)
plot(testmod3.performance,add=TRUE, col="black",lwd=2)
plot(testmod4.performance,add=TRUE, col="dark green",lwd=2)
legend(0.5,0.35,c("All","Chosen","Watson","Forest"), col= c("blue","red","black","dark green"),lty=1:1,cex=0.8)
