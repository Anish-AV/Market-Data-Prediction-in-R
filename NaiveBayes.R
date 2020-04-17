###################################################################################
#                           Importing the libraries                               #
###################################################################################
library(mlr)
library(caret)


###################################################################################
#                           Importing the data                                    #
###################################################################################

Market_data= read.csv("data\\Market.csv")
Market_data_pred=read.csv("data\\Market_pred.csv")



###################################################################################
#                           Splitting the training data                           #
###################################################################################
n=nrow(Market_data)
train=sample(n,0.7*n)
test=setdiff(1:n,train)



###################################################################################
#                                     Naive Bayes                                 #
###################################################################################
#Creating the task
task=makeClassifTask(data=Market_data,target="purchase")

learner_nb=makeLearner("classif.naiveBayes")
#print(getParamSet(lrn_nb))
model_nb=mlr::train(learner=learner_nb , task = task, subset = train)
predict_nb=predict(model_nb,task = task,subset =test[-ncol(Market_data)] )
accuracy_nb=performance(predict_nb,measures = acc)
cm_nb=confusionMatrix(predict_nb[["data"]][["truth"]],predict_nb[["data"]][["response"]])
plot(cm_nb$table,main="confusion matrix")
