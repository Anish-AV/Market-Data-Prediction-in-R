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
#                           One Hot encoding                                      #
###################################################################################
dmy <- dummyVars(" ~ .", data = Market_data[-15])
new_data=data.frame(predict(dmy, newdata = Market_data[-15]))
new_data=cbind(new_data,Market_data[15])

###################################################################################
#                     Cross Validation and Hyper Parameter Tuning                 #
###################################################################################

CV_Hyper_parameter_tuning=function(task,learner,parameter_set){
  
  #Cross Validation by making Resampling Decision
  rdesc = makeResampleDesc("CV", iters=3)
  #rsi=makeResampleInstance(rdesc,task)
  
  #hyperparameter tuning
  ctrl = makeTuneControlGrid()
  res = tuneParams(learner, task = task, resampling = rdesc, par.set = parameter_set, control =ctrl, measures = acc)
  
  return(res)
}

###################################################################################
#                                     KNN                                         #
###################################################################################
#Creating the task
new_task=makeClassifTask(data=new_data,target="purchase")

#creating the learner
learner_knn=makeLearner("classif.knn")
#print(getParamSet(learner_knn))
parameter_set_knn=makeParamSet(makeDiscreteParam("k", values = seq(1,5,1))
)
resample_knn=CV_Hyper_parameter_tuning(new_task,learner_knn,parameter_set_knn)
#View(resample_knn$x)
#View(resample_knn$y)

learner_knn=makeLearner("classif.knn",k=resample_knn$x$k)
model_knn= mlr::train(learner_knn,new_task,subset = train)
predict_knn=predict(model_knn,task=new_task,subset = test)
accuracy_knn=performance(predict_knn,measures = acc)
cm_knn=confusionMatrix(predict_knn[["data"]][["truth"]],predict_knn[["data"]][["response"]])
plot(cm_knn$table,main="confusion matrix")
#print(acc_knnr)
