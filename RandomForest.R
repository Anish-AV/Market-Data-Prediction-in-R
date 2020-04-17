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
#                       Cross Validation and Hyper Parameter Tuning              #
###################################################################################
#funciton to implementcross-validation and hyper parameter tuning
CV_Hyper_parameter_tuning=function(task,learner,ps){
  set.seed(6052)
  #Cross Validation
  rdesc = makeResampleDesc("CV", iters=3)
  #rsi=makeResampleInstance(rdesc,task)
  
  #hyperparameter tuning
  ctrl = makeTuneControlGrid()
  res = tuneParams(learner, task = task, resampling = rdesc, par.set = ps, control =ctrl, measures = acc)
  
  return(res)
}


###################################################################################
#                                     Decision Tree                               #
###################################################################################
#Creating the task
task=makeClassifTask(data=Market_data,target="purchase")

#Random Forest robust evaluation
learner_rf=makeLearner("classif.randomForest")

#print(getParamSet(learner_rf))
paramset_rf=makeParamSet(makeDiscreteParam("ntree",values=seq(50,500,50))
)
res_rf=CV_Hyper_parameter_tuning(task,learner_rf,paramset_rf)
#View(res_rf$x)
#View(res_rf$y)

#learner
learner_rf=makeLearner("classif.randomForest",ntree=res_rf$x$ntree)

#model
model_rf=mlr::train(learner_rf,task,subset = train)

#predict
predict_rf=predict(model_rf,task,subset=test[-ncol(Market_data)])

#performance evaluation
accuracy_rf=performance(predict_rf,measures = acc)
cm_rf=confusionMatrix(predict_rf[["data"]][["truth"]],predict_rf[["data"]][["response"]])
plot(cm_rf$table,main="confusion matrix")
#print(acc_rf)