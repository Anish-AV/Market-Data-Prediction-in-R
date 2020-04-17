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

#Decision Tree
learner_dt=makeLearner("classif.rpart")
#print(getParamSet(lrn_dt))
paramset_dt=makeParamSet(makeIntegerParam("minsplit", lower=1, upper=100),
                   makeIntegerParam("maxdepth", lower=2, upper=50),
                   makeDiscreteParam("cp", values = seq(0.001, 0.006, 0.002)))



res_dt=CV_Hyper_parameter_tuning(task,learner_dt,paramset_dt)
#View(res_dt$x)
#View(res_dt$y)


#Learner
learner_dt=makeLearner("classif.rpart", minsplit=res_dt$x$minsplit, maxdepth=res_dt$x$maxdepth,cp=res_dt$x$cp)

#model
model_dt=mlr::train(learner_dt,task,subset = train)

#prediction
predict_dt=predict(model_dt,task,subset=test[-ncol(Market_data)])

#Performance evaluation
accuracy_dt=performance(predict_dt,measures = acc)
cm_dt=confusionMatrix(predict_dt[["data"]][["truth"]],predict_dt[["data"]][["response"]])
plot(cm_dt$table,main="confusion matrix")
#print(accuracy_dt)
