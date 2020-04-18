###################################################################################
#                           Importing the libraries                               #
###################################################################################
library(mlr)
library(caret)


###################################################################################
#                           Importing the data                                    #
###################################################################################

Market_data= read.csv("data\\Market.csv")
Market_data_pred=read.csv("Predictions\\Market_pred.csv")



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
#                                     Random Forest                               #
###################################################################################
#Predictions using random forest on the market pred
market_task=makeClassifTask(data = Market_data,target="purchase")

market_learner=makeLearner("classif.randomForest")
#print(getParamSet(learner_rf))
market_paramset=makeParamSet(makeDiscreteParam("ntree",values=seq(50,500,50))
)
market_res=CV_Hyper_parameter_tuning(market_task,market_learner,market_paramset)
#View(market_res$x)
#View(market_res$y)

market_learner=makeLearner("classif.randomForest",ntree=market_res$x$ntree)
market_model=mlr::train(market_learner,market_task)
market_predict=predict(market_model,newdata=Market_data_pred)
Market_data_pred=Market_data_pred[-15]
purchase= market_predict[["data"]][["response"]]
Market_data_pred=cbind(Market_data_pred,purchase)

write.csv(Market_data_pred,"Predictions\\Market_prediction.csv")
