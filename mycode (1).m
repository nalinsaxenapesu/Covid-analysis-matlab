
%%

trainData = readtable("Training_Data.xls");
testData = readtable("Testing_Data.xls");

trainData = convertvars(trainData,10:16,"categorical");
trainData = trainData(:,3:end);
testData = convertvars(testData,10:16,"categorical");
testData = testData(:,3:end);
%%

mdl = fitcecoc(trainData,"Class",'Learners',templateSVM('Standardize',true','KernelFunction','gaussian'))
%%

trainerror = resubLoss(mdl)
trainAccuracy = 1-trainerror

%%
ypred = predict(mdl,trainData);

confusionchart(trainData.Class,ypred)
