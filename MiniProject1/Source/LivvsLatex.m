load('featureMat_Liv_train_bioLBP')
load('featureMat_Latex_train_bioLBP')
load('featureMat_Liv_test_bioLBP')
load('featureMat_Latex_test_bioLBP')

train_data=[featureMat_liv_train_bioLBP' featureMat_Latex_train_bioLBP'];
labels=cat(2,ones(1,1000),zeros(1,200));
NBmodel=fitcnb(train_data',labels','Prior',[0.6 0.4]);
NBmodel.Prior
%NBmodel.Prior = [0.5 0.5]

test_data=[featureMat_liv_test_bioLBP' featureMat_Latex_test_bioLBP'];
test_output = predict(NBmodel,test_data');

%error = sum(labels' == test_output)/size(labels, 2);
%error = 1 - error

LossLat = loss(NBmodel, test_data', labels')

ResubLat = resubLoss(NBmodel)

