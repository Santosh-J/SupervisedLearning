load('featureMat_Liv_train_bioLBP')
load('featureMat_Gelatine_train_bioLBP')

load('featureMat_Liv_test_bioLBP')
load('featureMat_Gelatine_test_bioLBP')

train_data=[featureMat_liv_train_bioLBP' featureMat_Gelatine_train_bioLBP'];
labels=cat(2,ones(1,1000),zeros(1,200));
NBmodel=fitcnb(train_data',labels');
NBmodel.Prior
%NBmodel.Prior = [0.5 0.5]

test_data=[featureMat_liv_test_bioLBP' featureMat_Gelatine_test_bioLBP'];
test_output = predict(NBmodel,test_data');

%error = sum(labels' == test_output)/size(labels, 2);
%error = 1 - error

LossGela = loss(NBmodel, test_data', labels')

ResubGela = resubLoss(NBmodel)
