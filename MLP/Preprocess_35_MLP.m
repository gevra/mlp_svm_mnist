clc 
clear all;

% This script create 4 sets of data: 
% - train: contains 4000 images of 3 and 5 that we will use for training
% - val: contains 2000 images of 3 and 5 that we will use for validation
% - t_train and t_val the corresponding targets for val and train

% the 2 sets train and val are normalized


load('mp_3-5_data.mat');

%'normalize' function nurmalizez 'Xtrain' and 'Xtest' with respect to Xtrain values
[set_Train, set_Test] = normalize(Xtrain, Xtest);
%we first normalize and then split
[val, train, t_val, t_train] = split(set_Train, Ytrain);

clear Xtrain
clear Ytrain

clear set_Train
clear set_Test
