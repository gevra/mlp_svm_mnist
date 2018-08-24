clc 
clear all;

% This script creates 4 sets of data: 
% - 'set_train': contains 6000 images of 4 and 9 that we will use for training
% - 'set_test': contains 1991 images of 4 and 9 that we will use for testing
% - 't_train' and 't_test' are the corresponding targets for 'set_train' and 'set_test'

% the 2 sets 'set_train' and 'set_test' will be are normalized & randomized


load('mp_4-9_data.mat');

[set_a, set_b] = normalize(Xtrain, Xtest);    %just normalizing the image sets w.r.t. Xtrain

 
rand_index_a = randperm( size(set_a, 1) );  %this returns randomly ordered numbers from 1 to 6000
rand_index_b = randperm( size(set_b, 1) );  %this returns randomly ordered numbers from 1 to 1991


set_train = set_a ( rand_index_a, :);      % this takes 6000 random rows from set_a 

t_train = Ytrain ( rand_index_a, :);       % 6000 corresponding targets  

set_test = set_b ( rand_index_b, :);       % this takes 1991 random rows from set_b

t_test = Ytest ( rand_index_b, :);         % 1991 corresponding targets  


clear Xtrain
clear Ytrain

clear set_a
clear set_b

clear rand_index_a
clear rand_index_b