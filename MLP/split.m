function [set_Val, set_Train, target_Val, target_Train] = split(set_X, target)

a = size(set_X, 1);        %this returns 6000

rand_index = randperm(a);  %this returns randomly ordered numbers from 1 to 6000

% so rand_index is an array 6000x1 with randomly ordered numbers from 1 to 6000
% rand_index (1:(a/3)) are the first 2000 numbers of the array... some numbers between 1 and 2000

set_Val = set_X ( rand_index( 1:(a/3) ), :);      % this takes 2000 random rows from Xtrain 

target_Val = target ( rand_index( 1:(a/3) ), :);

set_Train = set_X ( rand_index( (a/3+1):a ), :);

target_Train = target ( rand_index( (a/3+1):a ), :);