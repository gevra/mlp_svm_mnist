function [set_Train, set_Test]  = normalize(set_X, set_Y)

%we take the maximum and minumum of only the training set
max_val = max( max(set_X) );
min_val = min( min(set_X) );

%but we normalize with that values both the training and validation sets
set_Train = (1 / (max_val-min_val)) * ( set_X - (ones(size(set_X)) * min_val) );
set_Test = (1 / (max_val-min_val)) * ( set_Y - (ones(size(set_Y)) * min_val) );

end
