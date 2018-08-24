function error = logistic_error(target, a2)

%this function returns the logistic error

v = -target .* a2';
pos_pos = find(v >= 0);      %this extracts positions of positive v's
pos_neg = find(v < 0);       %this extracts positions of negative v's



error_pos = sum( v(pos_pos) + log1p(exp(-v(pos_pos))) )  ;
error_neg = sum( v(pos_neg) + log(1+exp(-v(pos_neg))) )  ;

error = ( 1 / size(target, 1) ) * ( sum(error_pos) + sum(error_neg) );

end