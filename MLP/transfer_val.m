function z = tranfer(a)

i = 1;

z = zeros( size(a, 1) / 2 , size(a, 2) );

while i < numel(a) / 2 + 1
    
    z(i) = a( 2 * i - 1 ) / ( 1 + exp( -a( 2 * i ) ) );
    
    i = i + 1;

end

end
