function r1 = residual1(r2, w2, g_prime)

i = 1;

h1 = size(g_prime, 1);              % is the number of units in the first hidden layer

w2_twice = [1; 1] * w2;             % this is a matrix 2x(h1/2). We need just for the next step
W2 = reshape (w2_twice, 1, h1);     % this is a vector with repeated w2 elements

r1 = r2 * W2' .* g_prime;

end