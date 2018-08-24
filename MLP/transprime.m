function gprime = transprime(a)

%grime is matrix(h1x1)...

gprime = zeros(size(a));        %by this operation we force gprime to be a colum because by default it was a row

h = size(a, 1);

one = ones (h, 1);

a_plus = circshift(a, -1);      %this is vector of a(q+1) circularly shifted from a(q)
a_minus = circshift(a, 1);      %this is vector of a(q-1) circularly shifted from a(q)

gprime_odd = one ./ ( one + exp(-a_plus) );                    %this is the function to applied for odd indices
gprime_even = a_minus ./ ( (one+exp(-a)) .* (one+exp(a)) );    %this is the function to applied for even indices

one_zero = mod (1:h, 2)';
zero_one = circshift(one_zero, 1);


gprime = one_zero .* gprime_odd + zero_one .* gprime_even;


end

