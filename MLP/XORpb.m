%%%


% d=2
% h1=4

clc

number_points = 4;
epochs = 400;
errorvect = zeros(epochs,1);
points = [0 0 ; 0 1 ; 1 0 ; 1 1];
targetorigin = [1; 0; 0; 1];
target = (2 * targetorigin) - ones(number_points, 1);
w1 = normrnd(0, 1/5, [8, 2]);
b1 = normrnd(0, 1/5, [8, 1]);
b2 = 0;
X = points';
w2 = normrnd(0, 1/5, [1, 4]);

j = 1;
lr = 1/4;
a2 = zeros(1,number_points);

while j < epochs + 1
i = 1;
while i < number_points + 1
    
a1 = activation(w1, X(:,i), b1);
z1 = transfer(a1);
a2(i) = activation(w2, z1, b2);


%----BACKPROPAGATION----

r2 = (-target(i)) / (1 + ( exp( target(i) * a2(i) ) ) );
grad2 = r2 * z1;
w2 = w2 - lr * grad2';

g_prime = transprime(a1);

r1 = residual1(r2, w2, g_prime);





grad1 = r1 * X(:,i)';
w1 = w1 - lr * grad1;




i = i + 1;
end



errorvect(j) = logistic_error(target, a2);
j = j + 1;
end





plot(errorvect);

