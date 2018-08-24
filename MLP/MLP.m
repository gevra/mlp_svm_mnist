%%%

% d=784 (dimentionality of input vector)

clc

%Preprocess_35_MLP.m is normalizing datasets and then splitting the Xtrain to 'train' and 'val' sets.
%we run it just ones before running the MLP...

num_patt_t = size(train, 1);       % 4000
num_patt_v = size(val, 1);         % 2000


%%
%these are the 8 parameters/initializations to play with!
epochs = 40;
mu = 0.8;                                   %the momentum term....use 0.8 for 3vs5 and 0.9 for 4vs9
h1 = 64;                                    %number of units in the first hidden layer....use 64
lr = 0.06;                                  %the learning rate....use 0.06 for 3vs5 and 0.08 for 4vs9


w1 = normrnd(0, 0.1, [h1, 784]);            %initialization of w1 with normal distribution
b1 = normrnd(0, 0.1, [h1, 1]);              %initialization of b1 with normal distribution
w2 = normrnd(0, 0.5, [1, h1/2]);            %initialization of w2 with normal distribution
b2 = 0;
%%

X = train';                                 %this is the training set
a2 = zeros(1, num_patt_t);                  %............just for initialization
a2_v = zeros(1, num_patt_v);                %............just for initialization
j = 1;                                      %............just for initialization
train_error_p = zeros(num_patt_t, 1);       %............just for initialization
train_error = zeros(epochs, 1);             %............just for initialization
val_error = zeros(epochs, 1);               %............just for initialization
dw2 = zeros( size(w2) );                    %............just for initialization
dw1 = zeros( size(w1) );                    %............just for initialization
train_error_01 = zeros(epochs, 1);          %............initialization
val_error_01 = zeros(epochs, 1);            %............initialization
val_error_min = 1;                          %............just for initialization



disp ('now the MLP is learning the best w1 and w2')

%%
while j < epochs + 1                        %iteration on the epochs for training the MLP
    
    i = 1;

%%  
    while i < num_patt_t + 1                %iteration on the points

        %----FORWARD PASS----
        
        a1 = activation(w1, X(:,i), b1);
        z1 = transfer(a1);
        a2(i) = activation(w2,z1,b2);
        
        %----END OF FORWARD PASS----




        %----BACKPROPAGATION----

        r2 = (-t_train(i)) / (1 + exp( t_train(i)*a2(i) ) );
        grad2 = r2 * z1;
        dw2 = mu*dw2 - (1-mu)*(lr/j)*grad2';
        w2 = w2 + dw2;

        g_prime = transprime(a1);                          %derivative of the transfer function
        r1 = residual1(r2, w2, g_prime);
        grad1 = r1 * X(:,i)';
        dw1 = mu*dw1 - (1-mu)*(lr/j)*grad1;
        w1 = w1 + dw1;

        train_error_p(i) = logistic_error(t_train, a2);    %error on a point of the training set

        %----END OF BACKPROPAGATION----
        
        i = i + 1;

    end
%% 

    train_error_01(j) = 100 * size ( find (t_train .* a2' < 0), 1 ) / num_patt_t;     %calculating the 0/1 error on the training set
    
    train_error(j) = logistic_error(t_train, a2);        %error on the training set
    

%%    %----COMPUTING THE VALIDATION ERROR----

    V = val';
    b1_v = ones(num_patt_v, 1) * b1';
    b2_v = ones(num_patt_v, 1) * b2';
    
    a1_v = activation(w1, V, b1_v');
    z1_v = transfer_val(a1_v);
    a2_v = activation(w2, z1_v, b2_v');
    
    val_error(j) = logistic_error(t_val, a2_v);          %logistic error on the validation set
    
    if val_error(j) > val_error_min
        disp ('overfitting occurs in')
        epoch = j
        early_stop = j-2;
        disp ('the percentage of 0/1 error on the validation set is')
        val_error_01(j-1)
        disp ('now we are going to test the MLP on the validation set')
        j = epochs + 1;
        break
    end
    
    val_error_min = val_error(j);
    w2_best = w2;
    w1_best = w1;
    
    val_error_01(j) = 100 * size (find( t_val .* a2_v' < 0), 1 ) / num_patt_v;     %calculating the 0/1 error on the validation set   

    %----END OF COMPUTING THE VALIDATION ERROR----
%%



    current_epoch_is = j               %this line is just printing the number of epoch that we are in
    j = j+1;
end


%% TESTING MLP ON THE TESTING SET

num_patt_test = size(Ytest, 1);

rand_index_test = randperm(num_patt_test);

Test_set = Xtest(rand_index_test, :);
Test_target = Ytest(rand_index_test);

b1_test = ones(num_patt_test, 1) * b1';
b2_test = ones(num_patt_test, 1) * b2';

a1_test = activation(w1_best, Test_set', b1_test');
z1_test = transfer_val(a1_test);
a2_test = activation(w2_best, z1_test, b2_test');

disp ('logistic error on the testing set is')
test_error = logistic_error(Test_target, a2_test);          %logistic error on the testing set

disp ('percentage of misclassified patterns on the testing set is')
test_error_01 = 100 * size (find( Test_target .* a2_test' < 0), 1 ) / num_patt_test     %calculating the 0/1 error on the testing set   

%here are 2 examples of misclassified patterns
misclass = find (Test_target .* a2_test' == min(Test_target .* a2_test'));
misclass_almost = find ( 1./(Test_target .* a2_test') == min( 1./(Test_target .* a2_test') ) );

subplot(2,2,3); imagesc( reshape(Test_set(misclass, :), 28, 28) );
                title ('a misclassified pattern')
subplot(2,2,4); imagesc( reshape(Test_set(misclass_almost, :), 28, 28) );
                title ('another misclassified pattern')

%----END OF TESTING MLP ON THE TESTING SET
%% 


subplot(2,2,1); plot(1:early_stop, train_error(1:early_stop), 1:early_stop, val_error(1:early_stop))
               xlabel('epochs')
               ylabel('logistic error')
               legend ('training error', 'validation error')
subplot(2,2,2); plot(1:early_stop, train_error_01(1:early_stop), 1:early_stop, val_error_01(1:early_stop))
               xlabel('epochs')
               ylabel('0/1 error (%)')
               legend ('0/1 error on training set', '0/1 error on validation set')
              
               
               