clc

images = set_train;
target = t_train;



%% these are the 2 best parameters that we learned from cross validation

disp ('---------------------PARAMETERS LEARNED FROM CROSS-VALIDATION---------------------');

C = 2
tau = 0.064

%% TRAINING

disp ('---------------------TRAINING---------------------');


n = size(images,1);         %this is number of all the pattern that we have

one = ones (n, 1);

dist = sum(images.*images, 2);   %this is nxn
Kernel = exp ( -tau * ( 1/2 * dist * one' + 1/2 * one * dist' - images * images' ) );  % this is nxn


[ bias, alpha, conv_crit, PHI] = SMO_full_dataset(target, C, tau, images);

y = Kernel * ( alpha .* target ) - bias * one;


%R is the "risk", we see how far was the 'y' from the 'target'. we compute the risk on all the
%points and not only on misclassified ones
R = ( y - target )' * ( y - target );

disp ('- Squared Error - ')
R_mean = mean(R)

disp ('- Number of 0/1 errrors -')
error01 = size( find( y.*target <= 0 ), 1 )

disp ('- % of missclassified  -')
error = 100 * error01 / n


disp ('- PHI -')
PHIfinal = one' * alpha - 1/2 * alpha' * diag(target) * Kernel * diag(target) * alpha %see course notes p.163 in bottom

%following 3 lines print the misclassified digit with the largest negative y*t
index = find( y.*target == min(y.*target),1 )
subplot(2,2,3); imagesc( reshape(images(index,:),28,28) )
                title('missclassified example:');

% -1 is 9
% +1 is 4


subplot(2,2,1); plot(PHI(1:20:end))
               xlabel('x20 iterations')
               ylabel('\phi (the SVM criterion)')
               %legend ('training error', 'validation error')
subplot(2,2,2); plot( log(conv_crit(1:20:end)) )
               xlabel('x20 iterations')
               ylabel('the SMO convergence criterion)')
               
               
               
%% -----------  TESTING -----------
disp ('---------------------TESTING ---------------------');

%We use the alpha and beta that we learned from the previous part (whole datase) 


%we reshuffle the test set each time we run SVM
rand_index = randperm( size(t_test, 1) );

 
images_test = set_test(rand_index);
target_test = t_test(rand_index);
n_test = size(images_test, 1);         %this is number of all the pattern that we have


alpha_test = ones(n_test, n) * alpha;


one = ones (n_test, 1);

dist = sum(images_test.*images_test, 2);   %this is nxn
Kernel = exp ( -tau * ( 1/2 * dist * one' + 1/2 * one * dist' - images_test * images_test' ) );  % this is nxn


y_test = Kernel * ( alpha_test .* target_test ) + bias * one;

%R is the "risk for testing", we see how far was the 'y' from the 'target'
R_test = ( y_test - target_test )' * ( y_test - target_test );

disp ('- Squared Error - ')
R_test_mean = mean(R_test)


disp ('- Number of 0/1 errors -')
error01_test = size( find( y_test.*target_test <= 0 ), 1 )

disp ('- % of missclassified  -')

error = 100 * error01_test / n_test
%%
