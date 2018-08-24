clc

images = set_train;
target = t_train;



n = size(images,1);              %this is number of all the pattern that we have

folds = 10;

number = floor(n/folds);         %this is number of patterns in each fold

y = zeros (number, folds-1);

dist = sum(images.*images, 2);   %this is nxn

    
%% these are the 2 free parameters that we change

%put here from which values you want to start the growth
C = 1;                      
tau_initial = 0.0005;

C_range = 10;            %put here how many C values you want to try
tau_range = 10;          %put here how many tau values you want to try

%put here the additive C stepsize and multiplicative tau size
C_step = 1;
tau_step = 2;

%these are not the max values you'll reach... but +1 iteration!
Cmax = C + C_range * C_step;
taumax = tau_initial * tau_step^tau_range;

%%

R_final = zeros(C_range, tau_range);
error01_final = zeros(C_range, tau_range);

CC = 1;

while C < Cmax
    
    tau = tau_initial;
    tautau = 1;
    
    while tau < taumax
        
        Kernel = exp ( -tau * ( 1/2 * dist * ones(1, n) + 1/2 * ones(n, 1) * dist' - images * images' ) );  % this is nxn
        
        i=1;
        
        while i < folds                     %so-called Cross-Validation
            
            %i
            
            %we circularly shift the image set and target set to iterate over folds
            part_image = circshift(images, -number*i);
            part_target = circshift(target, -number*i);
            
            [ bias, alpha ] = SMO(part_target(1:n-number), C, tau, part_image(1:n-number, :));
            
            
            K = circshift(Kernel, [-number*(i-1), -number*i]);
            
            y(:, i) = K(1:number, 1:n-number) * ( alpha .* part_target(1:n-number) ) - bias * ones(number, 1);
            %R is the "risk", we see how far was the 'y' from the 'target'
            R(:, i) = 1/number * (y(:, i) - part_target(n-number+1:n))' * (y(:, i) - part_target(n-number+1:n));
            
            error01(i) = size( find( y(:,i).*part_target(n-number+1:n) <= 0 ), 1 ) / number;
            
            
            i = i + 1;
            
        end
        
        
        %for the last fold we call the SMO separately, cause in last fold there could be more elements than in the others
        [ bias, alpha ] = SMO ( target(1:n-number), C, tau, images( 1:n-number, : ) ); 
        
        
        K = circshift(Kernel, [-number*(folds-1), 0]);
        
        y_last = K(1:n-number*(folds-1), 1:n-number) * ( alpha .* target(1:n-number) ) - bias * ones(n-number*(folds-1), 1);
        R_last = 1/(n-number*(folds-1)) * (y_last - target(n-number+1:n))' * (y_last - target(n-number+1:n));
        
        error01(folds) = size( find( y_last.*target(n-number+1:n) <= 0 ), 1 ) / (n-number*(folds-1));
        
        %maybe we need to substract the bias and not to add? 
        R_final (CC, tautau) = ( (folds-1)*mean(R) + R_last ) / folds;
        error01_final (CC, tautau) = 100 * mean(error01);
        
        tau = tau * tau_step;
        
        tautau
        tautau = tautau + 1;
        
    end
    
    C = C + C_step;
    CC
    CC = CC + 1;
    
end


disp ('the minimum error we get is')
min ( min( (error01_final) ) )

disp ('the best C/tau combination is the following')
C_tau_best = find ( error01_final == min ( min( (error01_final) ) ) );
C_best = mod( C_tau_best, C_range);
tau_best = (C_tau_best - C_best) / C_range + 1;
C = C - (CC - C_best) * C_step
tau = tau / ( tau_step^(tautau - tau_best))



