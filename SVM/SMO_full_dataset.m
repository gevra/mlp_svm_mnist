function [ bias, alpha, conv_crit, PHI ] = SMO(t, C, tau, X )

num_patt = size(t, 1);
alpha = zeros(num_patt, 1);         %initialization of alpha vector
f = -t;                             %initialization of f vector
one = ones (num_patt, 1);
dist = sum(X.*X, 2);
Kernel = exp ( -tau * ( 1/2 * dist * ones(1, num_patt) + 1/2 * ones(num_patt, 1) * dist' - X * X' ) );


% Iup is the set of those indices 'i' that
% 1) 0 < alpha(i) < C
% 2) t(i)=1 and alpha(i)=0
% 3) t(i)=-1 and alpha(i)=C
Iup = find( ((alpha>1e-015) & (alpha<C-1e-015)) | ((t==1) & (alpha<=1e-015)) | ((t==-1) & (alpha>=C-1e-015)) );

% Ilow is the set of those indices 'i' that
% 1) 0 < alpha(i) < C     ..... this is the same as for Iup
% 2) t(i)=-1 and alpha(i)=0
% 3) t(i)=1 and alpha(i)=C
Ilow = find( ((alpha>1e-015) & (alpha<C-1e-015)) | ((t==-1) & (alpha<=1e-015)) | ((t==1) & (alpha>=C-1e-015)) );


i_up = 1;
i_low = 1;


k = 1;
while 1  
    
       
    %this line calls a function which finds the most violated pair, and returns indices of those
    [i_low, i_up] = most_violated_pair (Ilow, Iup, f, tau);
    
    
    %this 3 line are to exit the main while when we have converged, see equation (2) in SVM documentation
    if i_up == -1
        disp ('we did "k" iterations till reaching the convergence criterion')
        k-1
        break;                          %this will stop the big while loop
    end
    
    
    alpha_old_low = alpha(i_low);       %we store old alphas, cause we are going to use them
    alpha_old_up = alpha(i_up);         %we store old alphas, cause we are going to use them
    
    
    sigma = t(i_up) * t(i_low);         %sigma can be +1 or -1
    
    
    indic1 = (sigma + 1) / 2;
    %returns 1 when sigma=+1 and 0 when sigma=-1. We use this indicator to compte L.
    indic2 = (-sigma + 1) / 2;
    %returns 1 when sigma=-1 and 0 when sigma=+1. We use this indicator to compte H.
    
    
    w = alpha(i_low) + sigma * alpha(i_up);
    
    
    L = max( 0,(sigma*w - indic1*C) );
    H = min( C,(sigma*w + indic2*C) );
    
    
    eta = Kernel(i_low, i_low) + Kernel(i_up, i_up) - 2 * Kernel(i_low, i_up);
    
    
    if eta > 1e-015
        
        %alpha_unc is the unconstrained minimizer of PHI
        alpha_unc = alpha(i_up) + ( t(i_up) * (f(i_low)-f(i_up)) ) / eta;
        
        if ((alpha_unc >= L) && (alpha_unc <= H))
            alpha(i_up) = alpha_unc;
            
        elseif alpha_unc < L
           alpha(i_up) = L;
           
        elseif alpha_unc > H
            alpha(i_up) = H;
            
        end
        
    else    %if eta<1e-15, that means the second derivative of PHI is negative. But eta is always positive!
        
        
        [ phi_H, phi_L ] = phi_funct(Kernel, w, sigma, L, H, t, f, i_up, i_low, alpha);
                
        if phi_L > phi_H
            alpha(i_up) = H;
        else
            alpha(i_up) = L;
        end
        
    end
    
    %computing new alpha(i_low) from the new alpha(i_up)
    alpha(i_low) = alpha_old_low + sigma * ( alpha_old_up - alpha(i_up) );
    %from the full alpha vector we have updated 2 elements, exactly what was needed.
    
    %updating the f...
    f = f + t(i_low) * ( alpha(i_low)-alpha_old_low ) * Kernel(:,i_low) + t(i_up) * ( alpha(i_up)-alpha_old_up ) * Kernel(:,i_up);
    
    %Updating the Iup and Ilow sets...
    Iup = find( ((alpha>1e-015) & (alpha<C-1e-015)) | ((t==1) & (alpha<=1e-015)) | ((t==-1) & (alpha>=C-1e-015)) );
    Ilow = find( ((alpha>1e-015) & (alpha<C-1e-015)) | ((t==-1) & (alpha<=1e-015)) | ((t==1) & (alpha>=C-1e-015)) );
    
    %to check whether the SMO works or no, we check if the con_crit keeps decreasing

       
    conv_crit(k) = max( f(Ilow) ) - min( f(Iup) );
    PHI (k) = one' * alpha - 1/2 * alpha' * diag(t) * Kernel * diag(t) * alpha; %see course notes p.163 in bottom

    
    k = k + 1;
end

bias = 1/2 * ( min( f(Iup) ) + max( f(Ilow) ) );

%length of PHI is (k-1)



end

