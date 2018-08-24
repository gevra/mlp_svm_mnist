function [ i_low, i_up ] = most_violated_pair ( Ilow, Iup, f, tau)

%this function returns indices of the most violated pair

f_up = f(Iup);
f_low = f(Ilow);



val_up = find(f_up==min(f_up), 1);
val_low = find(f_low==max(f_low), 1);


i_up = Iup(val_up);
i_low = Ilow(val_low);


if f(i_low) <= f(i_up) + 2*tau
    i_up = -1;
    i_low = -1;
end


end

