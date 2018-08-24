function [ phi_H, phi_L ] = phi_funct (K, w, sigma, L, H, t, f, i_up, i_low, alpha )

v_i_low = f(i_low) + t(i_low) - alpha(i_low) * t(i_low) * K(i_low, i_low) - alpha(i_up) * t(i_up) * K(i_low, i_up);
v_i_up = f(i_up) + t(i_up) - alpha(i_low) * t(i_low) * K(i_low, i_up) - alpha(i_up) * t(i_up) * K(i_up, i_up);

Li = w - sigma * L;
Hi = w - sigma * H;


phi_L = 1/2 * ( K(i_low, i_low) * Li^2 + K(i_up,i_up) * L^2 ) + sigma * K(i_low, i_up) * Li * L + t(i_low) * Li * v_i_low + t(i_up) * L * v_i_up - Li - L;
phi_H = 1/2 * ( K(i_low, i_low) * Hi^2 + K(i_up,i_up) * H^2 ) + sigma * K(i_low, i_up) * Hi * H + t(i_low) * Hi * v_i_low + t(i_up) * H * v_i_up - Hi - H;


end

