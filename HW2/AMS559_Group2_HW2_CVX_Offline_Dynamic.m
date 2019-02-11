% ------------------------------------------------------------------------
% AMS 559
% Author: John Buckheit
% Group 2
% 15/11/18
% Homework 2, Due: 2/12/18
% Offline Dynamic CVX
% MATLAB version R2018b
% ------------------------------------------------------------------------

% Choosing a home number -------------------------------------------------
home_num = input('Enter the number of the desired home (1-10): ');
if home_num == 1
    data = csvread('Home1_yr1.csv');
elseif home_num == 2
    data = csvread('Home2_yr1.csv');
elseif home_num == 3
    data = csvread('Home3_yr1.csv');
elseif home_num == 4
    data = csvread('Home4_yr1.csv');
elseif home_num == 5
    data = csvread('Home5_yr1.csv');
elseif home_num == 6
    data = csvread('Home6_yr1.csv');
elseif home_num == 7
    data = csvread('Home7_yr1.csv');
elseif home_num == 8
    data = csvread('Home8_yr1.csv');
elseif home_num == 9
    data = csvread('Home9_yr1.csv');
elseif home_num == 10
    data = csvread('Home10_yr1.csv');
else
    disp('Not a valid home number');
    return
end
T = 4*24*7; % The data is separated into 15min time steps. T is our window
% that we are concerned with optimizing. In this case, we use 1 week of
% data.

% Interval of interest
y = data(32161:32161 + T -1);

% p(t) is the price of electricity at time t, we use a constant initially
% a is the penalty of insufficient provisioning
% b is the switching cost coefficient

t = length(y);

cvx_begin
obj = 0;
p = 0.4/4; % 0.4 kWh divided by 4 to give kW15min
a = 4/4;
b = 4/4;
variables x(t)
x(1) == 0;
for k = 2:t
    obj = obj + p*x(k) + a*max(0,y(k) - x(k))+ b*abs(x(k) - x(k-1));
end
minimize(obj)

cvx_end

x_CVX_dynamic = x;

figure,
plot(y)
hold on
plot(x)
title(sprintf('CVX Offline Dynamic Optimization (Home %d)',home_num))
xlabel('Timestep (15min)')
ylabel('Energy (kW15min)')
legend('Energy Demand (Actual)','Energy Supplied')
txt = sprintf(...
    'a = %.02f $/kW15min \nb = %.02f $/kW15min \nOptimal Cost: %.03f'...
    ,a,b,obj);
text(T*0.05,max(y)*.9,txt)

