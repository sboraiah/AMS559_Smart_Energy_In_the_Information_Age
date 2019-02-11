% ------------------------------------------------------------------------
% AMS 559
% Author: John Buckheit
% Group 2
% 1/12/18
% Homework 2, Due: 2/12/18
% Committed Horizon Control
% MATLAB version R2018b
% ------------------------------------------------------------------------

% home 1,4,8 are recommended
% Choosing a home number -------------------------------------------------
home_num = input('Enter the number of the desired home (1-10): ');
if home_num == 1
    data = csvread('Home1_yr1.csv');
    pred = load('hw1_prediction_home1.txt');
elseif home_num == 2
    data = csvread('Home2_yr1.csv');
    pred = load('hw1_prediction_home2.txt');
elseif home_num == 3
    data = csvread('Home3_yr1.csv');
    pred = load('hw1_prediction_home3.txt');
elseif home_num == 4
    data = csvread('Home4_yr1.csv');
    pred = load('hw1_prediction_home4.txt');
elseif home_num == 5
    data = csvread('Home5_yr1.csv');
    pred = load('hw1_prediction_home5.txt');
elseif home_num == 6
    data = csvread('Home6_yr1.csv');
    pred = load('hw1_prediction_home6.txt');
elseif home_num == 7
    data = csvread('Home7_yr1.csv');
    pred = load('hw1_prediction_home7.txt');
elseif home_num == 8
    data = csvread('Home8_yr1.csv');
    pred = load('hw1_prediction_home8.txt');
elseif home_num == 9
    data = csvread('Home9_yr1.csv');
    pred = load('hw1_prediction_home9.txt');
elseif home_num == 10
    data = csvread('Home10_yr1.csv');
    pred = load('hw1_prediction_home10.txt');
else
    disp('Not a valid home number');
    return
end

T = 4*24*7; % The data is separated into 15min time steps. T is our window
% that we are concerned with optimizing. In this case, we use 1 week of
% data.

%{
% iterate for optimal committment
upp = 10;
for v = 1:upp
%}

% Interval of interest
y = data(32161:32161 + T -1);

y_pred = pred;

% Costs and penalties
p = 0.4/4; % 0.4 kWh divided by 4 to give kW15min
a = 4/4;
b = 4/4;
% variables
stepsize = 22/T; % fixed depending on each home, Use 22,11,25 for homes 1,4,8 respectively
w = 3; % fixed depending on each home, Use 3,10,9 for homes 1,4,8 respectively
v = 3; % Use 3,10,2 for homes 1,4,8 respectively
% Get inital values for committment
clear x
for i = 1:v
    x(i) = data(32161 - i);
end
x = fliplr(x);
x = [x,0];
x_w = x;
x_v = x;

% Determine supply based on gradient, window size, and committment level
for t = v+1:T-1-w
    for k = t:t+w
        x_w(k+1) = x_w(k) - stepsize*grad(x_w(k),x_w(k-1),y_pred(k),p,a,b);
    end
    for k = (t-v+1:t)
        x_v(k+1) = x_v(k) - stepsize*grad(x_v(k),x_v(k-1),y_pred(k),p,a,b);
    x(t+1) = (sum(x_w(t:t+w)) + sum(x_v(t-v:t)))/(w + v);
    end
end
% Fill in missing values outside final window
for t = T - w:T-1
    x(t+1) = x(t) - stepsize*grad(x(t),x(t-1),y_pred(t),p,a,b);
end


x = x';
cost = 0;
% find objective cost for the determined supply (x)
for t = 2:T
    cost = cost + p*x(t) + a*max(0,y(t) - x(t)) + b*abs(x(t) - x(t-1));
end

x_CHC = x;

%{
% find optimal committment
CHCcost(v) = cost;
opt_cost = min(CHCcost);
q = 0;
j = 0;
while q ~= opt_cost
    j = j + 1;
    q = CHCcost(j);
end
x_hat = x;
clear x
end
x = x_hat;
%}

% Plots ------------------------------------------------------------------
figure,
plot(y_pred)
hold on
plot(x)
title(sprintf('CHC Optimization (Home %d)',home_num))
xlabel('Timestep (15min)')
ylabel('Energy (kW15min)')
legend('Energy Demand (Predicted)','Energy Supplied')
txt = sprintf(...
    'a = %.02f $/kW15min \nb = %.02f $/kW15min \nOptimal Cost: %.03f\nCommitment Level: %d'...
    ,a,b,cost,v);
text(T*0.6,max(y_pred)*.15,txt)

%{
% Plot for optimal window
figure,
plot(CHCcost)
hold on
scatter(j,opt_cost)
title(sprintf('CHC Optimal Window (Home %d)',home_num))
legend('Objective Function Cost')
xlabel('Committment (v)')
ylabel('Objective Function Cost')
txt = sprintf('Optimal Commitment = %d',j);
text(3,max(CHCcost)-5,txt)
%}

% Functions --------------------------------------------------------------
% The gradient is represented by the derivative of the objective function
function df = grad(x,x0,y,p,a,b)
if y < x
    df = p + b*((x - x0)/(abs(x0 - x)));
else
    df = p - a + b*((x - x0)/abs(x0 - x));
end
end




