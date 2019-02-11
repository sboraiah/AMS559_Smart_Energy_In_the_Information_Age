% ------------------------------------------------------------------------
% AMS 559
% Author: John Buckheit
% Group 2
% 29/11/18
% Homework 2, Due: 2/12/18
% Online Gradient Descent
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

% iterate for optimal step size
%{
for j = 1:100
    stepsize = j/T;
%}
% fixed/vary a/b
%for b = 1:100

% Interval of interest
y = data(32161:32161 + T -1);

y_pred = pred;

% Costs and penalties
p = 0.4/4; % 0.4 kWh divided by 4 to give kW15min
a = 4/4;
b = 4/4;
% variables
step = 22; % Use 22,11,25 for homes 1,4,8 respectively
stepsize = step/T;
% inital values
x(1) = data(32160);
x(2) = 0;

% Determine supply based on gradient
for t = 2:T-1
    x(t+1) = x(t) - stepsize*grad(x(t),x(t-1),y_pred(t),p,a,b);
end

x = x';
cost = 0;
% find objective cost for the determined supply (x)
for t = 2:T
    cost = cost + p*x(t) + a*max(0,y(t) - x(t)) + b*abs(x(t) - x(t-1));
end

x_OGD = x;

% find optimal step size
%{
ogdcost(j) = cost;
opt_cost = min(ogdcost);
q = 0;
j = 0;
while q ~= opt_cost
    j = j + 1;
    q = ogdcost(j);
end
end
%}
% find optimal penalty
%{
ogdcost(b) = cost;
opt_cost = min(ogdcost);
q = 0;
j = 0;
while q ~= opt_cost
    j = j + 1;
    q = ogdcost(j);
end
end
%}

% Plots ------------------------------------------------------------------
figure,
plot(y_pred)
hold on
plot(x)
title(sprintf('OGD Optimization (Home %d)',home_num))
xlabel('Timestep (15min)')
ylabel('Energy (kW15min)')
legend('Energy Demand (Predicted)','Energy Supplied')
txt = sprintf(...
    'a = %.02f $/kW15min \nb = %.02f $/kW15min \nOptimal Cost: %.03f\nStep size: %d/%d'...
    ,a,b,cost,step,T);
text(T*0.6,max(y_pred)*.15,txt)

% Plot for optimal stepsize
%{
figure,
plot(ogdcost)
hold on
scatter(j,opt_cost)
title(sprintf('OGD Optimal Step Size (Home %d)',home_num))
legend('Objective Function Cost')
xlabel('Step Size / T')
ylabel('Objective Function Cost')
txt = sprintf('Optimal step size = %d/%d',j,T);
text(j,max(ogdcost)*.9,txt)
%}

% Plot for optimal a/b
%{
figure,
plot(ogdcost)
hold on
scatter(j,opt_cost)
title(sprintf('OGD Optimal Penalty (b) (Home %d)',home_num))
legend('Objective Function Cost')
xlabel('Penalty (b)')
ylabel('Objective Function Cost')
txt = sprintf('Optimal (b) = %d',j);
text(j,max(ogdcost)*.9,txt)
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




