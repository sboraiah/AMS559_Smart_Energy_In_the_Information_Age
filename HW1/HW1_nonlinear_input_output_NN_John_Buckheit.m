% AMS 559
% Author: John Buckheit
% Group 2
% 4/10/18
% Homework 1, Due: 16/10/18
% Nonlinear Input-Output Time Series Neural Network
% MATLAB version R2018b

%{
The goal of this script is to create a neural network that can accurately
predict household energy load demand. The network does this by receiving
input parameters of daily and weekly time step tags, and making a guess at
what the energy load is at that time step. This guess is compared to the
actual energy load, and the network adjusts accordingly. This is repeated
for every 15min interval. When 96 15min intervals have passed, the daily
time step input repeats its cycle, indicating a new day. This suggests to
the network that we can expect a similar energy load at the same time of
day. The same is true of the weekly time step, but with 672 15min
intervals. The success of the network is determined by the Mean Absolute
Error (MAE) between the predictions and the actual energy loads.
%}

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

data_full = data;

% Only 364 days of data are used so that the last day (365) can be used for
% one day ahead comparison.
n = 364;
if n < 365
    y_1day_ahead = data((n*96)+1:(n+1)*96); % data for 1 day ahead prediction
    condition3 = 0;
else
    y_1day_ahead = zeros(1,96);
    condition3 = 1;
end
% condition3 : only return day ahead mae if user chooses less than 365 days

% data for training/validation/testing
data = data(1:n*96); % Shorten data to view a smaller number of days

% Get daily and weekly 15min time steps for the year ---------------------
%{
It is suspected that energy usage follows some daily and weekly patterns.
So the input x(t) for this network are looped values corresponding to each
15min time step through the data, reset daily and weekly.
%}
tsd = 96; % 15 min time steps in a day
tsw = 672; % 15 min time steps in a week

% Allocate space for all daily and weekly time step inputs
weekly_time_step = zeros(1,length(data)); 
daily_time_step = zeros(1,length(data));
% Initial vector of all time steps for a week (w) and day (d)
w = (1:tsw);
d = (1:tsd);

w_iter = 1;
d_iter = 1;
count = 0; 
condition = 0; % while loop break condition
while condition == 0
    count = count + 1;
    weekly_time_step(count) = w(w_iter);
    daily_time_step(count) = d(d_iter);
    d_iter = d_iter + 1;
    w_iter = w_iter + 1;
    if w_iter > 672 % after 672 15min steps, restart week
        w_iter = 1;
    end
    if d_iter > 96 % after 96 15min steps, restart day
        d_iter = 1;
    end
    if count == (n+1)*96 % loop until enough time steps have been made to match data
        condition = 1;
    end 
end

% Get time steps for the 1 day ahead prediction
wts_1day_ahead = weekly_time_step((n*96)+1:(n+1)*96);
dts_1day_ahead = daily_time_step((n*96)+1:(n+1)*96);
% Get the time steps that will be used to train/validate/test network
weekly_time_step = weekly_time_step(1:96*n);
daily_time_step = daily_time_step(1:96*n);

% Combine daily/weekly time steps into one input matrix
time_step = [daily_time_step; weekly_time_step]';
ts_1day_ahead = [dts_1day_ahead; wts_1day_ahead]';

X = tonndata(time_step,false,false);
T = tonndata(data,false,false);

% Training Function: Levenberg-Marquardt backpropagation.
trainFcn = 'trainlm'; 

% Create a Time Delay Network
inputDelays = 1:2; % number of delays
hiddenLayerSize = 10; % number of hidden layers
% hidden layers are adjusted between 1 to 13 to minimized MAE
net = timedelaynet(inputDelays,hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Prepare the Data for Training and Simulation
[x1,xi,ai,t] = preparets(net,X,T);

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'time';  % Divide up every sample
net.divideParam.trainRatio = 70/100; % 70% training
net.divideParam.valRatio = 20/100; % 20% validation
net.divideParam.testRatio = 10/100; % 10% testing

% Choose a Performance Function
net.performFcn = 'mse';  % Mean Squared Error
% Network requires a performance function, but it for some reason it 
% returns an error if mae is chosen and defaults to MSE, so MAE
% is computed manually instead later in the script

% Choose Plot Functions
net.plotFcns = {'plotperform','plottrainstate', 'ploterrhist', ...
    'plotregression', 'plotresponse', 'ploterrcorr', 'plotinerrcorr'};

% Train the Network
[net,tr] = train(net,x1,t,xi,ai);

% Test the Network
y = net(x1,xi,ai);
e = gsubtract(t,y);
performance = perform(net,t,y); %%

% Recalculate Training, Validation and Test Performance
trainTargets = gmultiply(t,tr.trainMask);
valTargets = gmultiply(t,tr.valMask);
testTargets = gmultiply(t,tr.testMask);
trainPerformance = perform(net,trainTargets,y); %%
valPerformance = perform(net,valTargets,y); %%
testPerformance = perform(net,testTargets,y); %%

% View the Network
view(net) %

% Get one day ahead (96 time steps) prediction
ts_1day_CELL = mat2cell(ts_1day_ahead',2,ones(1,length(ts_1day_ahead)));
one_day_ahead_prediction = net(ts_1day_CELL);

% Convert back to matricies for plotting
forecast = (cell2mat(y))';
oda_prediction = (cell2mat(one_day_ahead_prediction))';

% Mean Absolute Error (MAE) ----------------------------------------------
sum = 0;
for k = 3:length(data)
    sum = sum + abs(forecast(k-2) - data(k));
end
mae = sum/(length(data) - 2)
sum = 0;
if condition3 == 0
    for k = 3:length(y_1day_ahead)
        sum = sum + abs(oda_prediction(k) - y_1day_ahead(k));
    end
    mae_one_day_ahead = sum/(length(y_1day_ahead) - 2)
end


% Plots ------------------------------------------------------------------
figure,
plot(data(1:10*96))
hold on
plot(forecast(1:10*96))
title('15min Electrical Load Demand (First 10 Days)')
xlabel('15min Time Steps')
ylabel('Electrical Load Demand')
legend('Observed','Predicted')

figure,
plot((1:n*96),data)
hold on
plot((3:n*96),forecast)
hold on
plot(((n*96)+1:(n+1)*96),y_1day_ahead)
hold on
plot(((n*96)+3:(n+1)*96),oda_prediction(3:96))
title('15min Electrical Load Demand (Full Data)')
xlabel('15min Time Steps')
ylabel('Electrical Load Demand')
legend('Observed','Predicted','Observed(One day Ahead)','Predicted(One Day Ahead')

figure,
plot(y_1day_ahead)
hold on
plot(oda_prediction(3:96))
title('15min Electrical Load Demand (One day ahead prediction)')
xlabel('15min Time Steps')
ylabel('Electrical Load Demand')
legend('Actual','Predicted')

% Autogenerated network results
figure, plotperform(tr)
figure, plottrainstate(tr)
figure, ploterrhist(e)
figure, plotregression(t,y)
figure, plotresponse(t,y)
figure, ploterrcorr(e)
