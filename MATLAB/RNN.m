clear
clc

% dataset
data = csvread('C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\Data.csv');

% input
y = data(1:end-1,:);

% output
t = data(2:end,:);

% creating and training RNN
p = con2seq(y);
t = con2seq(t);
lrn_net = newlrn(p,t,8);
lrn_net.trainFcn = 'trainbr';
lrn_net.trainParam.show = 5;
lrn_net.trainParam.epochs = 50;
lrn_net = train(lrn_net,p,t);

% plotting
y = sim(lrn_net,p);
plot(cell2mat(y));