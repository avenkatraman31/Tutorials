%% Noise-free GP

clear
clc
close all

% Input space for target function
xt=linspace(0,2*pi,37)';

% Target function
yt=sin(2*xt)+1.25*cos(xt)-0.5;

% Randomly generated input points
x=2*pi*rand(7,1);

% Function values of training points
y=sin(2*x)+1.25*cos(x)-0.5;

% Characteristic length scale
l=0.01;

% Covariance of training set
C_train=zeros(length(x),length(x));
for r=1:length(x)
    for c=1:length(x)
        C_train(r,c)=exp(-0.5*(x(r)-x(c)).^2/l.^2);        
    end
end
C_train=C_train+1e-9*eye(size(C_train));

% Covariance of test set
C_test=zeros(length(x),length(x));
for r=1:length(xt)
    for c=1:length(xt)
        C_test(r,c)=exp(-0.5*(xt(r)-xt(c)).^2/l.^2);        
    end
end
C_test=C_test+1e-9*eye(size(C_test));

% Cross-covariance between test and training points
k=zeros(length(x),length(x));
for r=1:length(x)
    for c=1:length(xt)
        k(r,c)=exp(-0.5*(x(r)-xt(c)).^2/l.^2);        
    end
end

% Inverse of training covariance matrix
C_inv=(C_train)^-1;

% Mean prediction of the entire input space
m=k'*C_inv*y;
% Variance (uncertainty) of the predictions
S=C_test-k'*C_inv*k;

% Visualization
figure
h_real=plot(xt,yt,'b-','LineWidth',2);
hold on
h_mean=plot(xt,m,'r-','LineWidth',1);
h_unc=plot(xt,m+sqrt(diag(S)),'r--','LineWidth',1);
plot(xt,m-sqrt(diag(S)),'r--','LineWidth',1)
h_train=plot(x,y,'ko','MarkerFaceColor','black','MarkerSize',6);
xlim([0 2*pi])
set(gcf,'Color','w')
set(gca,'FontSize',14)
xlabel('x','FontSize',14)
ylabel('y','FontSize',14)
l=legend([h_train h_mean h_unc h_real],'Training points','Mean prediction',...
    'Uncertainty 1\sigma','Target function');
set(l,'FontSize',14)
grid on

%% Noisy GP

clear
clc
close all

% Input space for target function
xt=linspace(0,2*pi,73)';

% Target function
yt=sin(2*xt)+1.25*cos(xt)-0.5;

% Randomly generated input points
x=2*pi*rand(10,1);
% Function values of training points
y=sin(2*x)+1.25*cos(x)-0.5+0.5*randn(size(x));

% Signal noise
sig=1.5;
% Characteristic length scale
l=0.75;
% Output noise
b=0.5;

% Covariance of training set
C_train=zeros(length(x),length(x));
for r=1:length(x)
    for c=1:length(x)
        C_train(r,c)=sig^2*exp(-0.5*(x(r)-x(c)).^2/l.^2)+b.^2*eq(r,c);
    end
end
C_train=C_train+1e-9*eye(size(C_train));

% Covariance of test set
C_test=zeros(length(x),length(x));
for r=1:length(xt)
    for c=1:length(xt)
        C_test(r,c)=sig^2*exp(-0.5*(xt(r)-xt(c)).^2/l.^2)+b.^2*eq(r,c);
    end
end
C_test=C_test+1e-9*eye(size(C_test));

% Cross-covariance between test and training points
k=zeros(length(x),length(xt));
for r=1:length(x)
    for c=1:length(xt)
        k(r,c)=sig^2*exp(-0.5*(x(r)-xt(c)).^2/l.^2);
    end
end

% Inverse of training covariance matrix
C_inv=(C_train)^-1;

% Mean prediction of the entire input space
m=k'*C_inv*y;
% Variance (uncertainty) of the predictions
S=C_test-k'*C_inv*k;

% Visualization
figure
h_real=plot(xt,yt,'b-','LineWidth',2);
hold on
h_mean=plot(xt,m,'r-','LineWidth',1);
h_unc=plot(xt,m+sqrt(diag(S)),'r--','LineWidth',1);
plot(xt,m-sqrt(diag(S)),'r--','LineWidth',1)
h_train=plot(x,y,'ko','MarkerFaceColor','black','MarkerSize',6);
xlim([0 2*pi])
set(gcf,'Color','w')
set(gca,'FontSize',14)
xlabel('x','FontSize',14)
ylabel('y','FontSize',14)
l=legend([h_train h_mean h_unc h_real],'Training points','Mean prediction',...
    'Uncertainty 1\sigma','Target function');
set(l,'FontSize',14)
grid on

%% Multiple length scales

clear
clc
close all

% Generation of input grid in 2-D space
[xt,yt]=meshgrid(linspace(0,2*pi,37),linspace(0,2*pi,37));

% Characteristic length scale in x dimension
lx=0.5;
% Characteristic length scale in y dimension
ly=2.5;
% Signal noise
sig=0.5;

% Sampling from a zero mean GP to generate a function
C_sample=zeros(length(xt));
for r=1:length(xt(:))
    for c=1:length(yt(:))
        C_sample(r,c)=sig^2*exp(-0.5*(xt(r)-xt(c)).^2/lx.^2-0.5*(yt(r)-yt(c)).^2/ly.^2);
    end
end
R=mvnrnd(zeros(size(xt(:))),0.5*(C_sample+C_sample'));
% Adding noise to the function we try to predict
zt=reshape(R,size(xt))+0.2*randn(size(xt));

% Visualization of target function
subplot(1,3,1)
s=surf(xt,yt,reshape(R,size(xt)));
set(s,'FaceColor',[1 0.7 0.7],'EdgeAlpha',0.1,'FaceAlpha',0.8)
grid on
axis square
set(gca,'FontSize',14)
set(gcf,'Color','w')
xlim([0 2*pi])
ylim([0 2*pi])
xlabel('x','FontSize',14)
ylabel('y','FontSize',14)
zlabel('z','FontSize',14)

% Number of training points
nots=150;
% Use a portion of the input grid as the training set
train_ind=randperm(length(xt(:)));
% Defind the input and target points
x=xt(train_ind(1:nots)); y=yt(train_ind(1:nots)); z=zt(train_ind(1:nots));

% Extraction of the hyperparameters by using ARD SE kernel
mdl=fitrgp([x',y'],z','KernelFunction','ardsquaredexponential');
mdl.KernelInformation.KernelParameterNames
mdl.KernelInformation.KernelParameters

% Mean prediction of the input space based on ARD SE kernel
[zh,zh_sd,~]=predict(mdl,[xt(:),yt(:)]);

% Visualization of mean prediction and uncertainty based on ARD SE kernel
subplot(1,3,2)
s=surf(xt,yt,reshape(zh,size(xt)));
hold on
plot3(x,y,z,'k*')
set(s,'FaceColor',[1 0.7 0.7],'EdgeAlpha',0.1,'FaceAlpha',0.8)
s=surf(xt,yt,reshape(zh,size(xt))+reshape(zh_sd,size(xt)));
set(s,'FaceColor',[0.7 1 0.7],'EdgeAlpha',0.1,'FaceAlpha',0.2)
s=surf(xt,yt,reshape(zh,size(xt))-reshape(zh_sd,size(xt)));
set(s,'FaceColor',[0.7 1 0.7],'EdgeAlpha',0.1,'FaceAlpha',0.2)
grid on
axis square
set(gca,'FontSize',14)
set(gcf,'Color','w')
xlim([0 2*pi])
ylim([0 2*pi])
xlabel('x','FontSize',14)
ylabel('y','FontSize',14)
zlabel('z','FontSize',14)

% Extraction of hyperparameters of SE kernel (single length scale for both input dimensions)
mdl=fitrgp([x',y'],z','KernelFunction','squaredexponential');
mdl.KernelInformation.KernelParameterNames
mdl.KernelInformation.KernelParameters

% Mean prediction of the input space based on SE kernel
[zh,zh_sd,~]=predict(mdl,[xt(:),yt(:)]);

% Visualization of mean prediction and uncertainty based on SE kernel
subplot(1,3,3)
s=surf(xt,yt,reshape(zh,size(xt)));
hold on
plot3(x,y,z,'k*')
set(s,'FaceColor',[1 0.7 0.7],'EdgeAlpha',0.1,'FaceAlpha',0.8)
s=surf(xt,yt,reshape(zh,size(xt))+reshape(zh_sd,size(xt)));
set(s,'FaceColor',[0.7 1 0.7],'EdgeAlpha',0.1,'FaceAlpha',0.2)
s=surf(xt,yt,reshape(zh,size(xt))-reshape(zh_sd,size(xt)));
set(s,'FaceColor',[0.7 1 0.7],'EdgeAlpha',0.1,'FaceAlpha',0.2)
grid on
axis square
set(gca,'FontSize',14)
set(gcf,'Color','w')
xlim([0 2*pi])
ylim([0 2*pi])
xlabel('x','FontSize',14)
ylabel('y','FontSize',14)
zlabel('z','FontSize',14)

%% Local minima - Multiple solutions

clear
clc
close all

% Loading matlab data
load(fullfile(matlabroot,'examples','stats','gprdata2.mat'));

% Extraction of default kernel
gprMdl1 = fitrgp(x,y,'KernelFunction','squaredexponential');
gprMdl1.KernelInformation.KernelParameterNames
gprMdl1.KernelInformation.KernelParameters

% Prediction of points based on default kernel
ypred1 = resubPredict(gprMdl1);

% Visualization of target points and the predictions based on default kernel
figure(1);
plot(x,y,'r.');
hold on
grid on
plot(x,ypred1,'b');
xlabel('x','FontSize',14);
ylabel('y','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'Color','w')

% Specification of hyperparameters (signal noise, sigmaf and characteristic length scale, l)
params=hyperparameters('fitrgp',x,y);
params(1).Range=[0.1 1];
params(1).Optimize=true;
params(4).Range=[0.01 5];
params(4).Optimize=true;
% Extraction of the optimum hyperparameters through global optimization
gprMdl2=fitrgp(x,y,'KernelFunction','squaredexponential','OptimizeHyperparameters',params,...
    'HyperparameterOptimizationOptions',...
    struct('MaxObjectiveEvaluations',10,'ShowPlots',true,'Repartition',true));

% Prediction of points based on optimized kernel
ypred2 = resubPredict(gprMdl2);

% Visualization of predictions based on optimized kernel
figure(1)
plot(x,ypred2,'g');
l=legend('Target','Default kernel','Optimized kernel','Location','southeast');
set(l,'FontSize',14)