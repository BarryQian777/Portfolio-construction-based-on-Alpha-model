temp1 = importdata('Input Monthly Data.xlsx'); 
data1 = temp1.data; 
tempdata = importdata('int rate monthly.xlsx');
data2=tempdata/100;

%(1)Loop regression for beta with vix
output_vix_beta= zeros(96,30);
output_vix_alpha=zeros(96,30);
coe_beta1=zeros(2,30);
for i=1:30
 for j=1:96
 vix = data1(j:j+23,33);
 rf1 = data2(j:j+23,:);
 Y1 = data1(j:j+23,i)-rf1;
 X1 = [ones(length(vix),1) vix];
 coe_beta1 = regress(Y1,X1);
 output_vix_beta(j,i) = coe_beta1(2,1);
 output_vix_alpha(j,i)=coe_beta1(1,1);
 end
end
beta1=output_vix_beta;
miu_alpha1=transpose(mean(output_vix_alpha));
miu_beta1=transpose(mean(beta1));
md4 = fitlm(vix,Y1)

%cross section backtest
output_vix=zeros(96,1);
for k=1:96
    X2=transpose(beta1(k,1:30));
    X2_=[ones(length(X2),1) X2];
    Y2=transpose(data1(k+23,1:30)-data2(k+23,:));
    coe_vix=regress(Y2,X2_);
    output_vix(k,:)=coe_vix(2,:);
end
miu_vix=mean(output_vix);
std_vix=std(output_vix)/sqrt(96);
t_statistic_vix=(miu_vix)/(std_vix);


%(2)Loop regression for beta with construction factor turnover
output_factorturnover_beta= zeros(96,30);
output_factorturnover_alpha=zeros(96,30);
coe_beta2=zeros(2,30);
for m=1:30
 for n=1:96
 factorturnover = data1(n:n+23,36);
 rf2 = data2(n:n+23,:);
 Y3 = data1(n:n+23,m)-rf2;
 X3 = [ones(length(factorturnover),1) factorturnover];
 coe_beta2 = regress(Y3,X3);
 output_factorturnover_beta(n,m) = coe_beta2(2,1);
 output_factorturnover_alpha(n,m)=coe_beta2(1,1);
 end
end
beta2=output_factorturnover_beta;
miu_alpha2=transpose(mean(output_factorturnover_alpha));
miu_beta2=transpose(mean(beta2));
md3 = fitlm(factorturnover,Y3)

%cross section backtest
output_factorturnover=zeros(96,1);
for k1=1:96
    X4=transpose(beta2(k1,1:30));
    X4_=[ones(length(X4),1) X4];
    Y4=transpose(data1(k1+23,1:30)-data2(k1+23,:));
    coe_factorturnover=regress(Y4,X4_);
    output_factorturnover(k1,:)=coe_factorturnover(2,:);
end
miu_factorturnover=mean(output_factorturnover);
std_factorturnover=std(output_factorturnover)/sqrt(96);
t_statistic_factorturnover=(miu_factorturnover)/(std_factorturnover);


%(3)Loop regression for beta with vix and factor turnover rate
output_turnover_beta4= zeros(96,30);
output_vix_beta4=zeros(96,30);
output_twofactors_alpha=zeros(96,30);
coe_beta5=zeros(4,30);
for p2=1:30
 for q2=1:96
 vol = data1(q2:q2+23,34);
 factorturnover3 = data1(q2:q2+23,36);
 vix3 = data1(q2:q2+23,33);
 rf5 = data2(q2:q2+23,:);
 Y9 = data1(q2:q2+23,p2)-rf5;
 X9= [ones(length(factorturnover3),1) factorturnover3 vix3];
 coe_beta5 = regress(Y9,X9);
 output_turnover_beta4(q2,p2) = coe_beta5(2,1);
 output_vix_beta4(q2,p2)=coe_beta5(3,1);
 output_twofactors_alpha(q2,p2)=coe_beta5(1,1);
 end
end
beta_turnover3=output_turnover_beta4;
beta_vix3=output_vix_beta4;
miu_turnover_beta3=transpose(mean(beta_turnover3));
miu_vix_beta3=transpose(mean(beta_vix3));
miu_alpha5=transpose(mean(output_twofactors_alpha));


%cross section backtest
output_vix_twofactors=zeros(96,1);
output_turnover3=zeros(96,1);
for k4=1:96
    X10_turnover=transpose(beta_turnover3(k4,1:30));
    X10_vix=transpose(beta_vix3(k4,1:30));
    X10=[ones(length(X10_turnover),1) X10_turnover X10_vix];
    Y10=transpose(data1(k4+23,1:30)-data2(k4+23,:));
    coe_twofactors=regress(Y10,X10);
    output_turnover3(k4,:)=coe_twofactors(2,:);
    output_vix_twofactors(k4,:)=coe_twofactors(3,:);
end
miu_turnover3=mean(output_turnover3);
std_turnover3=std(output_turnover3)/sqrt(96);
t_statistic_turnover_twofactors=(miu_turnover3)/(std_turnover3);
miu_vix_twofactors=mean(output_vix_twofactors);
std_vix_twofactors=std(output_vix_twofactors)/sqrt(96);
t_statistic_vix_twofactors=(miu_vix_twofactors)/(std_vix_twofactors);
 
 
%(4)Loop regression for beta with vix and factor turnover rate(adding market return factor)
output_turnover_beta= zeros(96,30);
output_vix_beta2=zeros(96,30);
output_market_beta=zeros(96,30);
output_threefactors_alpha=zeros(96,30);
coe_beta3=zeros(4,30);
for p=1:30
 for q=1:96
 rm3 = data1(q:q+23,31);
 factorturnover1 = data1(q:q+23,36);
 vix1 = data1(q:q+23,33);
 rf3 = data2(q:q+23,:);
 Y5 = data1(q:q+23,p)-rf3;
 X5= [ones(length(factorturnover1),1) factorturnover1 vix1 rm3-rf3];
 coe_beta3 = regress(Y5,X5);
 output_turnover_beta(q,p) = coe_beta3(2,1);
 output_vix_beta2(q,p)=coe_beta3(3,1);
 output_market_beta(q,p)=coe_beta3(4,1);
 output_threefactors_alpha(q,p)=coe_beta3(1,1);
 end
end
beta_turnover=output_turnover_beta;
beta_vix=output_vix_beta2;
beta_market=output_market_beta;
miu_turnover_beta=transpose(mean(beta_turnover));
miu_vix_beta=transpose(mean(beta_vix));
miu_market_beta=transpose(mean(beta_market));
miu_alpha3=transpose(mean(output_threefactors_alpha));
x=[factorturnover1 vix1 rm3-rf3];
mdl = fitlm(x,Y5)



%cross section backtest
output_vix_threefactors=zeros(96,1);
output_turnover=zeros(96,1);
output_market=zeros(96,1);
for k2=1:96
    X6_turnover=transpose(beta_turnover(k2,1:30));
    X6_vix=transpose(beta_vix(k2,1:30));
    X6_market=transpose(beta_market(k2,1:30));
    X6=[ones(length(X6_turnover),1) X6_turnover X6_vix X6_market];
    Y6=transpose(data1(k2+23,1:30)-data2(k2+23,:));
    coe_twofactors=regress(Y6,X6);
    output_turnover(k2,:)=coe_twofactors(2,:);
    output_vix_threefactors(k2,:)=coe_twofactors(3,:);
    output_market(k2,:)=coe_twofactors(4,:);
end
miu_turnover=mean(output_turnover);
std_turnover=std(output_turnover)/sqrt(96);
t_statistic_turnover=(miu_turnover)/(std_turnover);
miu_vix_threefactors=mean(output_vix_threefactors);
std_vix_threefactors=std(output_vix_threefactors)/sqrt(96);
t_statistic_vix_threefactors=(miu_vix_threefactors)/(std_vix_threefactors);
miu_market=mean(output_market);
std_market=std(output_market)/sqrt(96);
t_statistic_market=(miu_market)/(std_market);

 %factorturnover is more significant in the full model, and vix is more
 %significant alone
 %get adjusted R-Squared 0.242
 
 
%test of adding trading volume change of Nasdaq to model
%(5)Loop regression for beta with vix and factor turnover(adding trading volume change factor)
output_turnover_beta3= zeros(96,30);
output_vix_beta3=zeros(96,30);
output_vol_beta=zeros(96,30);
output_threefactors2_alpha=zeros(96,30);
coe_beta4=zeros(4,30);
for p1=1:30
 for q1=1:96
 vol = data1(q1:q1+23,34);
 factorturnover2 = data1(q1:q1+23,36);
 vix2 = data1(q1:q1+23,33);
 rf4 = data2(q1:q1+23,:);
 Y7 = data1(q1:q1+23,p1)-rf4;
 X7= [ones(length(factorturnover2),1) factorturnover2 vix2 vol];
 coe_beta4 = regress(Y7,X7);
 output_turnover_beta3(q1,p1) = coe_beta4(2,1);
 output_vix_beta3(q1,p1)=coe_beta4(3,1);
 output_vol_beta(q1,p1)=coe_beta4(4,1);
 output_threefactors2_alpha(q1,p1)=coe_beta4(1,1);
 end
end
beta_turnover2=output_turnover_beta3;
beta_vix2=output_vix_beta3;
beta_vol=output_vol_beta;
miu_turnover_beta2=transpose(mean(beta_turnover2));
miu_vix_beta2=transpose(mean(beta_vix2));
miu_vol_beta=transpose(mean(beta_vol));
miu_alpha4=transpose(mean(output_threefactors2_alpha));
x1=[factorturnover2 vix2 vol];
md2 = fitlm(x1,Y7)

%cross section backtest
output_vix_threefactors2=zeros(96,1);
output_turnover2=zeros(96,1);
output_vol=zeros(96,1);
for k3=1:96
    X8_turnover=transpose(beta_turnover2(k3,1:30));
    X8_vix=transpose(beta_vix2(k3,1:30));
    X8_vol=transpose(beta_vol(k3,1:30));
    X8=[ones(length(X8_turnover),1) X8_turnover X8_vix X8_vol];
    Y8=transpose(data1(k3+23,1:30)-data2(k3+23,:));
    coe_twofactors2=regress(Y8,X8);
    output_turnover2(k3,:)=coe_twofactors2(2,:);
    output_vix_threefactors2(k3,:)=coe_twofactors2(3,:);
    output_vol(k3,:)=coe_twofactors2(4,:);
end
miu_turnover2=mean(output_turnover2);
std_turnover2=std(output_turnover2)/sqrt(96);
t_statistic_turnover2=(miu_turnover2)/(std_turnover2);
miu_vix_threefactors2=mean(output_vix_threefactors2);
std_vix_threefactors2=std(output_vix_threefactors2)/sqrt(96);
t_statistic_vix_threefactors2=(miu_vix_threefactors2)/(std_vix_threefactors2);
miu_vol=mean(output_vol);
std_vol=std(output_vol)/sqrt(96);
t_statistic_vol=(miu_vol)/(std_vol);


%optimization
Return_predicted=zeros(96,30);
alpha=zeros(96,30);
x_threefactors=zeros(30,96);
TE_threefactors=zeros(96,1);
IR_threefactors=zeros(96,1);
Return_portfolio=zeros(96,1);
Return_benchmark=zeros(96,1);
Return_portfolio_post=zeros(95,1);
Return_benchmark_post=zeros(95,1);

for i2=1:96 %26-121 month
    cov_matrix=cov(data1(i2+1:i2+24,1:30));
     for j2=1:30
     Return_predicted(i2,j2)=beta_turnover2(i2,j2)*data1(i2+24,36)+beta_vix2(i2,j2)*data1(i2+24,33)+beta_vol(i2,j2)*data1(i2+24,34)+output_threefactors2_alpha(i2,j2);
     end
    %ex-ante method
     for m2=1:30
     alpha(i2,m2)=Return_predicted(i2,m2)-transpose(ones(30,1)*(1/30))*transpose(Return_predicted(i2,1:30));
     end
    lambda=71.55;
    H=lambda*cov_matrix;
    f=transpose(alpha(i2,1:30));
    A=eye(30)*(-1);
    b=ones(30,1)*(1/30);
    Aeq=transpose([ones(30,1) transpose(beta_turnover2(i2,1:30)) transpose(beta_vix2(i2,1:30)) transpose(beta_vol(i2,1:30))]);
    beq=ones(4,1)*0;
    x_threefactors(1:30,i2) = quadprog(H,f,A,b,Aeq,beq);
    TE_threefactors(i2,1)=sqrt((transpose(x_threefactors(1:30,i2)))*cov_matrix*(x_threefactors(1:30,i2)));
    IR_threefactors(i2,1)=(transpose((x_threefactors(1:30,i2)+ones(30,1)*(1/30)))*transpose(Return_predicted(i2,1:30))-transpose(ones(30,1)*(1/30))*transpose(Return_predicted(i2,1:30)))/TE_threefactors(i2,1);
    Return_portfolio(i2,1)=transpose((x_threefactors(1:30,i2)+ones(30,1)*(1/30)))*transpose(Return_predicted(i2,1:30));
    Return_benchmark(i2,1)=transpose(ones(30,1)*(1/30))*transpose(Return_predicted(i2,1:30));
end
miu_TE=mean(TE_threefactors);
TE_annual=miu_TE*sqrt(12);
miu_IR=mean(IR_threefactors);
ret=1+Return_portfolio(1,1);
for t=1:95
    ret=ret*(1+Return_portfolio(t+1,1));
end
Portfolio_annual=ret^(1/8)-1;
ret_benchmark=1+Return_benchmark(1,1);
for t1=1:95
    ret_benchmark=ret_benchmark*(1+Return_benchmark(t1+1,1));
end
Benchmark_annual=(ret_benchmark)^(1/8)-1;
    
%IC
for i3=1:95
    Return_portfolio_post(i3,1)=transpose((x_threefactors(1:30,i3)+ones(30,1)*(1/30)))*transpose(data1(i3+25,1:30));
    Return_benchmark_post(i3,1)=data1(i3+25,32);
end
IC=corrcoef(Return_portfolio(1:95,1),Return_portfolio_post(1:95,1));%26-120 month
 %get adjusted R-Squared 0.336
 
