

%function Welfare = competitive_equilibrium(tax)
clear all; close all; %format short;
tic;
                            % opt wealth tax 0.010428750000000 % Requires at least 2 minutes for 10, 249 sec for 20
N_sim = 15;
max_tax = 0.2514;             % OPT TAX =  0.2514
for index = 1:N_sim

%--------------------------------------------------
%PARAMETERS
g_bar = 0.0;          % long-run growth rate 
ga    = 2;             % CRRA utility with parameter gamma
alpha = 0.36;          % Production function F = K^alpha * L^(1-alpha) 
eta   = 0.02;          % death probability
delta = 0.08;  % Capital depreciation 10% + growth rate - eath rate
zmean = 1.038;           % mean O-U process (in levels). This parameter has to be adjusted to ensure that the mean of z (truncated gaussian) is 1.
Corr  = 0.4;           % persistence   O-U
sig2  = 0.2^2 * (1 - (1-Corr)^2);  % sigma^2 O-U
rho   = 0.04;    % discount rate + detah prob + (1-gamma)* growth_rate
tax_range = linspace(0.0,max_tax,N_sim);
tax     = tax_range(1,index);
%tax   = 0.00                          % Wealth tax, optimum  0.0104

K     = 5;  % initial aggregate capital. It is important to guess a value close to the solution for the algorithm to converge
relax = 0.90; % relaxation parameter 
I    = 300; % number of a points 
J    = 40;    % number of z points 
zmin = 0.2;   % Range z
zmax = 1.8;
amin = -2;   % borrowing constraint
amax = 100; % range a
amin_mod = 0;
%simulation parameters
maxit  = 100;   %maximum number of iterations in the HJB loop
maxitK = 1000;   %maximum number of iterations in the K loop
crit = 10^(-6); %criterion HJB loop
critK = 1e-4;   %criterion K loop
Delta = 1000;   %delta in HJB algorithm

%ORNSTEIN-UHLENBECK IN LEVELS
the = Corr;
%Var = sig2/(2*the);

%--------------------------------------------------
%VARIABLES 
a  = linspace(amin,amax,I)';  %wealth vector
da = (amax-amin)/(I-1);      
z  = linspace(zmin,zmax,J);   % productivity vector
dz = (zmax-zmin)/(J-1);
dz2 = dz^2;
aa = a*ones(1,J);
zz = ones(I,1)*z;

mu = the*(zmean - z);        %DRIFT (FROM ITO'S LEMMA)
s2 = sig2.*ones(1,J);        %VARIANCE (FROM ITO'S LEMMA)

%Finite difference approximation of the partial derivatives
Vaf = zeros(I,J);             
Vab = zeros(I,J);
% Vzf = zeros(I,J);
% Vzb = zeros(I,J);
% Vzz = zeros(I,J);
% c   = zeros(I,J);

%CONSTRUCT MATRIX Aswitch SUMMARIZING EVOLUTION OF z
yy = - s2/dz2 - mu/dz;
chi =  s2/(2*dz2);
zeta = mu/dz + s2/(2*dz2);

%This will be the upperdiagonal of the matrix Aswitch
updiag=zeros(I,1); %This is necessary because of the peculiar way spdiags is defined.
for j=1:J
    updiag=[updiag;repmat(zeta(j),I,1)];
end

%This will be the center diagonal of the matrix Aswitch
centdiag=repmat(chi(1)+yy(1),I,1);
for j=2:J-1
    centdiag=[centdiag;repmat(yy(j),I,1)];
end
centdiag=[centdiag;repmat(yy(J)+zeta(J),I,1)];

%This will be the lower diagonal of the matrix Aswitch
lowdiag=repmat(chi(2),I,1);
for j=3:J
    lowdiag=[lowdiag;repmat(chi(j),I,1)];
end

%Add up the upper, center, and lower diagonal into a sparse matrix
Aswitch=spdiags(centdiag,0,I*J,I*J)+spdiags(lowdiag,-I,I*J,I*J)+spdiags(updiag,I,I*J,I*J);

%----------------------------------------------------
%INITIAL GUESS
r = alpha     * K^(alpha-1) -delta; %interest rates
w = (1-alpha) * K^(alpha);          %wages
%v0 = (w*zz + (r +eta-tax).*aa + tax*K).^(1-ga)/(1-ga)/(rho+eta);
aa_mod = aa;

for i=1:I
    for j = 1:J
        if aa_mod(i,j) <0
            aa_mod(i,j) = 0;
        end
    end
end
g = 0;

v0 = ((w*zz + 1.*r.*aa -tax.*r.*aa_mod +eta.*aa) + tax.*r.*sum(sum(aa_mod.*g.*dz.*da))).^(1-ga)/(1-ga)/(rho+eta); 

%v0 = (w*zz + (((1-tax)*r) +eta-tax).*aa).^(1-ga)/(1-ga)/(rho+eta);
%(1-TAX)R*AA

v = v0;
dist = zeros(1,maxit);

%-----------------------------------------------------
%MAIN LOOP
for iter=1:maxitK
%disp('Main loop iteration')
%disp(iter)

        % HAMILTON-JACOBI-BELLMAN EQUATION %
    for n=1:maxit
        V = v;
        % forward difference
        Vaf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
        Vaf(I,:) = (w*z + ((1-tax).*r +eta).*amax  + tax.*r.*sum(sum(aa_mod.*g.*dz.*da))).^(-ga); %will never be used, but impose state constraint a<=amax just in case
        % backward difference
        Vab(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
        Vab(1,:) = (w*z + (r + eta).*amin  + tax.*r.*sum(sum(aa_mod.*g.*dz.*da))).^(-ga);  %state constraint boundary condition

        %I_concave = Vab > Vaf;              %indicator whether value function is concave (problems arise if this is not the case)

        %consumption and savings with forward difference
        cf = Vaf.^(-1/ga);
        %sf = w*zz + ((1-tax).*r +eta).*aa - cf +  tax*r*K; %ORIGINAL
        sf = w*zz +  1.*r.*aa -tax.*r.*aa_mod +eta.*aa - cf + tax.*r.*sum(sum(aa_mod.*g.*dz.*da)); %ORIGINAL

        %consumption and savings with backward difference
        cb = Vab.^(-1/ga);
        sb = w*zz +  1.*r.*aa -tax.*r.*aa_mod +eta.*aa - cb + tax.*r.*sum(sum(aa_mod.*g.*dz.*da));
        %consumption and derivative of value function at steady state
        c0 = w*zz +  1.*r.*aa -tax.*r.*aa_mod +eta.*aa + tax.*r.*sum(sum(aa_mod.*g.*dz.*da));
        Va0 = c0.^(-ga);

        % dV_upwind makes a choice of forward or backward differences based on
        % the sign of the drift
        If = sf > 0; %positive drift --> forward difference
        Ib = sb < 0; %negative drift --> backward difference
        I0 = (1-If-Ib); %at steady state
        %make sure backward difference is used at amax
        %     Ib(I,:) = 1; If(I,:) = 0;
        %STATE CONSTRAINT at amin: USE BOUNDARY CONDITION UNLESS sf > 0:
        %already taken care of automatically

        Va_Upwind = Vaf.*If + Vab.*Ib + Va0.*I0; %important to include third term

        c = Va_Upwind.^(-1/ga);
        u = c.^(1-ga)/(1-ga);

        %CONSTRUCT MATRIX A
        X = - min(sb,0)/da;
        Y = - max(sf,0)/da + min(sb,0)/da - eta;
        Z = max(sf,0)/da;
        
        updiag=0; %This is needed because of the peculiarity of spdiags.
        for j=1:J
            updiag=[updiag;Z(1:I-1,j);0];
        end
        
        centdiag=reshape(Y,I*J,1);
        
        lowdiag=X(2:I,1);
        for j=2:J
            lowdiag=[lowdiag;0;X(2:I,j)];
        end
        
        AA=spdiags(centdiag,0,I*J,I*J)+spdiags([updiag;0],1,I*J,I*J)+spdiags([lowdiag;0],-1,I*J,I*J);
        sum(sum(AA));
        A = AA + Aswitch;
        B = (1/Delta + rho)*speye(I*J) - A;

        u_stacked = reshape(u,I*J,1);
        V_stacked = reshape(V,I*J,1);

        b = u_stacked + V_stacked/Delta;

        V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS

        V = reshape(V_stacked,I,J);

        Vchange = V - v;
        v = V;

        dist(n) = max(max(abs(Vchange)));
        if dist(n)<crit
            %disp('Value Function Converged, Iteration = ')
            %disp(n)
            break
        end
    end
    %toc;
    % FOKKER-PLANCK EQUATION %
    AT = A';
    b = zeros(I*J,1);
    %AT  = AT - eta*speye(I*J); % Modified FK to include random deaths
    
    %need to fix one value, otherwise matrix is singular
    i_fix   = 1;
    b(i_fix)= -1;
%     row = [zeros(1,i_fix-1),1,zeros(1,I*J-i_fix)];
%     AT(i_fix,:) = row;
    
    
    %Solve linear system
    gg = AT\b;
    gg = max(gg,0); % To avoid numerical errors 
    g_sum = gg'*ones(I*J,1)*da*dz;
    gg = gg./g_sum;

    g = reshape(gg,I,J);
    
    % Update aggregate capital
    S = sum(g'*a*da*dz);
    %disp(S)
   
    clear A AA AT B
    if abs(K-S)<critK
        break
    end
    
    %update prices
    K = relax*K +(1-relax)*S;           %relaxation algorithm (to ensure convergence)
    r = alpha     * K^(alpha-1) -delta; %interest rates
    w = (1-alpha) * K^(alpha);          %wages
 
end

%SOME RELEVANT STATISTICS
%WELFARE
%Welfare  = -1/rho* sum(sum(g.*c.^(1-ga)/(1-ga) *dz *da))
Welfare  = -1/(rho+eta)* sum(sum(g.*c.^(1-ga)/(1-ga) *dz *da))

Welfare2 = 1/rho* sum(sum(g.*v *dz *da));

%NUMERICAL PARETO SLOPE

logg = log(sum(g(0.1*I:0.9*I,:),2)*dz); 
loga = log(a(0.1*I:0.9*I,:)); 
slope_numerical = -( (logg(end) - logg(1))/(loga(end) - loga(1))+1);

% THEORETICAL PARETO SLOPE
slope_theoretical = eta*ga/ (r-eta+g_bar - (rho - eta + (1-ga)*g_bar) -ga * g_bar);

%close all
%%
%GRAPHS
% 
% % %SAVINGS POLICY FUNCTION
figure
%subplot(131)
ss = w*zz + 1.*r.*aa -tax.*r.*aa_mod +eta.*aa - c  + tax.*r.*sum(sum(aa_mod.*g.*dz.*da));
icut = 60;
acut = a(1:icut);
sscut = ss(1:icut,:);
set(gca,'FontSize',14)
mesh(acut,z,sscut')
view([51 34])
xlabel('Wealth, $a$','FontSize',14,'interpreter','latex')
ylabel('Productivity, $z$','FontSize',14,'interpreter','latex')
zlabel('Savings $s(a,z)$','FontSize',14,'interpreter','latex')
xlim([amin max(acut)])
ylim([zmin zmax])
title('Savings policy','FontSize',14,'interpreter','latex')


% CONSUMPTION DISTRIBUTION
%subplot(122);
figure;
icut = 60;
acut = a(1:icut);
ccut = c(1:icut,:);
set(gca,'FontSize',14)
mesh(acut,z,ccut')
view([51 34])
xlabel('Wealth, $a$','FontSize',14,'interpreter','latex')
ylabel('Productivity, $z$','FontSize',14,'interpreter','latex')
zlabel('Consumption $s(a,z)$','FontSize',14,'interpreter','latex')
xlim([amin max(acut)])
ylim([zmin zmax])
title('Consumption policy','FontSize',14,'interpreter','latex')


% WEALTH DISTRIBUTION

figure;
icut = 60;
acut = a(1:icut);
gcut = g(1:icut,:);
set(gca,'FontSize',14)
mesh(acut,z,gcut')
view([45 25])
xlabel('Wealth, $a$','FontSize',14,'interpreter','latex')
ylabel('Productivity, $z$','FontSize',14,'interpreter','latex')
zlabel('Density $f(a,z)$','FontSize',14,'interpreter','latex')
xlim([amin max(acut)])
ylim([zmin zmax])
title('Wealth-productivity distribution','FontSize',14,'interpreter','latex')

disp('Simulation n.')
disp(index)
disp('Capital')
disp(K)
disp('Output')
disp(K^alpha)
Output = K^alpha;
disp('K/Y')
disp(K^(1-alpha))
K_Y = K^(1-alpha);
disp('C')

C = sum(sum(c.*g*da*dz));
disp(C)
disp('r')
disp(100*r)
r_star = 100*r;
disp('w')
disp(w)
w_star = w;
Welfare_new = sum(sum(u.*g*da*dz));
Welfare_original = Welfare; 

% Consumption near the borrowing constraint
Cons_bor_cons_1 = sum(c(1,:))
Cons_bor_cons_me = sum(c(62,:))
Cons_bor_cons_wb = sum(c(300,:))

% Savings near the borrowing constraint
Sav_bor_cons_1 = sum(ss(1,:))
Sav_bor_cons_me = sum(ss(62,:))
Sav_bor_cons_wb = sum(ss(300,:))


aggr_split_consumption = sum(c')';

%Cons_bor_cons_2 = sum(c(90,:))

K_ev(:,index) = K;              %Capital 
r_ev(:,index) = r_star;         % Interest rate
w_ev(:,index) = w_star;         % Wages
%tail_ev(:,index) = tail;        % Tail Exponent
Welfare_ev(:,index) = Welfare_new;  % Welfare function
Welfare_or_ev(:,index)  = Welfare_original;
C_ev(:, index) = C;
Cons_bor_cons_1_ev(:,index) = Cons_bor_cons_1;
Cons_bor_cons_me_ev(:,index) = Cons_bor_cons_me;
Cons_bor_cons_wb_ev(:,index) = Cons_bor_cons_wb;

Sav_bor_cons_1_ev(:,index) = Sav_bor_cons_1;
Sav_bor_cons_me_ev(:,index) = Sav_bor_cons_me;
Sav_bor_cons_wb_ev(:,index) = Sav_bor_cons_wb;

aggr_split_consumption_ev(:,index) = aggr_split_consumption;
% 
% % variations in consuption 
% for i=1:n_sim-1
%     rate_Cons_bor_cons_1_ev =(Cons_bor_cons_1_ev(i+1)-Cons_bor_cons_1_ev(i))/Cons_bor_cons_1_ev(i);
% end

slope_numerical_ev(:,index) = slope_numerical;
%Cons_bor_cons_2_ev(:,index) = Cons_bor_cons_2;
bc_dens = sum(gcut');
bc_dens(1)
close all
end

figure;
subplot(2,3,1)
plot(tax_range, Welfare_ev);
title('Welfare')
subplot(2,3,2)
plot(tax_range, K_ev);
title('Capital')
subplot(2,3,3)
plot(tax_range, r_ev);
title('Intereset rate')
subplot(2,3,4)
plot(tax_range, w_ev);
title('Wages')
subplot(2,3,5)
plot(tax_range, C_ev);
title('Consumption')
subplot(2,3,6)
plot(tax_range, slope_numerical_ev);
title('Pareto exponent')
% subplot(4,3,7)
% plot(tax_range, Cons_bor_cons_1_ev);
% title('C near the BC')
% subplot(4,3,8)
% plot(tax_range, Cons_bor_cons_me_ev);
% title('C median')
% subplot(4,3,9)
% plot(tax_range, Cons_bor_cons_wb_ev);
% title('C Warren Buffet')
% subplot(4,3,10)
% plot(tax_range, Sav_bor_cons_1_ev);
% title('SS near the BC')
% subplot(4,3,11)
% plot(tax_range, Sav_bor_cons_me_ev);
% title('SS median')
% subplot(4,3,12)
% plot(tax_range, Sav_bor_cons_wb_ev);
% title('SS Warren Buffet')

figure;
plot(tax_range, Welfare_ev);
title('Aggregate Utility')
xlabel('Capital income taxation, $tau$','FontSize',14,'interpreter','latex')
ylabel('Aggregate Utility, $U$','FontSize',14,'interpreter','latex')

figure;
subplot(1,3,1)
plot(tax_range, Cons_bor_cons_1_ev);
title('C near the BC')
subplot(1,3,2)
plot(tax_range, Cons_bor_cons_me_ev);
title('C median')
subplot(1,3,3)
plot(tax_range, Cons_bor_cons_wb_ev);
title('C Warren Buffet')

disp('Taxed Interest rate:');
disp(r*(1-tax));
disp('rho:');
disp(rho);
% PLOT SPLITTED AGGREGATE CONSUMPTION

% var_agg_cons = zeros(300,N_sim-1);
% for i=1:N_sim-1
%     for j=1:300
%     var_agg_cons(j,i) = (aggr_split_consumption_ev(j,i+1)./ aggr_split_consumption_ev(j,i))-1;%/aggr_split_consumption_ev(j,i);
%     end
% end
% figure;
% for i=1:10
%     plot(tax_range(1:end-1), var_agg_cons((i*10),:))
%     hold on
% end
% figure;
% for i=1:30
%     plot(tax_range, aggr_split_consumption_ev(i,:))
%     hold on
% end
% rate_Cons_bor_cons_1_ev = zeros(N_sim, 1);
% rate_Cons_bor_cons_me_ev = zeros(N_sim, 1);
% rate_Cons_bor_cons_wb_ev = zeros(N_sim, 1);
% variations in consuption 
% for i=1:N_sim-1
%     %rate_Cons_bor_cons_1_ev(i+1) =(Cons_bor_cons_1_ev(i+1)-Cons_bor_cons_1_ev(i))/Cons_bor_cons_1_ev(i);
%     rate_Cons_bor_cons_1_ev(1,i+1) = Cons_bor_cons_1_ev(1,i+1)/Cons_bor_cons_1_ev(1,i)
%     rate_Cons_bor_cons_me_ev(i+1) =(Cons_bor_cons_me_ev(i+1)-Cons_bor_cons_me_ev(i))/Cons_bor_cons_me_ev(i);
%     rate_Cons_bor_cons_wb_ev(i+1) =(Cons_bor_cons_wb_ev(i+1)-Cons_bor_cons_wb_ev(i))/Cons_bor_cons_wb_ev(i);
% end
% % 
% figure;
% plot(tax_range, rate_Cons_bor_cons_1_ev)
% legend('bc')
% hold on 
% plot(tax_range, rate_Cons_bor_cons_me_ev)
% hold on
% plot(tax_range, rate_Cons_bor_cons_wb_ev)
% hold off
% title('Consuption differentials due to taxation')


% %%
% figure
% set(gca,'FontSize',12)
% loglog(a(0.15*I:0.85*I),sum(g(0.15*I:0.85*I,:),2)*dz,'b','Linewidth',2)
% grid on
% hold on
% xlabel('Wealth, $a$','FontSize',14,'interpreter','latex')
% ylabel('Wealth density $f(a)$','FontSize',14,'interpreter','latex')
% axis 'tight'
