# Python Code by Carlo Valdes
import time as time
t = time.time()

# Numerical solution and speed tests of "Controlling a Distribution of Heterogeneous Agents", Nuno G. and Moll B., Nov. 2015, 
#def code(tax):
    
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags
from scipy.sparse import identity


tax = 0.0873
# PARAMETERS        
g_bar = 0.00;           # long-run growth rate 
ga    = 2.0;            # CRRA utility with parameter gamma
alpha = 0.36;           # Production function F = K^alpha * L^(1-alpha) 
eta   = 0.02;           # death probability
delta = 0.08;           # Capital depreciation 10# + growth rate - eath rate
zmean = 1.038;          # mean O-U process (in levels). This parameter has to be adjusted to ensure that the mean of z (truncated gaussian) is 1.
Corr  = 0.4;            # persistence   O-U
sig2  = (0.2)**2 * (1- (1- Corr)**2);       # sigma^2 O-U
rho   = 0.04;    # discount rate + detah prob + (1-gamma)* growth_rate  
K     = 5;  # initial aggregate capital. It is important to guess a value close to the solution for the algorithm to converge
relax = 0.99; # relaxation parameter 
#relax = 0.90; # relaxation parameter
I    = 300; # number of a points 
J    = 40;    # number of z points 
zmin = 0.2;   # Range z
zmax = 1.8;
amin = 0;   #USARE  LE VIRGOLE # borrowing constraint
amax = 100.0; # range a
maxit  = 100
maxitK = 1000
crit = 10**(-6); #criterion HJB loop
critK = 1e-4;   #criterion K loop
Delta = 1000;   #delta in HJB algorithm

the = Corr;
#Var = sig2/(2*the);
  
a  = np.linspace(amin,amax,I)
a = np.array([a])
a  = a.T
da = (amax-amin)/(I-1);      
z  = np.linspace(zmin,zmax,J);   # productivity vector
dz = (zmax-zmin)/(J-1);
dz2 = dz**2;
aa = a*np.ones((1,J));
zz = np.ones((I,1))*z;
mu = the*(zmean - z);        #DRIFT (FROM ITO'S LEMMA)
s2 = sig2*np.ones((1,J))        #VARIANCE (FROM ITO'S LEMMA)

#Finite difference approximation of the partial derivatives
Vaf =  np.zeros((I,J));             
Vab =  np.zeros((I,J));        
Vzf =  np.zeros((I,J));        
Vzb =  np.zeros((I,J));        
Vzz =  np.zeros((I,J));        
c   =  np.zeros((I,J));        

#CONSTRUCT MATRIX Aswitch SUMMARIZING EVOLUTION OF z
yy = - s2/dz2 - mu/dz;
chi =  s2/(2*dz2);
zeta = mu/dz + s2/(2*dz2);

updiag=np.zeros((I,1))

for j in range(J):
    updiag= np.vstack((updiag,np.matlib.repmat(zeta[0,j],I,1))) 
    

centdiag= np.matlib.repmat(chi[0,0]+yy[0,0],I,1);

for j in range (1,J-1):
    centdiag=np.vstack((centdiag,np.matlib.repmat(yy[0,j],I,1)));
#np.vstack((a,b))
centdiag= np.vstack((centdiag,np.matlib.repmat(yy[0,J-1]+zeta[0,J-1],I,1)));
   #centdiag=[centdiag;repmat(yy(J)+zeta(J),I,1)];
lowdiag=np.matlib.repmat(chi[0,1],I,1);

for j in range (2,J):
    lowdiag=np.vstack((lowdiag,np.matlib.repmat(chi[0,j],I,1)));
    
    
  #Aswitch=spdiags(centdiag,0,I*J,I*J)+spdiags(lowdiag,-I,I*J,I*J)+spdiags(updiag,I,I*J,I*J);
  
#Aswitch = spdiags(centdiag, 0, I*J, I*J) 
updiag_tr = updiag.T
centdiag_tr = centdiag.T
lowdiag_tr = lowdiag.T

Aswitch = spdiags(centdiag_tr, 0, I*J, I*J) + spdiags(lowdiag_tr, -I, I*J, I*J) + spdiags(updiag_tr, I, I*J, I*J)
#plt.spy(Aswitch,precision=0.000001, markersize=1)
   
#plt.spy(Aswitch)
r = alpha     * K**(alpha-1) - delta; #interest rates
w = (1-alpha) * K**(alpha);          #wages
v0 = (w*zz + ((1-tax)*r + eta)*aa + tax*r*K)**(1-ga)/(1-ga)/(rho+eta);
v = v0;
dist = np.zeros((1,maxit));

for iter in range(0,maxitK):
    
    for n in range(0,maxit):
        
        V = v;
        # forward difference
        Vaf[0:I-1,:] = (V[1:I,:]-V[0:I-1,:])/da;
        Vaf[I-1,:] = (w*z + ((1-tax)*r + eta)*amax + tax*r*K)**(-ga);
        # backward difference
        Vab[1:I,:] = (V[1:I,:]-V[0:I-1,:])/da;
        Vab[0,:] = (w*z + ((1-tax)*r + eta)*amin + tax*r*K)**(-ga);
        
        cf = Vaf**(-1/ga);
        sf = w*zz + ((1-tax)*r + eta)*aa - cf + tax*r*K;
        
        cb = Vab**(-1/ga);
        sb = w*zz + ((1-tax)*r + eta)*aa - cb + tax*r*K;
        
        c0 = w*zz + ((1-tax)*r + eta)*aa + tax*r*K;
        Va0 = c0**(-ga);
        
        If = sf > 0
        Ib = sb < 0; #negative drift --> backward difference
        I0 = (1-If-Ib); #at steady state

        Va_Upwind = Vaf*If + Vab*Ib + Va0*I0; #important to include third term
        c = Va_Upwind**(-1/ga);
        u = (c**(1-ga))/(1-ga);
        
#        X = np.zeros((I,J));
 #       X[:,:] = - (np.minimum(sb[:,:],0))/da;
        X = - (np.minimum(sb,0))/da;
        Y = - np.maximum(sf,0)/da + np.minimum(sb,0)/da - eta;
        Z = np.maximum(sf,0)/da;
        
        updiag = np.array([0]).reshape(1, 1)
        for j in range(J):
            updiag=np.vstack((updiag,np.array(Z[0:I-1,j]).reshape((I-1,1), order='F')));
            updiag=np.vstack((updiag,0))

        #centdiag = np.reshape(Y, (I*J, 1))
        centdiag = np.reshape(Y, (I*J, 1), order='F')
        lowdiag=np.array(X[1:I,0]).reshape((I-1, 1), order='F');
#        
        temp = np.array([0]).reshape(1, 1)
        for j in range(1,J):
            lowdiag = np.vstack((lowdiag,temp))
            temp_2 = np.array(X[1:I,j]).reshape((I-1,1), order='F')
            lowdiag = np.vstack((lowdiag,temp_2))
#    
            
        
        updiag_st = np.vstack((updiag,temp))
        updiag_tr = updiag_st.T;
        centdiag_tr = centdiag.T;
        lowdiag_st = np.vstack((lowdiag,temp))
        lowdiag_tr = lowdiag_st.T;
      
        AA= spdiags(centdiag_tr, 0, I*J, I*J) + spdiags(updiag_tr, 1, I*J, I*J) + spdiags(lowdiag_tr, -1, I*J, I*J)
        #plt.spy(Aswitch,precision=0.000001, markersize=1)
        
        A = AA+ Aswitch;
        #plt.spy(A,precision=0.000001, markersize=1)
        
        u_stacked = np.array(np.reshape(u, (I*J, 1), order='F'))
        V_stacked = np.array(np.reshape(V, (I*J, 1), order='F'))
        b = u_stacked + V_stacked/Delta;    
        #b_temp = np.ones(10000)

        temp4 = 1.0/Delta + rho
 
        B = temp4 * (sp.sparse.identity(I*J))-A

        V_stacked = sp.sparse.linalg.spsolve(B, b)

        
        #print('fatto')
        V = np.reshape(V_stacked,(I,J), order='F');
        Vchange = V - v
        v = V;
        absch = np.absolute(Vchange)
        if absch.max() < crit:
            break
        
    AT = A.T
    b = np.zeros((I*J)).reshape(I*J,1);
    x_tempo = np.ones(I*J)
    y_tempo = sp.sparse.spdiags(x_tempo, 0, I*J, I*J).toarray()
    #AT  = AT - eta * (sp.sparse.identity(I*J)); 
    
    i_fix   = 0;
    b[i_fix]= -1;
    gg = sp.sparse.linalg.spsolve(AT, b)
    
    g_sum = np.dot(gg.T, (np.ones((I*J)).reshape(I*J,1))) *da*dz
    gg = np.divide(gg, g_sum)
    
    g = np.reshape(gg,(I,J),order='F')
    
    S= np.sum(np.dot(g.T, a) * da * dz)

    del(A,AA,AT, B)
    if np.absolute(K-S)<critK:
        break
    
    K = relax*K +(1-relax)*S;           
    r = alpha     * K**(alpha-1) -delta; 
    w = (1-alpha) * K**(alpha);  


Welfare  = -1.0/rho* np.sum(((np.multiply(g, ((c**(1-ga))/(1-ga)))) *dz *da));
Welfare2 = 1.0/rho * np.sum(((np.multiply(g, v))) *dz *da)

#%NUMERICAL PARETO SLOPE
#logg = np.log((g[0.1*I:0.9*I,:].sum(axis=1))*dz);
#loga = np.log(a[0.1*I:0.9*I,:]);
#
#len_logg = len(logg);
#len_loga = len(loga);
#
#slope_numerical = -((logg[len_logg-1] - logg[0])/(loga[len_loga-1] - loga[0])+1);




#    return Welfare
#    
#import scipy 
#from scipy.optimize import minimize
#
#tax_opt = scipy.optimize.minimize(code, 0.08)

elapsed = time.time() - t
    
elapsed
