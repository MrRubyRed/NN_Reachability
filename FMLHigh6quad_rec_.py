# -*-: coding: utf-8 -*-
"""
Created on Thu May 26 13:37:12 2016

@author: cusgadmin
"""

import numpy as np
#import math
#import numpy.random as rnd
#import NLSysOne as nl
#import NLSysDos as nl
#import LSys as ls
import tensorflow as tf
from FAuxFuncs2_0 import TransDef
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import matplotlib.animation as animation
import time
import h5py
            

def main(layers,t_hor,ind,nrolls,bts,ler_r,mom,teps,renew,imp,q):
# Quad Params
    wMax = 3.0; 
    wMin = -1.0*wMax;
    aMax = 2*np.pi/10.0;
    aMin = -1.0*aMax;  

    #maxes = np.array([5.0,10.0,5.0,10.0,np.pi,10.0,1.0]); #VRR
    #maxes = np.array([2.0,2.0,2.0,2.0,2.0,2.0,1.0]);

#    def (ALL):
#	return np.divide(ALL,maxes);    
  
#    ind = 0;
    print 'Starting worker-' + str(ind)

    Nx = 41;
    minn = [-5.0, -np.pi,-5.0,6.0];
    maxx = [ 5.0, 2.988344231463461,5.0, 12.0];

#    reach100s = h5py.File('BRTx.mat'); reach100s = reach100s["BRTx"]; 
    reach100s = sio.loadmat('flat_1s.mat'); reach100s = reach100s["M"]; 
    reach100s[:,[1,2]] = reach100s[:,[2,1]];
    reach100s = np.concatenate((reach100s[:,[0,1]],np.sin(reach100s[:,2,None]),np.cos(reach100s[:,2,None]),reach100s[:,3:]),axis=1);
    #reach100s[:,6] = -1.0*reach100s[:,6];

    nofparams = 0;
    for i in xrange(len(layers)-1):
        nofparams += layers[i]*layers[i+1] + layers[i+1];
    print 'Number of Params is: ' + str(nofparams)
    
    H_length = t_hor;#-1.0; #Has to be negative                                 #VAR
    iters = 1000000;                                                            #VAR
    #center = np.array([[0.0,0.0]])
    center = np.array([[0.0,0.0,0.0,0.0,0.0,0.0]])
    ballr = 0.1;
    depth = 2.0;
    incl = 1.0;
    #######################  DEBUG  ########################
    log_loss = np.zeros((iters/1000,1));
    log_error = np.zeros((iters/1000,1));
    log_avg_error = np.zeros((iters/1000,1));
    log_pde_loss = np.zeros((iters/1000,1));
    ##################### DEFINITIONS #####################
    #layers = [2 + 1,10,1];                                                    #VAR
    ssize = layers[0] - 1;
    dt = 0.01;                                                                 #VAR
    ##################### INSTANTIATIONS #################
    states,y,Tt,l_r,lb,t_vec,reg = TransDef(layers,depth,incl,center);
    var_grad = tf.gradients(Tt,states)[0]
    theta = tf.trainable_variables();


    # DEFINE LOSS

    lmbda = 0.0;
    #L = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y,Tt)),1,keep_dims=True)));
    L = tf.reduce_mean(tf.mul(tf.exp(imp*t_vec),tf.abs(tf.sub(y,Tt)))) + lmbda*reg;

    # DEFINE OPTIMIZER

    #nu = 5.01;
    nunu = ler_r;#0.00005;
    nu = tf.placeholder(tf.float32, shape=[])                                         #VAR

    #train_step = tf.train.GradientDescentOptimizer(nu).minimize(L)
    train_step = tf.train.MomentumOptimizer(learning_rate=nu,momentum=mom).minimize(L)
    #train_step = tf.train.RMSPropOptimizer(learning_rate=nu,momentum=mom).minimize(L)
    #train_step = tf.train.AdagradOptimizer(learning_rate=nu,initial_accumulator_value=0.5).minimize(L);

    # INITIALIZE GRAPH
    sess = tf.Session();
    init = tf.initialize_all_variables();
    sess.run(init);

    def p_corr(ALL_x):
        ALL_x = np.mod(ALL_x + 1.0,2.0) - 1.0;
        return ALL_x;

    def F(grad,ALL_x):
       
       opt_dir_1_ = np.sign(-1.0*np.multiply(grad[:,2,None],ALL_x[:,3,None]) + np.multiply(grad[:,3,None],ALL_x[:,2,None]))*wMin;
       opt_dir_2_ = np.sign(grad[:,4,None])*aMin #np.floor((np.sign(grad[:,3,None])+1.0)/2.0)*aMin + np.ceil((np.sign(grad[:,3,None])-1.0)/2.0)*aMax;
       opt_dir = np.concatenate((opt_dir_1_,opt_dir_2_),axis=1);
       #opt_norm = np.linalg.norm(opt_dir,axis=1,keepdims=True) + 0.000000000001;
       #opt_dir = np.divide(opt_dir,opt_norm)*T1Max;
       
       col1 = np.multiply(ALL_x[:,4,None],ALL_x[:,2,None]);
       col2 = np.multiply(ALL_x[:,4,None],ALL_x[:,3,None]);
       col3 = -1.0*np.multiply(ALL_x[:,3,None],opt_dir[:,0,None]);
       col4 = np.multiply(ALL_x[:,2,None],opt_dir[:,0,None]);
       col5 = opt_dir[:,1,None];
       
       return np.concatenate((col1,col2,col3,col4,col5),axis=1);
       

    ####################### RECURSIVE FUNC ####################

    def V_ret(ALL_x,ALL_t,steps,d):
        
        z_t = np.zeros((nrolls,1));
        dtt = dt*np.ones((nrolls,1));
    
        b_ind = ALL_t + dtt > 0;
        dtt[b_ind] = -1.0*ALL_t[b_ind];
        ALL_dt = np.concatenate((ALL_x,ALL_t+dtt),axis=1);
        #ALL = np.concatenate((ALL_x,ALL_t),axis=1);        
        
        dV = sess.run(var_grad, feed_dict={states:ALL_dt})
        dVdx = dV[:,:-1]; #### !!!
        k1 = F(dVdx,ALL_dt);  #### !!!
        # ~~~~ Compute optimal input (k2)
        ALL_tmp = ALL_dt + np.concatenate((np.multiply(dtt/2.0,k1),z_t),axis=1);
        ALL_tmp[:,[2,3]] = p_corr(ALL_tmp[:,[2,3]]);
        dV = sess.run(var_grad, feed_dict={states:ALL_tmp});
        dVdx = dV[:,:-1];  #### !!!
        k2 = F(dVdx,ALL_tmp);  #### !!!
        # ~~~~ Compute optimal input (k3)
        ALL_tmp = ALL_dt + np.concatenate((np.multiply(dtt/2.0,k2),z_t),axis=1);
        ALL_tmp[:,[2,3]] = p_corr(ALL_tmp[:,[2,3]]);
        dV = sess.run(var_grad, feed_dict={states:ALL_tmp});
        dVdx = dV[:,:-1];  #### !!!
        k3 = F(dVdx,ALL_tmp);  #### !!!
        # ~~~~ Compute optimal input (k4)
        ALL_tmp = ALL_dt + np.concatenate((np.multiply(dtt,k3),z_t),axis=1);
        ALL_tmp[:,[2,3]] = p_corr(ALL_tmp[:,[2,3]]);
        dV = sess.run(var_grad, feed_dict={states:ALL_tmp});
        dVdx = dV[:,:-1];   #### !!!
        k4 = F(dVdx,ALL_tmp);  #### !!!
        
        Snx = ALL_x + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4));
        Snx[:,[2,3]] = p_corr(Snx[:,[2,3]]);
        
        ALL_dxdt = np.concatenate((Snx,ALL_t + dtt),axis=1);        #V(x+f(x,a*,b*)dt,t+dt)
        if (d == steps):
            V = np.min(np.concatenate((sess.run(Tt,{states:ALL_dt}),sess.run(Tt,{states:ALL_dxdt})),axis=1),axis=1,keepdims=True);
        else:
            d = d + 1;
            first = np.transpose(sess.run(Tt,{states:ALL_dt}) <= sess.run(Tt,{states:ALL_dxdt}));
            second = np.transpose(sess.run(Tt,{states:ALL_dt}) > sess.run(Tt,{states:ALL_dxdt}));
            new_ALL_x = np.zeros(ALL_x.shape);
            new_ALL_x[first[0]] = ALL_x[first[0]];
            new_ALL_x[second[0]] = Snx[second[0]];
            #V = np.min(np.concatenate((V_ret(ALL_x,ALL_t+dtt,steps,d),V_ret(Snx,ALL_t+dtt,steps,d)),axis=1),axis=1,keepdims=True);
            V = V_ret(new_ALL_x,ALL_t+dtt,steps,d);
        return V; 

    # *****************************************************************************
    #
    # ============================= MAIN LOOP ====================================
    #                     ( where the good stuff happens)    
    # *****************************************************************************
    t1 = time.time();
    k=0; kk = 0; beta=3.0; batch_size = bts; tau = 1000.0; steps = teps;
    for i in xrange(iters):
        
        if(np.mod(i,renew) == 0):
        
            #nunu = nunu*(tau/max(i,tau));
            # ~~~~ Sample points in x and t
            ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0] - 1));
            ALL_x[:,2] = ALL_x[:,2]*np.pi/5.0;
            ALL_x[:,3] = ALL_x[:,2]; 
            ALL_x[:,2] = np.sin(ALL_x[:,3]);
            ALL_x[:,3] = np.cos(ALL_x[:,3]);
            ALL_x[:,4] = ALL_x[:,3]*3.0/5.0 + 9.0;
            ALL_t = np.random.uniform(0.0,H_length,(nrolls,1));
            V = V_ret(ALL_x,ALL_t,steps,0);
            ALL = np.concatenate((ALL_x,ALL_t),axis=1);                 #V(x,t)
            #ch = np.arange(len(V));

        # |||||||||||| ----  PRINT ----- ||||||||||||    

        if(np.mod(i,1000) == 0):

            mse = sess.run(L,{states:ALL,y:V});
            log_loss[k] = mse;
            k += 1;
      
        # |||||||||||| ----  ERROR CHECKING ----- ||||||||||||

        if(np.mod(i,1000) == 0):
            nchks = 3000;
            ALL_x_ = np.random.uniform(-5.0,5.0,(nchks,layers[0] - 1));     #### !!!
            ALL_x_[:,2] = ALL_x_[:,2]*np.pi/5.0;
            ALL_x_[:,3] = ALL_x_[:,2]; 
            ALL_x_[:,2] = np.sin(ALL_x_[:,3]);
            ALL_x_[:,3] = np.cos(ALL_x_[:,3]);
            ALL_x_[:,4] = ALL_x_[:,3]*3.0/5.0 + 9.0;
            ALL_t_ = np.random.uniform(0.0,H_length,(nchks,1));
            ALL_full = np.concatenate((ALL_x_,ALL_t_),axis=1);
    
            #r_sel = np.arange(50000);
            #np.random.shuffle(r_sel)
            #r_tmp = r_sel[:50000];
            #into_nn = reach100s[:,:-1];
            #targ_nn = reach100s[:,-1,None];
            Val = np.concatenate((reach100s[:,:-1],-1.0*np.ones(reach100s[:,-1,None].shape)),axis=1);
            ZR = sess.run(Tt,{states:Val});
            error1 = ZR - reach100s[:,-1,None];
            
            #error1 = 0.0;#targ_nn - sess.run(Tt,{states:into_nn});
                    
    
            log_avg_error[kk] = np.max(np.abs(error1));
            log_error[kk] = np.mean(np.abs(error1));        
    
            # ~~ Computing PDE error
            dV = sess.run(var_grad, feed_dict={states:ALL_full})
            LHS = dV[:,-1];
            dVdx = dV[:,:-1]; #### !!!
            k1 = F(dVdx,ALL_x_); 
    
            f_x = k1;
    
            RHS = np.sum(np.multiply(dVdx,f_x),axis=1,keepdims=True);
            RHS[RHS > 0.0] = 0;
            allE = LHS[:,None] + RHS;#np.array([LHS[ii] + RHS[ii] for ii in xrange(np.size(RHS))]);
            PDE_ERROR = np.mean(np.abs(allE));
            log_pde_loss[kk] = PDE_ERROR;
            kk+=1;

            plt.clf();
            plt.plot(ALL_t_, np.abs(allE), 'ro');
            plt.axis([-0.2, 0.0, 0.0, 10.0])
            plt.pause(0.01);
               
            print str(ind) + ") PDE_ERR=" + str(PDE_ERROR) + " | (" + str(np.max(np.abs(error1))) + " , " + str(np.mean(np.abs(error1))) + " V(3,pi,0,6) = " + ") | MSE = " + str(mse) + "|ITR=" + str(i)
	    #print str(ind) + ")  V((-3,pi/8,0,6),-0.4) = " + str(sess.run(Tt,{states:np.array([[-3.0,np.pi/(4.0*2.0),0.0,6.0,-0.4]])})) + "|  V((-3,pi/4,0,6),-0.4) = " + str(sess.run(Tt,{states:np.array([[-3.0,np.pi/4.0,0.0,6.0,-0.4]])})) + "|  V((-3,3*pi/8,0,6),-0.4) = " + str(sess.run(Tt,{states:np.array([[-3.0,3.0*np.pi/8.0,0.0,6.0,-0.4]])})) +  ") | MSE = " + str(mse) + "|ITR=" + str(i)

           # ~~~~ Perform gradient step update 
           # np.random.shuffle(ch) 
           # tmp = ch[:batch_size];                                                      #VAR         


        tmp = np.random.randint(nrolls, size=bts);   
        sess.run(train_step, feed_dict={states:ALL[tmp],y:V[tmp],nu:nunu});     

layers1 = [5+1,10,5,1];
t_hor = -1.0;
main(layers1,t_hor,0,2000,25,0.001,0.9999,0,1000,0.0,0);
