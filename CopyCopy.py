# -*-: coding: utf-8 -*-
"""
Created on Thu May 26 13:37:12 2016

@author: cusgadmin
"""

import numpy as np
import tensorflow as tf
from FAuxFuncs2_0 import TransDef
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
from mpl_toolkits.mplot3d import Axes3D
from dqn_utils import PiecewiseSchedule,LinearSchedule,linear_interpolation
import time
import h5py
            

def main(layers,t_hor,ind,nrolls,bts,ler_r,mom,teps,renew,imp,q):
# Quad Params
    wMax = 3.0; 
    wMin = -1.0*wMax;
    aMax = 2*np.pi/10.0;
    aMin = -1.0*aMax;  


    print 'Starting worker-' + str(ind)

    Nx = 101;
    minn = [-5.0,-5.0,0.0,6.0];
    maxx = [ 5.0, 5.0,2*np.pi,12.0];
    
    X = np.linspace(minn[0],maxx[0],Nx);
    Y = np.linspace(minn[1],maxx[1],Nx);
    X,Y = np.meshgrid(X, Y);
    XX = np.reshape(X,[-1,1]);
    YY = np.reshape(Y,[-1,1]);
    grid_eval = np.concatenate((XX,YY,0.0*np.ones(XX.shape),6.0*np.ones(XX.shape)),axis=1);
    grid_eval_ = np.concatenate((XX,YY,(2.0/3.0)*np.pi*np.ones(XX.shape),6.0*np.ones(XX.shape)),axis=1);
    grid_eval__ = np.concatenate((XX,YY,(4.0/3.0)*np.pi*np.ones(XX.shape),6.0*np.ones(XX.shape)),axis=1);
    grid_evall = np.concatenate((XX,YY,0.0*np.ones(XX.shape),12.0*np.ones(XX.shape)),axis=1);
    grid_evall_ = np.concatenate((XX,YY,(2.0/3.0)*np.pi*np.ones(XX.shape),12.0*np.ones(XX.shape)),axis=1);
    grid_evall__ = np.concatenate((XX,YY,(4.0/3.0)*np.pi*np.ones(XX.shape),12.0*np.ones(XX.shape)),axis=1);    

    reach100s = sio.loadmat('flat_1s.mat'); reach100s = reach100s["M"]; 
    reach100s[:,[1,2]] = reach100s[:,[2,1]];
    reach100s[:,2] = np.mod(reach100s[:,2],2.0*np.pi);
    #mean_data = np.mean(reach100s[:,:-1],axis=0);
    #std_data = np.std(reach100s[:,:-1],axis=0);

    nofparams = 0;
    for i in xrange(len(layers)-1):
        nofparams += layers[i]*layers[i+1] + layers[i+1];
    print 'Number of Params is: ' + str(nofparams)
    
    H_length = t_hor;#-1.0; #Has to be negative                                 #VAR
    iters = 1000000;                                                            #VAR
    #center = np.array([[0.0,0.0]])
    center = np.array([[0.0,0.0,0.0,0.0,0.0,0.0]])
    depth = 2.0;
    incl = 1.0;

    ##################### DEFINITIONS #####################
    #layers = [2 + 1,10,1];                                                    #VAR
    ssize = layers[0] - 1;
    dt = 0.1;                                                                 #VAR
    ##################### INSTANTIATIONS #################
    states,y,Tt,l_r,lb,reg = TransDef("Value_Function",False,layers,depth,incl,center);
    states_,y_,Tt_t,l_r_,lb_,reg_ = TransDef("Targ_Val_Function",False,layers,depth,incl,center);
    
    V_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Value_Function');
    target_V_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Targ_Val_Function');
    
    var_grad = tf.gradients(Tt_t,states_)[0]
    var_grad_ = tf.gradients(Tt,states)[0]
    grad_x = tf.slice(var_grad_,[0,0],[-1,layers[0]-1]);
    theta = tf.trainable_variables();

    update_target_fn = []
    for var, var_target in zip(sorted(V_func_vars,        key=lambda v: v.name),
                               sorted(target_V_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    set_to_zero = []
    for var, var_target in zip(sorted(V_func_vars,        key=lambda v: v.name),
                               sorted(target_V_func_vars, key=lambda v: v.name)):
        set_to_zero.append(var_target.assign(tf.zeros(tf.shape(var))))
    set_to_zero = tf.group(*set_to_zero)

    # DEFINE LOSS

    lmbda = 0.0;#1.0**(-3.5);#0.01;
    beta = 0.00;
    L = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y,Tt)),1,keep_dims=True))) + beta*tf.reduce_mean(tf.reduce_max(tf.abs(grad_x),reduction_indices=1,keep_dims=True));
    #L = tf.reduce_mean(tf.mul(tf.exp(imp*t_vec),tf.abs(tf.sub(y,Tt)))) + lmbda*reg;
    #L = tf.reduce_mean(tf.abs(tf.sub(y,Tt))) + lmbda*reg;    

    # DEFINE OPTIMIZER

    #nu = 5.01;
    #nunu = ler_r;#0.00005;
    nu = tf.placeholder(tf.float32, shape=[])                                         #VAR

    #lr_multiplier = ler_r
    lr_schedule = PiecewiseSchedule([
                                         (0, 0.01),
                                         (renew*2 / 4, 0.007 ),
                                         (renew*3 / 4, 0.005 ),
                                         (renew*4 / 4, 0.002 ),
                                    ],
                                    outside_value=0.001)

    #train_step = tf.train.GradientDescentOptimizer(nu).minimize(L)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=nu,momentum=mom);#.minimize(L)
    #optimizer = tf.train.AdamOptimizer(learning_rate=nu);
    optimizer = tf.train.RMSPropOptimizer(learning_rate=nu,momentum=mom);
    gvs = optimizer.compute_gradients(L,V_func_vars);
    capped_gvs = [(tf.clip_by_value(grad, -3., 3.), var) for grad, var in gvs];
    train_step = optimizer.apply_gradients(capped_gvs);
    #train_step = tf.train.AdagradOptimizer(learning_rate=nu,initial_accumulator_value=0.5).minimize(L);

    # INITIALIZE GRAPH
    sess = tf.Session();
    init = tf.initialize_all_variables();
    sess.run(init);

    def p_corr(ALL_x):
        ALL_x = np.mod(ALL_x,2.0*np.pi);
        return ALL_x;

    def opt_ac(grad):
       opt_dir_1_ = np.sign(grad[:,2,None])*wMin #np.floor((np.sign(grad[:,1,None])+1.0)/2.0)*wMin + np.ceil((np.sign(grad[:,1,None])-1.0)/2.0)*wMin;
       opt_dir_2_ = np.sign(grad[:,3,None])*aMin #np.floor((np.sign(grad[:,3,None])+1.0)/2.0)*aMin + np.ceil((np.sign(grad[:,3,None])-1.0)/2.0)*aMax;
       opt_a = np.concatenate((opt_dir_1_,opt_dir_2_),axis=1);
       return opt_a,None

    def F(ALL_x,opt_a,opt_b):
       sin_phi = np.sin(ALL_x[:,2,None]);
       cos_phi = np.cos(ALL_x[:,2,None]);

       col1 = np.multiply(ALL_x[:,3,None],cos_phi);
       col2 = np.multiply(ALL_x[:,3,None],sin_phi);
       col3 = opt_a[:,0,None];
       col4 = opt_a[:,1,None];

       return np.concatenate((col1,col2,col3,col4),axis=1);


    ####################### RECURSIVE FUNC ####################

    def RK4(ALL_x,dtt,dV):
        opt_a,opt_b = opt_ac(dV);  #### !!!

        k1 = F(ALL_x,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k2)
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k1);
        ALL_tmp[:,2] = p_corr(ALL_tmp[:,2]);

        k2 = F(ALL_tmp,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k3)
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k2);
        ALL_tmp[:,2] = p_corr(ALL_tmp[:,2]);

        k3 = F(ALL_tmp,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k4)
        ALL_tmp = ALL_x + np.multiply(dtt,k3);
        ALL_tmp[:,2] = p_corr(ALL_tmp[:,2]);

        k4 = F(ALL_tmp,opt_a,opt_b);  #### !!!

        Snx = ALL_x + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4));
        Snx[:,2] = p_corr(Snx[:,2]);
        return Snx;


    def V_ret(ALL_x):
        
        dV = sess.run(var_grad, feed_dict={states_:ALL_x})
        Snx = RK4(ALL_x,dt,dV);
        
        uno = sess.run(Tt_t,{states_:ALL_x});
        due = sess.run(Tt_t,{states_:Snx});
        filt = [((Snx[:,k,None] > maxx[k]) | (Snx[:,k,None] < minn[k])) for k in range(len(minn))];
        filt = np.any(filt,axis=0);
        due[filt] = np.inf;        
        
        V = np.min(np.concatenate((uno,due),axis=1),axis=1,keepdims=True);
        
        return V; 

    # *****************************************************************************
    #
    # ============================= MAIN LOOP ====================================
    #                     ( )    
    # *****************************************************************************
    t1 = time.time();
    t = 0.0;
    mse = np.inf;
    k=0; kk = 0; beta=3.0; batch_size = bts; tau = 1000.0; steps = teps; 
    nunu = lr_schedule.value(k);
    for i in xrange(iters):
        
        #if(mse < 0.005 and k > renew):
        if(np.mod(i,renew) == 0 and i is not 0): 
            
            k = 0;
            sess.run(update_target_fn);
            ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]));
            ALL_x[:,2] = ALL_x[:,2]*np.pi/5.0 + np.pi;
            ALL_x[:,3] = ALL_x[:,3]*3.0/5.0 + 9.0;
            V = V_ret(ALL_x);
            
            ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls/1000,layers[0]));
            ALL_x_[:,2] = ALL_x_[:,2]*np.pi/5.0 + np.pi;
            ALL_x_[:,3] = ALL_x_[:,3]*3.0/5.0 + 9.0;
            V_ = V_ret(ALL_x_);

            ZR = sess.run(Tt,{states:reach100s[:,:-1]});
            error1 = ZR - reach100s[:,-1,None];
            
            #error1 = 0.0;#targ_nn - sess.run(Tt,{states:into_nn});
                    
    
#            log_avg_error[kk] = np.max(np.abs(error1));
#            log_error[kk] = np.mean(np.abs(error1));

            
            Z000 = np.reshape(sess.run(Tt,{states:grid_eval}),X.shape);
            Z001 = np.reshape(sess.run(Tt,{states:grid_eval_}),X.shape);
            Z002 = np.reshape(sess.run(Tt,{states:grid_eval__}),X.shape);
            #filter_in = (Z000 <= 0.05) #& (Z000 >= 0.05);
            filter_out = (Z000 > 0.00) #| (Z000 < -0.05);       
            filter_out_ = (Z001 > 0.00) #| (Z000 < -0.05);       
            filter_out__ = (Z002 > 0.00) #| (Z000 < -0.05);       
            #Z000[filter_in] = 1.0;
            Z000[filter_out] = 0.0;
            Z001[filter_out_] = 0.0;
            Z002[filter_out__] = 0.0;
            
            Z000l = np.reshape(sess.run(Tt,{states:grid_evall}),X.shape);
            Z001l = np.reshape(sess.run(Tt,{states:grid_evall_}),X.shape);
            Z002l = np.reshape(sess.run(Tt,{states:grid_evall__}),X.shape);
            #filter_in = (Z000 <= 0.05) #& (Z000 >= 0.05);
            filter_outl = (Z000l > 0.00) #| (Z000 < -0.05);       
            filter_out_l = (Z001l > 0.00) #| (Z000 < -0.05);       
            filter_out__l = (Z002l > 0.00) #| (Z000 < -0.05);       
            #Z000[filter_in] = 1.0;
            Z000l[filter_outl] = 0.0;
            Z001l[filter_out_l] = 0.0;
            Z002l[filter_out__l] = 0.0;

            plt.clf();
            #plt.plot(ALL_t_, np.abs(allE), 'ro');
            #plt.axis([-1.0, 0.0, 0.0, 10.0])
            plt.subplot(2,3,1)
            plt.imshow(Z000,cmap='gray');
            plt.subplot(2,3,2)
            plt.imshow(Z001,cmap='gray');
            plt.subplot(2,3,3)
            plt.imshow(Z002,cmap='gray');
            plt.subplot(2,3,4)
            plt.imshow(Z000l,cmap='gray');
            plt.subplot(2,3,5)
            plt.imshow(Z001l,cmap='gray');
            plt.subplot(2,3,6)
            plt.imshow(Z002l,cmap='gray');            
            plt.pause(0.01);

               
            print str(t) + " || " + str(np.max(np.abs(error1))) + " , " + str(np.mean(np.abs(error1))) + " REG = " + str(sess.run(reg)) + ") | MSE = " + str(mse) + "|ITR=" + str(i)                                                #VAR         
            t = t - dt;            
            
        #elif(i is 0):
        elif(np.mod(i,renew) == 0 and i is 0):
            
            k = 0;
            sess.run(set_to_zero);
            ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]));
            ALL_x[:,2] = ALL_x[:,2]*np.pi/5.0 + np.pi;
            ALL_x[:,3] = ALL_x[:,3]*3.0/5.0 + 9.0;
            V = V_ret(ALL_x);     
            
            ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls/1000,layers[0]));
            ALL_x_[:,2] = ALL_x_[:,2]*np.pi/5.0 + np.pi;
            ALL_x_[:,3] = ALL_x_[:,3]*3.0/5.0 + 9.0;
            V_ = V_ret(ALL_x_);
            
                
            #ch = np.arange(len(V));

        # |||||||||||| ----  PRINT ----- ||||||||||||    

        if(np.mod(i,200) == 0):

            mse = sess.run(L,{states:ALL_x,y:V});
            test_e = sess.run(L,{states:ALL_x_,y:V_});
            print str(i) + ") | MSE = " + str(mse) + " | Test_E = " + str(test_e) + " | Lerning Rate = " + str(nunu)
            
            
        nunu = 0.01;#lr_schedule.value(k);
        k = k + 1;
        #nunu = ler_r/(np.mod(i,renew)+1.0);
        tmp = np.random.randint(len(ALL_x), size=bts);
        sess.run(train_step, feed_dict={states:ALL_x[tmp],y:V[tmp],nu:nunu});     

layers1 = [4,10,10,1];
t_hor = -1.0;

main(layers1,t_hor,0,1000000,500,0.001,0.99,99,2000,0.0,0);
