# -*-: coding: utf-8 -*-
"""
Parent: CopyCopyNew_4D.py
In this script you implemented a rechability algorithm which:
- Takes in a  NN which computes a policy directly.( X -> NN -> softmax(output) which selects the actions. 
- The policy parameters are stored and used for subsequent training.

Created on Thu May 26 13:37:12 2016

@author: vrubies 

"""

#TODO: Add metrics of how the policies change thru times (e.g. average KL divergence. If policies converge we will know.)

import numpy as np
import tensorflow as tf
import itertools
from FAuxFuncs3_0 import TransDef
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
from mpl_toolkits.mplot3d import Axes3D
from dqn_utils import PiecewiseSchedule,LinearSchedule,linear_interpolation
import time
import h5py
            

def main(layers,t_hor,ind,nrolls,bts,ler_r,mom,teps,renew,imp,q):
    # Quad Params
    max_list = [0.1,0.1,11.81];
    min_list = [-0.1,-0.1,7.81];
    
    max_list_ = [0.5,0.5,0.5]
    min_list_ = [-0.5,-0.5,-0.5]
    
    g = 9.81;


    print 'Starting worker-' + str(ind)

    f = 1;
    Nx = 100*f + 1;
    minn = [-5.0,-10.0,-5.0,-10.0,0.0,-10.0];
    maxx = [ 5.0, 10.0, 5.0, 10.0,2*np.pi, 10.0];
    
    X = np.linspace(minn[0],maxx[0],Nx);
    Y = np.linspace(minn[2],maxx[2],Nx);
    Z = np.linspace(minn[4],maxx[4],Nx);
    X_,Y_,Z_ = np.meshgrid(X, Y, Z);    
    X,Y = np.meshgrid(X, Y);
    XX = np.reshape(X,[-1,1]);
    YY = np.reshape(Y,[-1,1]);
    XX_ = np.reshape(X_,[-1,1]);
    YY_ = np.reshape(Y_,[-1,1]);
    ZZ_ = np.reshape(Z_,[-1,1]); grid_check = np.concatenate((XX_,np.ones(XX_.shape),YY_,np.ones(XX_.shape),ZZ_,np.zeros(XX_.shape)),axis=1);
    grid_eval = np.concatenate((XX,YY,0.0*np.ones(XX.shape),6.0*np.ones(XX.shape)),axis=1);
    grid_eval_ = np.concatenate((XX,YY,(2.0/3.0)*np.pi*np.ones(XX.shape),6.0*np.ones(XX.shape)),axis=1);
    grid_eval__ = np.concatenate((XX,YY,(4.0/3.0)*np.pi*np.ones(XX.shape),6.0*np.ones(XX.shape)),axis=1);
    grid_evall = np.concatenate((XX,YY,0.0*np.ones(XX.shape),12.0*np.ones(XX.shape)),axis=1);
    grid_evall_ = np.concatenate((XX,YY,(2.0/3.0)*np.pi*np.ones(XX.shape),12.0*np.ones(XX.shape)),axis=1);
    grid_evall__ = np.concatenate((XX,YY,(4.0/3.0)*np.pi*np.ones(XX.shape),12.0*np.ones(XX.shape)),axis=1);    


    # Calculate number of parameters of the policy
    nofparams = 0;
    for i in xrange(len(layers)-1):
        nofparams += layers[i]*layers[i+1] + layers[i+1];
    print 'Number of Params is: ' + str(nofparams)
    
    H_length = t_hor;
    center = np.array([[0.0,0.0,0.0,0.0,0.0,0.0]])
    depth = 2.0;
    incl = 1.0;

    ##################### DEFINITIONS #####################
    #layers = [2 + 1,10,1];                                                    #VAR
    #ssize = layers[0] - 1;
    dt = 0.1;                                                                 #VAR
    num_ac = 3;
    iters = int(np.abs(t_hor)/dt)*renew + 1; 
    ##################### INSTANTIATIONS #################
    states,y,Tt,L,l_r,lb,reg, cross_entropy = TransDef("Control",False,layers,depth,incl,center);
    states_,y_,Tt_,L_,l_r_,lb_,reg_, cross_entropy_ = TransDef("Disturbance",False,layers,depth,incl,center);
    ola1 = tf.argmax(Tt,dimension=1)
    ola2 = tf.argmax(y,dimension=1)
    ola3 = tf.equal(ola1,ola2)
    accuracy = tf.reduce_mean(tf.cast(ola3, tf.float32));
    ola1_ = tf.argmax(Tt_,dimension=1)
    ola2_ = tf.argmax(y_,dimension=1)
    ola3_ = tf.equal(ola1_,ola2_)
    accuracy_ = tf.reduce_mean(tf.cast(ola3_, tf.float32));    
    #a_layers = layers;
    #a_layers[-1] = 2; #We have two actions
    #states_,y_,Tt_,l_r_,lb_,reg_ = TransDef("Actor",False,a_layers,depth,incl,center,outp=True);
    
    C_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Control');
    D_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Disturbance');
    
    #var_grad = tf.gradients(Tt_,states_)[0]
    var_grad_ = tf.gradients(Tt,states)[0]
    grad_x = tf.slice(var_grad_,[0,0],[-1,layers[0]-1]);
    #theta = tf.trainable_variables();

    set_to_zero = []
    for var  in sorted(C_func_vars,        key=lambda v: v.name):
        set_to_zero.append(var.assign(tf.zeros(tf.shape(var))))
    set_to_zero = tf.group(*set_to_zero)
    
    set_to_not_zero = []
    for var  in sorted(C_func_vars,        key=lambda v: v.name):
        set_to_not_zero.append(var.assign(tf.random_uniform(tf.shape(var),minval=-0.1,maxval=0.1)));
    set_to_not_zero = tf.group(*set_to_not_zero)    

    # DEFINE LOSS

    lmbda = 0.0;#1.0**(-3.5);#0.01;
    beta = 0.00;
    #L = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y,Tt)),1,keep_dims=True))) + beta*tf.reduce_mean(tf.reduce_max(tf.abs(grad_x),reduction_indices=1,keep_dims=True));
    #L = tf.reduce_mean(tf.mul(tf.exp(imp*t_vec),tf.abs(tf.sub(y,Tt)))) + lmbda*reg;
    #L = tf.reduce_mean(tf.abs(tf.sub(y,Tt))) + lmbda*reg;    

    # DEFINE OPTIMIZER

    #nu = 5.01;
    #nunu = ler_r;#0.00005;
    nu = tf.placeholder(tf.float32, shape=[])                                         #VAR

    #lr_multiplier = ler_r
    lr_schedule = PiecewiseSchedule([
                                         (0, 0.1),
                                         (10000, 0.01 ),
                                         (20000, 0.001 ),
                                         (30000, 0.0001 ),
                                    ],
                                    outside_value=0.0001)

    #optimizer = tf.train.GradientDescentOptimizer(nu)
    #optimizer
    #train_step = tf.train.MomentumOptimizer(learning_rate=nu,momentum=mom).minimize(L)
    #optimizer 
    #train_step = tf.train.AdamOptimizer(learning_rate=nu).minimize(L);
    train_step = tf.train.RMSPropOptimizer(learning_rate=nu,momentum=mom).minimize(L);
    train_step_ = tf.train.RMSPropOptimizer(learning_rate=nu,momentum=mom).minimize(L_);
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=nu,momentum=mom);
    #gvs = optimizer.compute_gradients(L,theta);
    #capped_gvs = [(tf.clip_by_value(grad, -3., 3.), var) for grad, var in gvs];
    #train_step = optimizer.apply_gradients(gvs);
    #train_step = tf.train.AdagradOptimizer(learning_rate=nu,initial_accumulator_value=0.5).minimize(L);

    hot_input = tf.placeholder(tf.int64,shape=(None));   
    make_hot = tf.one_hot(hot_input, 2**num_ac, on_value=1, off_value=0)

    # INITIALIZE GRAPH
    sess = tf.Session();
    init = tf.initialize_all_variables();
    sess.run(init);

    def V_0(x):
        #return np.linalg.norm(x,ord=np.inf,axis=1,keepdims=True) - 1.0
        return np.linalg.norm(x,axis=1,keepdims=True) - 1.0

    def p_corr(ALL_x):
        ALL_x = np.mod(ALL_x,2.0*np.pi);
        return ALL_x;

    def F(ALL_x,opt_a,opt_b):#(grad,ALL_x):
       col1 = ALL_x[:,3,None] - opt_b[:,0,None]
       col2 = ALL_x[:,4,None] - opt_b[:,1,None]
       col3 = ALL_x[:,5,None] - opt_b[:,2,None]
       col4 = g*opt_a[:,0,None]
       col5 = -g*opt_a[:,1,None]
       col6 = opt_a[:,2,None] - g
       
       return np.concatenate((col1,col2,col3,col4,col5,col6),axis=1);
       
    ####################### RECURSIVE FUNC ####################

    def RK4(ALL_x,dtt,opt_a,opt_b):

        k1 = F(ALL_x,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k2)
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k1);
        #ALL_tmp[:,4] = p_corr(ALL_tmp[:,4]);

        k2 = F(ALL_tmp,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k3)
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k2);
        #ALL_tmp[:,4] = p_corr(ALL_tmp[:,4]);

        k3 = F(ALL_tmp,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k4)
        ALL_tmp = ALL_x + np.multiply(dtt,k3);
        #ALL_tmp[:,4] = p_corr(ALL_tmp[:,4]);

        k4 = F(ALL_tmp,opt_a,opt_b);  #### !!!

        Snx = ALL_x + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4)); #np.multiply(dtt,k1)
        #Snx[:,4] = p_corr(Snx[:,4]);
        return Snx;

    perms = list(itertools.product([-1,1], repeat=num_ac))
    true_ac_list = [];
    for i in range(len(perms)): #2**num_actions
        ac_tuple = perms[i];
        ac_list = [(tmp1==1)*tmp3 +  (tmp1==-1)*tmp2 for tmp1,tmp2,tmp3 in zip(ac_tuple,min_list,max_list)]; 
        true_ac_list.append(ac_list);
        
    dist_ac = 3;    
    perms_ = list(itertools.product([-1,1], repeat=dist_ac))
    true_ac_list_ = [];
    for i in range(len(perms_)): #2**num_actions
        ac_tuple_ = perms_[i];
        ac_list_ = [(tmp1==1)*tmp3 +  (tmp1==-1)*tmp2 for tmp1,tmp2,tmp3 in zip(ac_tuple_,min_list_,max_list_)]; #ASSUMING: aMax = -aMin
        true_ac_list_.append(ac_list_);       
    
    def Hot_to_Cold(hots,ac_list):
        a = hots.argmax(axis=1);
        a = np.asarray([ac_list[i] for i in a]);
        return a;
    
    def getPI(ALL_x,F_PI=[], F_PI_=[], subSamples=1): #Things to keep in MIND: You want the returned value to be the minimum accross a trajectory.

        current_params = sess.run(C_func_vars);
        current_params_ = sess.run(D_func_vars);

        #perms = list(itertools.product([-1,1], repeat=num_ac))
        next_states_ = [];
        for k in range((len(perms))):
            next_states = [];
            opt_a = np.asarray(true_ac_list[k])*np.ones([ALL_x.shape[0],1]);
            for i in range(len(perms_)):
                opt_b = np.asarray(true_ac_list_[i])*np.ones([ALL_x.shape[0],1]);
                Snx = ALL_x;
                for _ in range(subSamples): 
                    Snx = RK4(Snx,dt/float(subSamples),opt_a,opt_b);
                next_states.append(Snx);
            next_states_.append(np.concatenate(next_states,axis=0));
        next_states_ = np.concatenate(next_states_,axis=0);
        #values = V_0(next_states[:,[0,2]]);
        
        
        for params,params_ in zip(F_PI,F_PI_):
            for ind in range(len(params)): #Reload pi*(x,t+dt) parameters
                sess.run(C_func_vars[ind].assign(params[ind]));
            for ind in range(len(params_)): #Reload pi*(x,t+dt) parameters
                sess.run(D_func_vars[ind].assign(params_[ind]));            

            tmp = ConvCosSin(next_states_);
            hots = sess.run(Tt,{states:tmp});
            opt_a = Hot_to_Cold(hots,true_ac_list)   
            hots = sess.run(Tt_,{states_:tmp});
            opt_b = Hot_to_Cold(hots,true_ac_list_)            
            for _ in range(subSamples):
                next_states_ = RK4(next_states_,dt/float(subSamples),opt_a,opt_b);
                #values = np.min((values,V_0(next_states[:,[0,2]])),axis=0);
        
        values_ = V_0(next_states_[:,[0,1,2]]);
        pre_compare_vals_ = values_.reshape([-1,ALL_x.shape[0]]).T;         #Changed to values instead of values_
        final_v = [];
        final_v_ = [];
        per = len(perms);
        for k in range(len(perms_)):
            final_v.append(np.argmax(pre_compare_vals_[:,k*per:(k+1)*per,None],axis=1))
            final_v_.append(np.max(pre_compare_vals_[:,k*per:(k+1)*per,None],axis=1))
        finalF = np.concatenate(final_v_,axis=1);
        index_best_a_ = np.argmin(finalF,axis=1);
        finalF_ = np.concatenate(final_v,axis=1);
        index_best_b_ = np.array([finalF_[k,index_best_a_[k]] for k in range(len(index_best_a_))]);
        
        for ind in range(len(current_params)): #Reload pi*(x,t+dt) parameters
            sess.run(C_func_vars[ind].assign(current_params[ind]));
        for ind in range(len(current_params_)): #Reload pi*(x,t+dt) parameters
            sess.run(D_func_vars[ind].assign(current_params_[ind]));
            
        return sess.run(make_hot,{hot_input:index_best_a_}),sess.run(make_hot,{hot_input:index_best_b_})

#    def getTraj(ALL_x,F_PI=[],F_PI_=[],subSamples=1,StepsLeft=None,Noise = False):
#
#        current_params = sess.run(C_func_vars);
#        current_params_ = sess.run(D_func_vars);
#        
#        if(StepsLeft == None): StepsLeft = len(F_PI);        
#        
#        next_states_ = ALL_x;
#        traj = [next_states_];
#        actions = [];
#
#        for params,params_ in zip(F_PI,F_PI_):
#            for ind in range(len(params)): #Reload pi*(x,t+dt) parameters
#                sess.run(C_func_vars[ind].assign(params[ind]));
#            for ind in range(len(params_)): #Reload pi*(x,t+dt) parameters
#                sess.run(D_func_vars[ind].assign(params_[ind]));            
#
#            tmp = ConvCosSin(next_states_);
#            hots = sess.run(Tt,{states:tmp});
#            opt_a = Hot_to_Cold(hots,true_ac_list)   
#            hots_ = sess.run(Tt_,{states_:tmp});
#            opt_b = Hot_to_Cold(hots_,true_ac_list_)            
#            for _ in range(subSamples):
#                next_states_ = RK4(next_states_,dt/float(subSamples),opt_a,opt_b);
#                traj.append(next_states_); 
#                actions.append(hots.argmax(axis=1)[0]);
#
#        for ind in range(len(current_params)): #Reload pi*(x,t+dt) parameters
#            sess.run(C_func_vars[ind].assign(current_params[ind]));
#        for ind in range(len(current_params_)): #Reload pi*(x,t+dt) parameters
#            sess.run(D_func_vars[ind].assign(current_params_[ind]));
#                        
#        return traj,actions#,V_0(next_states[:,[0,2]]),actions; 

    def getTraj(ALL_x,F_PI=[],F_PI_=[],subSamples=1,StepsLeft=None,Noise = False):

        current_params = sess.run(C_func_vars);
        current_params_ = sess.run(D_func_vars);
        
        if(StepsLeft == None): StepsLeft = len(F_PI);        
        
        next_states_ = ALL_x;
        traj = [next_states_];
        actions = [];
              
        for params,params_ in zip(F_PI[len(F_PI)-StepsLeft:],F_PI_[len(F_PI_)-StepsLeft:]):
            for ind in range(len(params)): #Reload pi*(x,t+dt) parameters
                sess.run(C_func_vars[ind].assign(params[ind]));
            for ind in range(len(params_)): #Reload pi*(x,t+dt) parameters
                sess.run(D_func_vars[ind].assign(params_[ind]));  
            
            tmp = ConvCosSin(next_states_);
            hots = sess.run(Tt,{states:tmp});
            opt_a = Hot_to_Cold(hots,true_ac_list)   
            if Noise == False:
                hots_ = sess.run(Tt_,{states_:tmp});
                opt_b = Hot_to_Cold(hots_,true_ac_list_)
            else:
                hots_ = np.zeros((1,2**dist_ac));
                hots_[0][np.random.randint(2**dist_ac)] = 1
                opt_b = Hot_to_Cold(hots_,true_ac_list_)
                        
            for _ in range(subSamples):
                next_states_ = RK4(next_states_,dt/float(subSamples),opt_a,opt_b);
                traj.append(next_states_); 
                actions.append(hots.argmax(axis=1)[0]);   

        for ind in range(len(current_params)): #Reload pi*(x,t+dt) parameters
            sess.run(C_func_vars[ind].assign(current_params[ind]));
        for ind in range(len(current_params_)): #Reload pi*(x,t+dt) parameters
            sess.run(D_func_vars[ind].assign(current_params_[ind]));
                        
        return traj,actions,V_0(next_states_[:,[0,2]])

    def ConvCosSin(ALL_x):
        pos = ALL_x[:,[0,1,2]]/5.0;
        vel = ALL_x[:,[3,4,5]]/10.0;
        ret_val = np.concatenate((pos,vel),axis=1)
        return ret_val
    # *****************************************************************************
    #
    # ============================= MAIN LOOP ====================================
    #
    # *****************************************************************************
    t1 = time.time();
    t = 0.0;
    mse = np.inf;
    k=0; kk = 0; beta=3.0; batch_size = bts; tau = 1000.0; steps = teps;
    ALL_PI = [];
    ALL_PI_= [];
    nunu = lr_schedule.value(k);
    
    act_color = ['r','g','b','y'];
    if(imp == 1.0):
        ALL_PI,ALL_PI_ = pickle.load( open( "policies6D_P&T_h40_h40.pkl", "rb" ) );
        while True:
            state_get = input('State: ');
            sub_smpl = input('SUBSAMPLING: ');
            pause_len = input('Pause: ')
            s_left = input("How many steps left to go (max. "+str(len(ALL_PI))+")? -> ")
            noise = input("Noise? (0/1): ")
            traj,act,_ = getTraj(state_get,F_PI=ALL_PI,F_PI_=ALL_PI_,subSamples=sub_smpl,StepsLeft=s_left,Noise=noise);
            act.append(act[-1]);
            all_to = np.concatenate(traj);
            plt.scatter(all_to[:,[0]],all_to[:,[2]])
            plt.pause(pause_len);
            #plt.colorbar()
    elif(imp == 2.0):
        ALL_PI,ALL_PI_ = pickle.load( open( "policies6D_C&D_h40_h40.pkl", "rb" ) );
        fig = plt.figure(1)
        while True:
            sub_smpl = input('SUBSAMPLING: ');
            pause_len = input('Pause: ')
            s_left = input("How many steps left to go (max. "+str(len(ALL_PI))+")? -> ")  
            grid_check = np.random.uniform(-10.0,10.0,(nrolls,layers[0]-1));
            grid_check[:,0] = 1.5
            grid_check[:,2] = 1.5
            grid_check[:,4] = 1.5#grid_check[:,4]*np.pi/5.0 + np.pi;
            #grid_check[:,5] = 0
            #fig = plt.figure(1)
            #plt.clf();
            _,_,nn_vals = getTraj(grid_check,F_PI=ALL_PI,F_PI_=ALL_PI_,subSamples=sub_smpl,StepsLeft=s_left);
            fi = (nn_vals < 0.0)
            mini_reach_ = grid_check[fi[:,0]]
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(mini_reach_[:,1], mini_reach_[:,3], mini_reach_[:,5]); 
            #plt.xlim(-5, 5)
            #plt.ylim(-5, 5)
            plt.pause(pause_len);         
            
    
    for i in xrange(iters):
        
        if(np.mod(i,renew) == 0 and i is not 0):       
            
            ALL_PI.insert(0,sess.run(C_func_vars));
            ALL_PI_.insert(0,sess.run(D_func_vars)); 
            
#            fig = plt.figure(1)
#            plt.clf();
#            _,nn_vals,_ = getTraj(grid_check,ALL_PI,20)
#            fi = (np.abs(nn_vals) < 0.05)
#            mini_reach_ = grid_check[fi[:,0]]
#            ax = fig.add_subplot(111, projection='3d')
#            ax.scatter(mini_reach_[:,0], mini_reach_[:,2], mini_reach_[:,4]);            
#            plt.pause(0.25);            

            plt.figure(2) #TODO: Figure out why facing up vs facing down has same action... -> Solved: colors in a scatter plot only depend on the labels
            plt.clf();
            ALL_xx = np.array([[-1.0,0.0,1.0,0.0,0.0,0.0],
                               [1.0,0.0,1.0,0.0,0.0,0.0],
                               [1.0,0.0,-1.0,0.0,0.0,0.0],
                               [-1.0,0.0,-1.0,0.0,0.0,0.0]]);
            for tmmp in range(ALL_xx.shape[0]):                   
                traj,act,_ = getTraj(ALL_xx[[tmmp],:],F_PI=ALL_PI,F_PI_=ALL_PI_,subSamples=10);
                #act.append(act[-1]);
                all_to = np.concatenate(traj);
                plt.scatter(all_to[:,[0]],all_to[:,[2]])#c=[act_color[ii] for ii in act]);                   
            plt.pause(0.25)                   
 
#            plt.figure(3)
#            d = 0.1
#            plt.clf();
#            plt.title(str([str(i)+" : "+str(perms[i]) for i in range(len(perms))]))
#            ALL_xp = np.random.uniform(-5.0,5.0,(nrolls/100,layers[0]-1));
#            plt.subplot(2,3,1) #SUBPLOT
#            ALL_xp[:,1] = 0.0
#            ALL_xp[:,3] = 0.0
#            ALL_xp[:,4] = 0.0 + d
#            ALL_xp[:,5] = 0.0; 
#            letsee_ = sess.run(Tt,{states:ConvCosSin(ALL_xp)});
#            letsee_ = letsee_.argmax(axis=1);
#            plt.scatter(ALL_xp[:,0],ALL_xp[:,2],c=letsee_)
#            plt.colorbar()
#            plt.subplot(2,3,2) #SUBPLOT
#            ALL_xp[:,1] = 0.0
#            ALL_xp[:,3] = 0.0
#            ALL_xp[:,4] = np.pi/2.0 + d
#            ALL_xp[:,5] = 0.0; 
#            letsee_ = sess.run(Tt,{states:ConvCosSin(ALL_xp)});
#            letsee_ = letsee_.argmax(axis=1);
#            plt.scatter(ALL_xp[:,0],ALL_xp[:,2],c=letsee_)
#            plt.colorbar()
#            plt.subplot(2,3,3) #SUBPLOT
#            ALL_xp[:,1] = 0.0
#            ALL_xp[:,3] = 0.0
#            ALL_xp[:,4] = np.pi + d
#            ALL_xp[:,5] = 0.0; 
#            letsee_ = sess.run(Tt,{states:ConvCosSin(ALL_xp)});
#            letsee_ = letsee_.argmax(axis=1);
#            plt.scatter(ALL_xp[:,0],ALL_xp[:,2],c=letsee_)
#            plt.colorbar()
#            plt.subplot(2,3,4) #SUBPLOT
#            ALL_xp[:,1] = 0.0
#            ALL_xp[:,3] = 0.0
#            ALL_xp[:,4] = 0.0 - d
#            ALL_xp[:,5] = 0.0; 
#            letsee_ = sess.run(Tt,{states:ConvCosSin(ALL_xp)});
#            letsee_ = letsee_.argmax(axis=1);
#            plt.scatter(ALL_xp[:,0],ALL_xp[:,2],c=letsee_)
#            plt.colorbar()
#            plt.subplot(2,3,5) #SUBPLOT
#            ALL_xp[:,1] = 0.0
#            ALL_xp[:,3] = 0.0
#            ALL_xp[:,4] = np.pi/2 - d
#            ALL_xp[:,5] = 0.0; 
#            letsee_ = sess.run(Tt,{states:ConvCosSin(ALL_xp)});
#            letsee_ = letsee_.argmax(axis=1);
#            plt.scatter(ALL_xp[:,0],ALL_xp[:,2],c=letsee_)
#            plt.colorbar()
#            plt.subplot(2,3,6) #SUBPLOT
#            ALL_xp[:,1] = 0.0
#            ALL_xp[:,3] = 0.0
#            ALL_xp[:,4] = np.pi - d
#            ALL_xp[:,5] = 0.0; 
#            letsee_ = sess.run(Tt,{states:ConvCosSin(ALL_xp)});
#            letsee_ = letsee_.argmax(axis=1);
#            plt.scatter(ALL_xp[:,0],ALL_xp[:,2],c=letsee_)
#            plt.colorbar()         
#            plt.pause(0.1);            
                        
            
            k = 0;
            ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]));
            ALL_x[:,[3,4,5]] = ALL_x[:,[3,4,5]]*2.0
            PI_c,PI_d = getPI(ALL_x,ALL_PI,ALL_PI_,subSamples=1);
            pre_ALL_x = ConvCosSin(ALL_x);
            
            ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls/100,layers[0]));
            ALL_x_[:,[3,4,5]] = ALL_x_[:,[3,4,5]]*2.0
            PI_c_,PI_d_ = getPI(ALL_x_,ALL_PI,ALL_PI_,subSamples=1);
            pre_ALL_x_ = ConvCosSin(ALL_x_);

#            tmp = np.random.randint(len(reach100s[:,:-1]), size=12000);
#            _,ZR = getPI(reach100s[tmp,:-1],ALL_PI)
#            #ZR = sess.run(Tt,{states:reach100s[:,:-1]});
#            error1 = ZR - reach100s[tmp,-1,None];
#            
#           
#            plt.figure(2)
#            _,Z000 = getPI(grid_eval,ALL_PI);
#            _,Z001 = getPI(grid_eval_,ALL_PI);
#            _,Z002 = getPI(grid_eval__,ALL_PI);            
#            Z000 = np.reshape(Z000,X.shape);
#            Z001 = np.reshape(Z001,X.shape);
#            Z002 = np.reshape(Z002,X.shape);
#            #filter_in = (Z000 <= 0.05) #& (Z000 >= 0.05);
#            filter_out = (Z000 > 0.00) #| (Z000 < -0.05);       
#            filter_out_ = (Z001 > 0.00) #| (Z000 < -0.05);       
#            filter_out__ = (Z002 > 0.00) #| (Z000 < -0.05);       
#            #Z000[filter_in] = 1.0;
#            Z000[filter_out] = 0.0;
#            Z001[filter_out_] = 0.0;
#            Z002[filter_out__] = 0.0;
#
#            _,Z000l = getPI(grid_evall,ALL_PI);
#            _,Z001l = getPI(grid_evall_,ALL_PI);
#            _,Z002l = getPI(grid_evall__,ALL_PI);             
#            Z000l = np.reshape(Z000l,X.shape);
#            Z001l = np.reshape(Z001l,X.shape);
#            Z002l = np.reshape(Z002l,X.shape);
#            #filter_in = (Z000 <= 0.05) #& (Z000 >= 0.05);
#            filter_outl = (Z000l > 0.00) #| (Z000 < -0.05);       
#            filter_out_l = (Z001l > 0.00) #| (Z000 < -0.05);       
#            filter_out__l = (Z002l > 0.00) #| (Z000 < -0.05);       
#            #Z000[filter_in] = 1.0;
#            Z000l[filter_outl] = 0.0;
#            Z001l[filter_out_l] = 0.0;
#            Z002l[filter_out__l] = 0.0;
#
#            plt.clf();
#            #plt.plot(ALL_t_, np.abs(allE), 'ro');
#            #plt.axis([-1.0, 0.0, 0.0, 10.0])
#            plt.subplot(2,3,1)
#            plt.imshow(Z000,cmap='gray');
#            plt.plot([30*f, 30*f], [30*f, 70*f], 'r-', lw=1)
#            plt.plot([30*f, 70*f], [70*f, 70*f], 'r-', lw=1)
#            plt.plot([70*f, 70*f], [70*f, 30*f], 'r-', lw=1)
#            plt.plot([70*f, 30*f], [30*f, 30*f], 'r-', lw=1)
#            plt.subplot(2,3,2)
#            plt.imshow(Z001,cmap='gray');
#            plt.plot([30*f, 30*f], [30*f, 70*f], 'r-', lw=1)
#            plt.plot([30*f, 70*f], [70*f, 70*f], 'r-', lw=1)
#            plt.plot([70*f, 70*f], [70*f, 30*f], 'r-', lw=1)
#            plt.plot([70*f, 30*f], [30*f, 30*f], 'r-', lw=1)
#            plt.subplot(2,3,3)
#            plt.imshow(Z002,cmap='gray');
#            plt.plot([30*f, 30*f], [30*f, 70*f], 'r-', lw=1)
#            plt.plot([30*f, 70*f], [70*f, 70*f], 'r-', lw=1)
#            plt.plot([70*f, 70*f], [70*f, 30*f], 'r-', lw=1)
#            plt.plot([70*f, 30*f], [30*f, 30*f], 'r-', lw=1)
#            plt.subplot(2,3,4)
#            plt.imshow(Z000l,cmap='gray');
#            plt.plot([30*f, 30*f], [30*f, 70*f], 'r-', lw=1)
#            plt.plot([30*f, 70*f], [70*f, 70*f], 'r-', lw=1)
#            plt.plot([70*f, 70*f], [70*f, 30*f], 'r-', lw=1)
#            plt.plot([70*f, 30*f], [30*f, 30*f], 'r-', lw=1)
#            plt.subplot(2,3,5)
#            plt.imshow(Z001l,cmap='gray');
#            plt.plot([30*f, 30*f], [30*f, 70*f], 'r-', lw=1)
#            plt.plot([30*f, 70*f], [70*f, 70*f], 'r-', lw=1)
#            plt.plot([70*f, 70*f], [70*f, 30*f], 'r-', lw=1)
#            plt.plot([70*f, 30*f], [30*f, 30*f], 'r-', lw=1)
#            plt.subplot(2,3,6)
#            plt.imshow(Z002l,cmap='gray'); 
#            plt.plot([30*f, 30*f], [30*f, 70*f], 'r-', lw=1)
#            plt.plot([30*f, 70*f], [70*f, 70*f], 'r-', lw=1)
#            plt.plot([70*f, 70*f], [70*f, 30*f], 'r-', lw=1)
#            plt.plot([70*f, 30*f], [30*f, 30*f], 'r-', lw=1)
#            plt.pause(0.01);

            t = t - dt; 
            print('Again.')
#            sess.run(set_to_not_zero);
#            print str(t) + " || " + str(np.max(np.abs(error1))) + " , " + str(np.mean(np.abs(error1))) + "|ITR=" + str(i)                                                #VAR         
            
#            plt.figure(4)
#            plt.clf();
#            plt.title(str([str(i)+" : "+str(perms[i]) for i in range(len(perms))]))
#            b_sele = (ALL_x[:,-1] < 6.1); 
#            ALL_xp = ALL_x[b_sele]; 
#            letsee_ = PI[b_sele];
#            b_sele = (np.abs(ALL_xp[:,2]-np.pi/2.0 + 0.1) < 0.1);
#            ALL_xp = ALL_xp[b_sele];
#            letsee_ = letsee_[b_sele];  
#            _,_ = getPI(ALL_xp);
#            #plt.subplot(2,3,1) #SUBPLOT
#            letsee_ = letsee_.argmax(axis=1);
#            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
#            plt.colorbar()
#            plt.pause(0.01)
#            woot = np.array([[-0.15023694, -4.03420314,  1.56425333,  6.02741677],
#       [ 0.10373495, -4.34956515,  1.50186123,  6.08060291],
#       [ 0.13439703, -5.47363893,  1.60820922,  6.0519111 ],
#       [ 0.07739933, -4.93777028,  1.57579839,  6.00117299]])          
#            _,_ = getPI(woot,ALL_PI);
            
        #elif(i is 0):
        elif(np.mod(i,renew) == 0 and i is 0):

#            sess.run(set_to_zero);
            t = time.time()
            ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]));
            ALL_x[:,[3,4,5]] = ALL_x[:,[3,4,5]]*2.0                  
            PI_c,PI_d = getPI(ALL_x,F_PI=[],F_PI_=[],subSamples=1);
            pre_ALL_x = ConvCosSin(ALL_x);
            elapsed = time.time() - t
            print("Compute Data Time = "+str(elapsed))
            
            ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls/100,layers[0]));
            ALL_x_[:,[3,4,5]] = ALL_x_[:,[3,4,5]]*2.0
            PI_c_,PI_d_ = getPI(ALL_x_,F_PI=[],F_PI_=[],subSamples=1);
            pre_ALL_x_ = ConvCosSin(ALL_x_);           
#            sess.run(set_to_not_zero);

            

        # |||||||||||| ----  PRINT ----- |||||||||||| 

        if(np.mod(i,200) == 0):

            #xel = sess.run(L,{states:ALL_x,y:PI});
            #test_e = sess.run(L,{states:ALL_x_,y:PI_});
            train_acc = sess.run(accuracy,{states:pre_ALL_x,y:PI_c});
            test_acc = sess.run(accuracy,{states:pre_ALL_x_,y:PI_c_});
            train_acc_ = sess.run(accuracy_,{states_:pre_ALL_x,y_:PI_d});
            test_acc_ = sess.run(accuracy_,{states_:pre_ALL_x_,y_:PI_d_});             
            #o = np.random.randint(len(ALL_x));
            print str(i) + ") control | TR_ACC = " + str(train_acc) + " | TE_ACC = " + str(test_acc) + " | Learning Rate = " + str(nunu)
            print str(i) + ") disturb | TR_ACC = " + str(train_acc_) + " | TE_ACC = " + str(test_acc_) + " | Learning Rate = " + str(nunu)
            #print str(i) + ") | XEL = " + str(xel) + " | Test_E = " + str(test_e) + " | Lerning Rate = " + str(nunu)
            #print str(PI[[o],:]) + " || " + str(sess.run(l_r[-1],{states:ALL_x[[o],:]})) #+ " || " + str(sess.run(gvs[-1],{states:ALL_x,y:PI}))
            
        nunu = 0.001#/(np.sqrt(np.mod(i,renew))+1.0)#lr_schedule.value(i);
        #nunu = ler_r/(np.mod(i,renew)+1.0);
        tmp = np.random.randint(len(ALL_x), size=bts);
        sess.run(train_step, feed_dict={states:pre_ALL_x[tmp],y:PI_c[tmp],nu:nunu});
        sess.run(train_step_, feed_dict={states_:pre_ALL_x[tmp],y_:PI_d[tmp],nu:nunu});
        #tmp = np.random.randint(len(reach100s), size=bts);
        #sess.run(train_step, feed_dict={states:reach100s[tmp,:-1],y:reach100s[tmp,-1,None],nu:nunu});

    pickle.dump([ALL_PI,ALL_PI_],open( "policies6D_P&T_h40_h40.pkl", "wb" ));
#    while True:
#        state_get = input('State: ');
#        if(state_get == 0):
#            break;
#        _,VAL = getPI(state_get,ALL_PI);
#        print(str(VAL));

num_ac = 3;
layers1 = [6,40,40,2**num_ac];
t_hor = -2.0;

main(layers1,t_hor,0,200000,50000,0.001,0.95,99,1000,1.0,0); #MODIFIED!!!!!!!!!! 2000000 -> 2000000
