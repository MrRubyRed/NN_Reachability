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
            

def main(layers,t_hor,ind,nrolls,bts,ler_r,mom,teps,renew,imp,q):
    # Quad Params
    #Change to limit control in pitch or roll
    max_list = [0.1,0.1,11.81,1.0]; #w=1
    min_list = [-0.1,-0.1,7.81,-1.0]; 
    
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
    num_ac = 4;
    dist_ac = 3;
    iters = int(np.abs(t_hor)/dt)*renew + 1; 
    ##################### INSTANTIATIONS #################
    states,y,Tt,L,l_r,lb,reg, cross_entropy = TransDef("Control",False,layers,depth,incl,center);
    layers_ = layers[:]
    layers_[-1] = 2**dist_ac
    states_,y_,Tt_,L_,l_r_,lb_,reg_, cross_entropy_ = TransDef("Disturbance",False,layers_,depth,incl,center);
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
    make_hot_ = tf.one_hot(hot_input, 2**dist_ac, on_value=1, off_value=0)

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

#    def F(ALL_x,opt_a,opt_b):#(grad,ALL_x):
#       col1 = ALL_x[:,3,None] - opt_b[:,0,None]
#       col2 = ALL_x[:,4,None] - opt_b[:,1,None]
#       col3 = ALL_x[:,5,None] - opt_b[:,2,None]
#       col4 = g*opt_a[:,0,None]
#       col5 = -g*opt_a[:,1,None]
#       col6 = opt_a[:,2,None] - g
#       
#       return np.concatenate((col1,col2,col3,col4,col5,col6),axis=1);
   
    def F(ALL_x,opt_a,opt_b):#(grad,ALL_x):
       col1 = ALL_x[:,3,None] - opt_b[:,0,None]
       col2 = ALL_x[:,4,None] - opt_b[:,1,None]
       col3 = ALL_x[:,5,None] - opt_b[:,2,None]
       col4 = np.multiply(opt_a[:,2,None],np.multiply(np.cos(ALL_x[:,-1,None]),opt_a[:,0,None]) + np.multiply(np.sin(ALL_x[:,-1,None]),opt_a[:,1,None]))
       col5 = np.multiply(opt_a[:,2,None],-np.multiply(np.cos(ALL_x[:,-1,None]),opt_a[:,1,None]) + np.multiply(np.sin(ALL_x[:,-1,None]),opt_a[:,0,None]))
       col6 = np.multiply(opt_a[:,2,None],np.multiply(np.cos(opt_a[:,0,None]),np.cos(opt_a[:,1,None]))) - g
       col7 = opt_a[:,3,None]
       
       return np.concatenate((col1,col2,col3,col4,col5,col6,col7),axis=1);    
   
    ####################### RECURSIVE FUNC ####################

    def RK4(ALL_x,dtt,opt_a,opt_b): #Try Euler

        k1 = F(ALL_x,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k2)
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k1);
        ALL_tmp[:,-1] = p_corr(ALL_tmp[:,-1]);

        k2 = F(ALL_tmp,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k3)
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k2);
        ALL_tmp[:,-1] = p_corr(ALL_tmp[:,-1]);

        k3 = F(ALL_tmp,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k4)
        ALL_tmp = ALL_x + np.multiply(dtt,k3);
        ALL_tmp[:,-1] = p_corr(ALL_tmp[:,-1]);

        k4 = F(ALL_tmp,opt_a,opt_b);  #### !!!

        Snx = ALL_x + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4)); #np.multiply(dtt,k1)
        ALL_tmp[:,-1] = p_corr(ALL_tmp[:,-1]);
        return Snx;

    perms = list(itertools.product([-1,1], repeat=num_ac))
    true_ac_list = [];
    for i in range(len(perms)): #2**num_actions
        ac_tuple = perms[i];
        ac_list = [(tmp1==1)*tmp3 +  (tmp1==-1)*tmp2 for tmp1,tmp2,tmp3 in zip(ac_tuple,min_list,max_list)]; 
        true_ac_list.append(ac_list);
            
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
        values = V_0(next_states_[:,[0,1,2]]);
        
        
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
                values = np.max((values,V_0(next_states_[:,[0,1,2]])),axis=0);
        
        values_ = values;#V_0(next_states_[:,[0,1,2]]);
        pre_compare_vals_ = values_.reshape([-1,ALL_x.shape[0]]).T;         #Changed to values instead of values_
        final_v = [];
        final_v_ = [];
        per = len(perms_);
        for k in range(len(perms)):
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
            
        return sess.run(make_hot,{hot_input:index_best_a_}),sess.run(make_hot_,{hot_input:index_best_b_})

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

    def getTraj(ALL_x,F_PI=[],F_PI_=[],subSamples=1,StepsLeft=None,Noise=False, Static=False, justV=False, disturb = -1, steps = -1):

        current_params = sess.run(C_func_vars);
        current_params_ = sess.run(D_func_vars);
        
        if(StepsLeft == None): StepsLeft = len(F_PI);        
        
        next_states_ = ALL_x;
        traj = [next_states_];
        actions = [];
        
        values = V_0(next_states_[:,[0,1,2]]);
        
        if Static:
            if(steps < 0):
                disturb = input("Disturbance Policy = ")
                steps = input("How Many Steps? ")
            for ind in range(len(F_PI[len(F_PI)-StepsLeft])): #Reload pi*(x,t+dt) parameters
                sess.run(C_func_vars[ind].assign(F_PI[len(F_PI)-StepsLeft][ind]));
            for ind in range(len(F_PI_[len(F_PI_)-disturb])): #Reload pi*(x,t+dt) parameters
                sess.run(D_func_vars[ind].assign(F_PI_[len(F_PI_)-disturb][ind]));  
            for i in range(steps):                            
                for _ in range(subSamples):
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
                    
                    next_states_ = RK4(next_states_,dt/float(subSamples),opt_a,opt_b);
                    if not justV: 
                        traj.append(next_states_); 
                        actions.append(hots.argmax(axis=1)[0]);  
                    values = np.max((values,V_0(next_states_[:,[0,1,2]])),axis=0);
                    if i % 20 == 0:
                        print(i)
        else:      
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

        print(str(next_states_))        
                
        return traj,actions,values

    def ConvCosSin(ALL_x):
        sin_psi = np.sin(ALL_x[:,[6]])
        cos_psi = np.cos(ALL_x[:,[6]])
        pos = ALL_x[:,[0,1,2]]/5.0;
        vel = ALL_x[:,[3,4,5]]/10.0;
        ret_val = np.concatenate((pos,vel,sin_psi,cos_psi),axis=1)
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
        ALL_PI,ALL_PI_ = pickle.load( open( "policies7D_P&Tcoupled_h60_h60_h60.pkl", "rb" ) );
        cc = 0;
        while True:
            state_get = input('State: ');
            sub_smpl = input('SUBSAMPLING: ');
            pause_len = input('Pause: ')
            s_left = input("How many steps left to go (max. "+str(len(ALL_PI))+")? -> ")
            noise = input("Noise? (0/1): ")
            stat = input("Static? (0/1): ")
            traj,act,value = getTraj(state_get,F_PI=ALL_PI,F_PI_=ALL_PI_,subSamples=sub_smpl,StepsLeft=s_left,Noise=noise,Static=stat);
            print(value)
            act.append(act[-1]);
            all_to = np.concatenate(traj);
            plt.scatter(all_to[:,[0]],all_to[:,[1]],color=act_color[cc % len(act_color)])
            plt.pause(pause_len);
            cc = cc + 1;
            #plt.colorbar()       
    elif(imp == 2.0):
        ALL_PI,ALL_PI_ = pickle.load( open( "policies7D_P&Tcoupled_h60_h60_h60.pkl", "rb" ) );
        cc = 0;
        dist_bound = input("Distance: ")
        state_get = np.random.uniform(-5.0,5.0,(nrolls,layers[0]-1));
        state_get[:,:3] = dist_bound*state_get[:,:3]/np.linalg.norm(state_get[:,:3],axis=1,keepdims=True)
        sub_smpl = input('SUBSAMPLING: ');
        s_left = input("How many steps left to go (max. "+str(len(ALL_PI))+")? -> ")
        noise = input("Noise? (0/1): ");
        stat = input("Static? (0/1): ");
        traj,act,values = getTraj(state_get,F_PI=ALL_PI,F_PI_=ALL_PI_,subSamples=sub_smpl,StepsLeft=s_left,Noise=noise,Static=stat,justV=True);
        values = values + 1.0
        print(values.shape)
        filt = (values < dist_bound).T[0];
        print(filt.shape)
        subset = state_get[filt]
        print(len(subset))
        plt.hist(values,bins=100)   
        plt.pause(10)
        tracking_error_bound = np.max(abs(subset[:,:3]),axis=0)
        print(tracking_error_bound)
        print(subset)
        save_dict = {}
        save_dict["weights"]=(ALL_PI,ALL_PI_)
        save_dict["c_layers"]=layers1
        save_dict["d_layers"]=layers_
        save_dict["control_bounds_upper"]= max_list
        save_dict["control_bounds_lower"]= min_list
        save_dict["tracking_error_bound"]= tracking_error_bound
        save_dict["planner_params"]={"max_speed":[0.5,0.5,0.5],"max_vel_dist":[0.0,0.0,0.0],"max_acc_dist":[0.0,0.0,0.0]}
        save_dict["normalization_args"] = [5.0,5.0,5.0,10.0,10.0,10.0,-1]
        pickle.dump(save_dict,open( "TESTpolicies7Dubins_PT_h100_h100.pkl", "wb" ));
    elif(imp == 3.0):
        ALL_PI,ALL_PI_ = pickle.load( open( "policies7D_P&Tcoupled_h100_h100.pkl", "rb" ) );
        fig = plt.figure()
        tmp = 1
        vals = []
        for i in range(1,len(ALL_PI),2):
            for j in range(1,len(ALL_PI_),2):
                state_get = np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
                sub_smpl = 2
                pause_len = 10
                s_left = i
                noise = 0
                stat = 1
                traj,act,v = getTraj(state_get,F_PI=ALL_PI,F_PI_=ALL_PI_,subSamples=sub_smpl,StepsLeft=s_left,Noise=noise,Static=stat,disturb=j,steps=1000);
                vals.append(v[0][0])
                all_to = np.concatenate(traj);
                ax = fig.add_subplot(len(ALL_PI)/2,len(ALL_PI_)/2,tmp)
                tmp = tmp + 1
                ax.scatter(all_to[:,[0]],all_to[:,[2]])
                plt.pause(1.0);
        vals = np.array(vals).reshape((10,10))
        pickle.dump(vals,open( "avore.pkl", "wb" ));
        plt.pause(1000.0)
        cc = cc + 1;      
    else:

        for i in xrange(iters):
            
            if(np.mod(i,renew) == 0 and i is not 0):       
                
                ALL_PI.insert(0,sess.run(C_func_vars));
                ALL_PI_.insert(0,sess.run(D_func_vars));        
                            
                
                k = 0;
                ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]-1));
                ALL_x[:,[3,4,5]] = ALL_x[:,[3,4,5]]*2.0
                ALL_x[:,[6]] = np.mod(ALL_x[:,[6]]*np.pi/5.0,2.0*np.pi);  
                PI_c,PI_d = getPI(ALL_x,ALL_PI,ALL_PI_,subSamples=1);
                pre_ALL_x = ConvCosSin(ALL_x);
                
                ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls/100,layers[0]-1));
                ALL_x_[:,[3,4,5]] = ALL_x_[:,[3,4,5]]*2.0
                ALL_x_[:,[6]] = np.mod(ALL_x_[:,[6]]*np.pi/5.0,2.0*np.pi); 
                PI_c_,PI_d_ = getPI(ALL_x_,ALL_PI,ALL_PI_,subSamples=1);
                pre_ALL_x_ = ConvCosSin(ALL_x_);
    
                t = t - dt; 
                print('Again.')
                
            #elif(i is 0):
            elif(np.mod(i,renew) == 0 and i is 0):
    
    #            sess.run(set_to_zero);
                t = time.time()
                ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]-1));
                ALL_x[:,[3,4,5]] = ALL_x[:,[3,4,5]]*2.0
                ALL_x[:,[6]] = np.mod(ALL_x[:,[6]]*np.pi/5.0,2.0*np.pi);                
                PI_c,PI_d = getPI(ALL_x,F_PI=[],F_PI_=[],subSamples=1);
                pre_ALL_x = ConvCosSin(ALL_x);
                elapsed = time.time() - t
                print("Compute Data Time = "+str(elapsed))
                
                ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls/100,layers[0]-1));
                ALL_x_[:,[3,4,5]] = ALL_x_[:,[3,4,5]]*2.0
                ALL_x_[:,[6]] = np.mod(ALL_x_[:,[6]]*np.pi/5.0,2.0*np.pi); 
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
    
        pickle.dump([ALL_PI,ALL_PI_],open( "policies7D_P&Tcoupled_h100_h100.pkl", "wb" ));
    #    while True:19
    #        state_get = input('State: ');
    #        if(state_get == 0):
    #            break;
    #        _,VAL = getPI(state_get,ALL_PI);
    #        print(str(VAL));

num_ac = 4;
layers1 = [8,60,60,60,2**num_ac];
t_hor = -2.0;

main(layers1,t_hor,0,200000,5000,0.001,0.95,99,5000,1.0,0); #MODIFIED!!!!!!!!!! 2000000 -> 2000000
