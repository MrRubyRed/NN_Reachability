# -*-: coding: utf-8 -*-
"""
Parent: CopyCopyNew_4D.py
In this script you implemented a rechability algorithm which:
- Takes in a  NN which computes a policy directly.( X -> NN -> softmax(output) which selects the actions. 
- The policy parameters are stored and used for subsequent training.

Created on Thu May 26 13:37:12 2016

@author: vrubies 

"""

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
import warnings
#warnings.filterwarnings("error")
            

def main(layers,t_hor,ind,nrolls,bts,ler_r,mom,teps,renew,imp,q):
    # Quad Params
    m0 = 1.5;
    m1 = 0.5;
    m2 = 0.75;
    L1 = 0.5; l1 = L1/2.0;
    L2 = 0.75; l2 = L2/2.0;
    I1 = m1*L1**2 / 12.0;
    I2 = m2*L2**2 / 12.0;
    
    d1 = m0+m1+m2;
    d2 = (m1/2.0 + m2)*L1
    d3 = m2*l2
    d4 = (m1/3.0 + m2)*L1**2
    d5 = m2*L1*l2
    d6 = m2*l2**2 + I2
    
    

    g = 9.81;

    f1 = (m1*l1 + m2*L1)*g
    f2 = m2*l2*g 
    
    min_list = [-1.0];
    max_list = [1.0];
    
    print 'Starting worker-' + str(ind)

    f = 1;
    Nx = 100*f + 1;
    minn = [-5.0,-10.0,-5.0,-10.0,0.0,-10.0];
    maxx = [ 5.0, 10.0, 5.0, 10.0,2*np.pi, 10.0];
    
#    X = np.linspace(minn[0],maxx[0],Nx);
#    Y = np.linspace(minn[2],maxx[2],Nx);
#    Z = np.linspace(minn[4],maxx[4],Nx);
#    X_,Y_,Z_ = np.meshgrid(X, Y, Z);    
#    X,Y = np.meshgrid(X, Y);
#    XX = np.reshape(X,[-1,1]);
#    YY = np.reshape(Y,[-1,1]);
#    XX_ = np.reshape(X_,[-1,1]);
#    YY_ = np.reshape(Y_,[-1,1]);
#    ZZ_ = np.reshape(Z_,[-1,1]); grid_check = np.concatenate((XX_,np.ones(XX_.shape),YY_,np.ones(XX_.shape),ZZ_,np.zeros(XX_.shape)),axis=1);
#    grid_eval = np.concatenate((XX,YY,0.0*np.ones(XX.shape),6.0*np.ones(XX.shape)),axis=1);
#    grid_eval_ = np.concatenate((XX,YY,(2.0/3.0)*np.pi*np.ones(XX.shape),6.0*np.ones(XX.shape)),axis=1);
#    grid_eval__ = np.concatenate((XX,YY,(4.0/3.0)*np.pi*np.ones(XX.shape),6.0*np.ones(XX.shape)),axis=1);
#    grid_evall = np.concatenate((XX,YY,0.0*np.ones(XX.shape),12.0*np.ones(XX.shape)),axis=1);
#    grid_evall_ = np.concatenate((XX,YY,(2.0/3.0)*np.pi*np.ones(XX.shape),12.0*np.ones(XX.shape)),axis=1);
#    grid_evall__ = np.concatenate((XX,YY,(4.0/3.0)*np.pi*np.ones(XX.shape),12.0*np.ones(XX.shape)),axis=1);    


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
    num_ac = 1;
    iters = int(np.abs(t_hor)/dt)*renew + 1; 
    ##################### INSTANTIATIONS #################
    states,y,Tt,L,l_r,lb,reg, cross_entropy = TransDef("Critic",False,layers,depth,incl,center);
    ola1 = tf.argmax(Tt,dimension=1)
    ola2 = tf.argmax(y,dimension=1)
    ola3 = tf.equal(ola1,ola2)
    accuracy = tf.reduce_mean(tf.cast(ola3, tf.float32));
    #a_layers = layers;
    #a_layers[-1] = 2; #We have two actions
    #states_,y_,Tt_,l_r_,lb_,reg_ = TransDef("Actor",False,a_layers,depth,incl,center,outp=True);
    
    V_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Critic');
    #A_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Actor');
    
    #var_grad = tf.gradients(Tt_,states_)[0]
    var_grad_ = tf.gradients(Tt,states)[0]
    grad_x = tf.slice(var_grad_,[0,0],[-1,layers[0]-1]);
    #theta = tf.trainable_variables();

    set_to_zero = []
    for var  in sorted(V_func_vars,        key=lambda v: v.name):
        set_to_zero.append(var.assign(tf.zeros(tf.shape(var))))
    set_to_zero = tf.group(*set_to_zero)
    
    set_to_not_zero = []
    for var  in sorted(V_func_vars,        key=lambda v: v.name):
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
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=nu,momentum=mom);
    #gvs = optimizer.compute_gradients(L,theta);
    #capped_gvs = [(tf.clip_by_value(grad, -3., 3.), var) for grad, var in gvs];
    #train_step = optimizer.apply_gradients(gvs);
    #train_step = tf.train.AdagradOptimizer(learning_rate=nu,initial_accumulator_value=0.5).minimize(L);

    hot_input = tf.placeholder(tf.int64,shape=(None));   
    make_hot = tf.one_hot(hot_input, 2**num_ac, on_value=1, off_value=0)

    # INITIALIZE GRAPH
    theta = tf.trainable_variables();
    sess = tf.Session();
    init = tf.initialize_all_variables();
    sess.run(init);

    def V_0(x):
        #return np.linalg.norm(x,ord=np.inf,axis=1,keepdims=True)
        return np.linalg.norm(x,axis=1,keepdims=True)

    def p_corr(ALL_x):
        ALL_x = np.mod(ALL_x + np.pi,2.0*np.pi) - np.pi;
        return ALL_x;

    def F(ALL_x,opt_a,opt_b):
       v1 = ALL_x[:,3,None];
       w1 = ALL_x[:,4,None]; 
       w2 = ALL_x[:,5,None]; 
       cos_t1 = np.cos(ALL_x[:,1,None]);
       sin_t1 = np.sin(ALL_x[:,1,None]);
       t1 = np.cos(ALL_x[:,1,None]);
       cos_t2 = np.cos(ALL_x[:,2,None]);
       sin_t2 = np.sin(ALL_x[:,2,None]);
       t2 = np.cos(ALL_x[:,2,None]);
       
       #n_c = d4*(d3*cos_t2)**2.0 + d1*(d5*np.cos(t1-t2))**2.0 + d6*((d2*cos_t1)**2.0 -d1*d4) - 2.0*d2*d3*d5*cos_t1*np.cos(t1-t2)*cos_t2
       #n_c = (d1*d4*d6 - d1*(d5*np.cos(t2-t1))**2.0 - d6*(d2*cos_t1)**2.0 + 2.0*d2*d3*d5*cos_t2*cos_t1*np.cos(t2-t1) - d4*(d3*cos_t2)**2.0);
       try:       
           D11 = (d4*d6 - (np.cos(t1-t2)*d5)**2.0);              D12 = (d3*d5*cos_t2*np.cos(t1-t2) - d2*d6*cos_t1);  D13 = (d2*d5*np.cos(t1-t2)*cos_t1 - d3*d4*cos_t2);
           D21 = D12;                                            D22 = (d1*d6 - (d3*cos_t2)**2);                      D23 = (d2*d3*cos_t2*cos_t1 - d1*d5*np.cos(t1-t2));
           D31 = D13;                                            D32 = D23;                                           D33 = (d1*d4 - (d2*cos_t1)**2.0);       
    
           n_c_ = L1**2*L2**2*m2*(m0*m1 + m1**2*sin_t1**2 + m1*m2*sin_t1**2 + m0*m2*np.sin(t1-t2)**2)
           n_c = d1*D11 + d2*cos_t1*D12 + d3*cos_t2*D13
           n_c2 = d2*cos_t1*D21 + d4*D22 + d5*np.cos(t1-t2)*D23
           n_c3 = d3*cos_t2*D31 + d5*np.cos(t1-t2)*D32 + d6*D33
    
           C11 = 0.0; C12 = -d2*sin_t1*w1;          C13 = -d3*sin_t2*w2;
           C21 = 0.0; C22 = 0.0;                    C23 = d5*np.sin(t1-t2)*w2;
           C31 = 0.0; C32 = -d5*np.sin(t1-t2)*w1;   C33 = 0.0;
           
           G1 = 0.0; G2 = -f1*sin_t1; G3 = -f2*sin_t2;
           
           DC11 = 0.0; DC12 = D11*C12 + D13*C32; DC13 = D11*C13 + D12*C23;
           DC21 = 0.0; DC22 = D21*C12 + D23*C32; DC23 = D21*C13 + D22*C23;
           DC31 = 0.0; DC32 = D31*C12 + D33*C32; DC33 = D31*C13 + D32*C23;
           
           DG1 = D11*G1 + D12*G2 + D13*G3;
           DG2 = D21*G1 + D22*G2 + D23*G3;
           DG3 = D31*G1 + D32*G2 + D33*G3;       
       
           col1 = v1;
           col2 = w1;
           col3 = w2;
           col4 = ( -(DC11*v1 + DC12*w1 + DC13*w2) - 0.1*v1 - DG1 + D11*opt_a)/n_c_
           col5 = ( -(DC21*v1 + DC22*w1 + DC23*w2) - 0.1*w1 - DG2 + D21*opt_a)/n_c_
           col6 = ( -(DC31*v1 + DC32*w1 + DC33*w2) - 0.1*w2 - DG3 + D31*opt_a)/n_c_
       except RuntimeWarning:
           print("Whoops...")
       
       return np.concatenate((col1,col2,col3,col4,col5,col6),axis=1);

   #Dynamics
#    
#    (a) d1 = m0+m1+m2;
#    (b) d2 = m1*l1 + m2*L2
#    (c) d3 = m2*l2
#    (d) d4 = m1*l1**2 + m2*L1**2 + I1
#    (e) d5 = m2*L1*l2
#    (f) d6 = m2*l2**2 + I2
#
#    g = 9.81;
#
#    f1 = (m1*l1 + m2*L1)*g
#    f2 = m2*l2*g 


    ####################### RECURSIVE FUNC ####################

    def RK4(ALL_x,dtt,opt_a,opt_b):

        k1 = F(ALL_x,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k2)
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k1);
        ALL_tmp[:,[1,2]] = p_corr(ALL_tmp[:,[1,2]]);

        k2 = F(ALL_tmp,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k3)
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k2);
        ALL_tmp[:,[1,2]] = p_corr(ALL_tmp[:,[1,2]]);

        k3 = F(ALL_tmp,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k4)
        ALL_tmp = ALL_x + np.multiply(dtt,k3);
        ALL_tmp[:,[1,2]] = p_corr(ALL_tmp[:,[1,2]]);

        k4 = F(ALL_tmp,opt_a,opt_b);  #### !!!

        Snx = ALL_x + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4)); #np.multiply(dtt,k1)
        Snx[:,[1,2]] = p_corr(Snx[:,[1,2]]);
        return Snx;

    perms = list(itertools.product([-1,1], repeat=num_ac))
    true_ac_list = [];
    for i in range(len(perms)): #2**num_actions
        ac_tuple = perms[i];
        ac_list = [(tmp1==1)*tmp3 + (tmp1==-1)*tmp2 for tmp1,tmp2,tmp3 in zip(ac_tuple,min_list,max_list)]; 
        true_ac_list.append(ac_list);
    
    def Hot_to_Cold(hots,ac_list):
        a = hots.argmax(axis=1);
        a = np.asarray([ac_list[i] for i in a]);
        return a;
    
    def getPI(ALL_x,F_PI=[],subSamples=1): #Things to keep in MIND: You want the returned value to be the minimum accross a trajectory.

        current_params = sess.run(theta);

        #perms = list(itertools.product([-1,1], repeat=num_ac))
        next_states = [];
        for i in range(len(perms)):
            opt_a = np.asarray(true_ac_list[i])*np.ones([ALL_x.shape[0],1]);
            Snx = ALL_x;
            for _ in range(subSamples): 
                Snx = RK4(Snx,dt/float(subSamples),opt_a,None);
            next_states.append(Snx);
        next_states = np.concatenate(next_states,axis=0);
        values = V_0(next_states[:,[1,2,3]]);
        
        for params in F_PI:
            for ind in range(len(params)): #Reload pi*(x,t+dt) parameters
                sess.run(theta[ind].assign(params[ind]));
           
            for _ in range(subSamples):
                hots = sess.run(Tt,{states:ConvCosSin(next_states)});
                opt_a = Hot_to_Cold(hots,true_ac_list)                 
                next_states = RK4(next_states,dt/float(subSamples),opt_a,None);
                values = np.max((values,V_0(next_states[:,[1,2,3]])),axis=0);
        
        values_ = values#V_0(next_states);
        compare_vals_ = values_.reshape([-1,ALL_x.shape[0]]).T;         #Changed to values instead of values_
        index_best_a_ = compare_vals_.argmin(axis=1)                    #Changed to ARGMIN
        values_ = np.min(compare_vals_,axis=1,keepdims=True);
        
        filterr = 0#np.max(compare_vals_,axis=1) > -0.8
        #index_best_a_ = index_best_a_[filterr]
        #values_ = values_[filterr]
        #print("States filtered out: "+str(len(filterr)-np.sum(filterr)))
        
        for ind in range(len(current_params)): #Reload pi*(x,t+dt) parameters
            sess.run(theta[ind].assign(current_params[ind]));
        
        return sess.run(make_hot,{hot_input:index_best_a_}),values_,filterr

#    def getTraj(ALL_x,F_PI=[],subSamples=1,StepsLeft=None,Noise = False):
#
#        current_params = sess.run(theta);
#        
#        if(StepsLeft == None): StepsLeft = len(F_PI);        
#        
#        next_states = ALL_x;
#        traj = [next_states];
#        actions = [];
#              
#        for params in F_PI[len(F_PI)-StepsLeft:]:
#            for ind in range(len(params)): #Reload pi*(x,t+dt) parameters
#                sess.run(theta[ind].assign(params[ind]));
#            
#            hots = sess.run(Tt,{states:ConvCosSin(next_states)});
#            opt_a = Hot_to_Cold(hots,true_ac_list)
#            for _ in range(subSamples):
#                next_states = RK4(next_states,dt/float(subSamples),opt_a,None);
#                if Noise:
#                    next_states = next_states + np.random.normal(size=next_states.shape)*0.01
#                traj.append(next_states);
#                actions.append(hots.argmax(axis=1)[0]);
#                #values = np.min((values,V_0(next_states[:,[0,1]])),axis=0);    
#
#        for ind in range(len(current_params)): #Reload pi*(x,t+dt) parameters
#            sess.run(theta[ind].assign(current_params[ind]));
#                        
#        return traj,actions,V_0(next_states[:,[0,2]]);                 

    def getTraj(ALL_x,F_PI=[],subSamples=1,StepsLeft=None,Noise=False, Static=False):

        current_params = sess.run(theta);
        
        if(StepsLeft == None): StepsLeft = len(F_PI);        
        
        next_states = ALL_x;
        traj = [next_states];
        actions = [];
        
        if Static:
            steps = input("How Many Steps? ")
            for ind in range(len(F_PI[len(F_PI)-StepsLeft])): #Reload pi*(x,t+dt) parameters
                sess.run(theta[ind].assign(F_PI[len(F_PI)-StepsLeft][ind])); 
            for i in range(steps):                            
                for _ in range(subSamples):
                    tmp = ConvCosSin(next_states);
                    hots = sess.run(Tt,{states:tmp});
                    opt_a = Hot_to_Cold(hots,true_ac_list)   
                    if Noise == False:
                        next_states = next_states + np.random.normal(size=next_states.shape)*0.01
                    
                    next_states = RK4(next_states,dt/float(subSamples),opt_a,None);
                    traj.append(next_states); 
                    actions.append(hots.argmax(axis=1)[0]);   
        else:      
            for params in F_PI[len(F_PI)-StepsLeft:]:
                for ind in range(len(params)): #Reload pi*(x,t+dt) parameters
                    sess.run(theta[ind].assign(params[ind]));
                            
                for _ in range(subSamples):
                    tmp = ConvCosSin(next_states);
                    hots = sess.run(Tt,{states:tmp});
                    opt_a = Hot_to_Cold(hots,true_ac_list)   
                    if Noise == False:
                        next_states = next_states + np.random.normal(size=next_states.shape)*0.01                    
                    next_states = RK4(next_states,dt/float(subSamples),opt_a,None);
                    traj.append(next_states); 
                    actions.append(hots.argmax(axis=1)[0]);   

        for ind in range(len(current_params)): #Reload pi*(x,t+dt) parameters
            sess.run(theta[ind].assign(current_params[ind]));       
                
        return traj,actions,V_0(next_states[:,[0,2]])

    def ConvCosSin(ALL_x):
        sin_phi = np.sin(ALL_x[:,[1,2]])
        cos_phi = np.cos(ALL_x[:,[1,2]])
        pos = ALL_x[:,[0]]/5.0;
        vel = ALL_x[:,[3]]/10.0;
        arate = ALL_x[:,[4,5]]/5.0;
        ret_val = np.concatenate((pos,vel,arate,sin_phi,cos_phi),axis=1)
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
    nunu = lr_schedule.value(k);               

    act_color = ['r','g','b','y'];
    if(imp == 1.0):
        ALL_PI = pickle.load( open( "policies6Dreach_h50.pkl", "rb" ) );
        cc = 0;
        while True:
            state_get = input('State: ');
            sub_smpl = input('SUBSAMPLING: ');
            pause_len = input('Pause: ')
            s_left = input("How many steps left to go (max. "+str(len(ALL_PI))+")? -> ")
            noise = input("Noise? (0/1): ")
            stat = input("Static? (0/1): ")
            traj,act,_ = getTraj(state_get,F_PI=ALL_PI,subSamples=sub_smpl,StepsLeft=s_left,Noise=noise,Static=stat);
            #act.append(act[-1]);
            all_to = np.concatenate(traj);
            plt.scatter(all_to[:,[1]],all_to[:,[2]])#,color=act_color[cc % len(act_color)])
            plt.pause(pause_len);
            cc = cc + 1;
            #plt.colorbar()

    
    for i in xrange(iters):
        
        if(np.mod(i,renew) == 0 and i is not 0):       
            
            ALL_PI.insert(0,sess.run(theta))            

            plt.figure(2) #TODO: Figure out why facing up vs facing down has same action... -> Solved: colors in a scatter plot only depend on the labels
            plt.clf();
            ALL_xx = np.array([[0.0,0.1,0.1,0.0,0.0,0.0],
                               [0.0,np.pi,np.pi,0.0,0.0,0.0],
                               [0.5,0.0,0.0,0.0,0.0,0.0],
                               [0.0,-np.pi/2,np.pi/2,0.0,0.0,0.0]]);
            for tmmp in range(ALL_xx.shape[0]):                   
                traj,act,_ = getTraj(ALL_xx[[tmmp],:],F_PI=ALL_PI,subSamples=10);
                #act.append(act[-1]);
                all_to = np.concatenate(traj);
                plt.scatter(all_to[:,[1]],all_to[:,[2]])#c=[act_color[ii] for ii in act]);          

            plt.pause(0.25)   
                                             
            
            k = 0;
            ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]-2));
            ALL_x[:,[1,2]] = ALL_x[:,[1,2]]*np.pi/5.0;
            ALL_x[:,[3]] = ALL_x[:,[3]]*2.0; 
            ALL_x[:,[4,5]] = ALL_x[:,[4,5]];
            PI,_,filterr = getPI(ALL_x,ALL_PI,subSamples=3);
            #ALL_x = ALL_x[filterr]
            pre_ALL_x = ConvCosSin(ALL_x);
            
            ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls,layers[0]-2));
            ALL_x_[:,[1,2]] = ALL_x_[:,[1,2]]*np.pi/5.0;
            ALL_x_[:,[3]] = ALL_x_[:,[3]]*2.0; 
            ALL_x_[:,[4,5]] = ALL_x_[:,[4,5]];
            PI_,_,filterr = getPI(ALL_x_,ALL_PI,subSamples=3);
            #ALL_x_ = ALL_x_[filterr]
            pre_ALL_x_ = ConvCosSin(ALL_x_);

            t = t - dt; 
            print('Again.')
            
        elif(np.mod(i,renew) == 0 and i is 0):

#            sess.run(set_to_zero);
            t = time.time()
            ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]-2));
            ALL_x[:,[1,2]] = ALL_x[:,[1,2]]*np.pi/5.0;
            ALL_x[:,[3]] = ALL_x[:,[3]]*2.0; 
            ALL_x[:,[4,5]] = ALL_x[:,[4,5]];           
            PI,_,filterr = getPI(ALL_x,F_PI=[],subSamples=3);
            #ALL_x = ALL_x[filterr]
            pre_ALL_x = ConvCosSin(ALL_x);
            elapsed = time.time() - t
            print("Compute Data Time = "+str(elapsed))
            
            ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls,layers[0]-2));
            ALL_x_[:,[1,2]] = ALL_x_[:,[1,2]]*np.pi/5.0;
            ALL_x_[:,[3]] = ALL_x_[:,[3]]*2.0; 
            ALL_x_[:,[4,5]] = ALL_x_[:,[4,5]];
            PI_,_,filterr = getPI(ALL_x_,F_PI=[],subSamples=3);
            #ALL_x_ = ALL_x_[filterr]
            pre_ALL_x_ = ConvCosSin(ALL_x_);           
#            sess.run(set_to_not_zero);

            

        # |||||||||||| ----  PRINT ----- |||||||||||| 

        if(np.mod(i,200) == 0):

            train_acc = sess.run(accuracy,{states:pre_ALL_x,y:PI});
            test_acc = sess.run(accuracy,{states:pre_ALL_x_,y:PI_});       
            print str(i) + ") | TR_ACC = " + str(train_acc) + " | TE_ACC = " + str(test_acc) + " | Lerning Rate = " + str(nunu)
            
        nunu = 0.01
        tmp = np.random.randint(len(ALL_x), size=bts);
        sess.run(train_step, feed_dict={states:pre_ALL_x[tmp],y:PI[tmp],nu:nunu});

    pickle.dump(ALL_PI,open( "policies6Dreach_h50.pkl", "wb" ));

num_ac = 1;
layers1 = [8,23,23,2**num_ac];
t_hor = -2.0;

main(layers1,t_hor,0,1000000,50000,0.001,0.95,99,2000,1.0,0);
