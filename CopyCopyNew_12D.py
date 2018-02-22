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
            

def main(layers,t_hor,ind,nrolls,bts,ler_r,mom,teps,renew,imp,q):
    # Quad Params
    u1_max = 0.17;
    u1_min = 0;
    u2_max = 0.017;
    u2_min = 0;
    u3_max = 0.017;
    u3_min = 0;
    u4_max = 0.017;
    u4_min = 0;    
    max_list = [u1_max,u2_max,u3_max,u4_max];
    min_list = [u1_min,u2_min,u3_min,u4_min];
    
    I = [0.0224,0.0224,0.0436]; Ix = I[0]; Iy = I[1]; Iz = I[2];
    m = 0.65;
    L_ = 0.156;
    g = 9.8;


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
    dt = 0.05;                                                                 #VAR
    num_ac = 4;
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
        return np.linalg.norm(x,ord=np.inf,axis=1,keepdims=True) - 1.0
        #return np.linalg.norm(x,axis=1,keepdims=True) - 1.0

    def p_corr(ALL_x):
        ALL_x = np.mod(ALL_x,2.0*np.pi);
        return ALL_x;

    def F(ALL_x,opt_a,opt_b):#(grad,ALL_x):
       cos_phi = np.cos(ALL_x[:,6,None])
       sin_phi = np.sin(ALL_x[:,6,None])
       cos_the = np.cos(ALL_x[:,7,None])
       sin_the = np.sin(ALL_x[:,7,None]); tan_the = np.tan(ALL_x[:,7,None])
       cos_psi = np.cos(ALL_x[:,8,None]) 
       sin_psi = np.sin(ALL_x[:,8,None])       
    
    
       col1 = ALL_x[:,3,None]
       col2 = ALL_x[:,4,None]
       col3 = ALL_x[:,5,None]
       
       col4 = -(cos_phi*sin_the*cos_psi + sin_phi*sin_psi)*opt_a[:,0,None]/m;
       col5 = -(cos_phi*sin_the*cos_psi - sin_phi*cos_psi)*opt_a[:,0,None]/m;
       col6 = g - (cos_phi*cos_the)*opt_a[:,0,None]/m
       
       col7 = ALL_x[:,9,None] + sin_phi*tan_the*ALL_x[:,10,None] + cos_phi*tan_the*ALL_x[:,11,None]
       col8 = cos_phi*ALL_x[:,10,None] - sin_phi*ALL_x[:,11,None];
       col9 = (sin_phi/cos_phi)*ALL_x[:,10,None] + (cos_phi/cos_the)*ALL_x[:,11,None];
       
       col10 = ALL_x[:,10,None]*ALL_x[:,11,None]*((Iy-Iz)/Ix) + (L_/Ix)*opt_a[:,1,None]
       col11 = ALL_x[:,9,None]*ALL_x[:,11,None]*((Iz-Ix)/Iy) + (L_/Iy)*opt_a[:,2,None]
       col12 = ALL_x[:,9,None]*ALL_x[:,10,None]*((Ix-Iy)/Iz) + (L_/Iz)*opt_a[:,3,None]
       
       return np.concatenate((col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12),axis=1); 

      #     \dot x    x_1  = x_4
      #     \dot y    x_2  = x_5
      #     \dot z    x_3  = x_6
      #     \dot vx   x_4  = -(\cos x_7 \sin x_8 \cos x_9 + \sin x_7 \sin x_9) u_1/m
      #     \dot vy   x_5  = -(\cos x_7 \sin x_8 \sin x_9 - \sin x_7 \cos x_9) u_1/m
      #     \dot vz   x_6  = g - (\cos x_7 \cos x_8) u_1/m
      #     \dot phi  x_7  = x_10 + \sin x_7 \tan(x_8) x_11 + \cos x_7 \tan(x_8) x_12 
      #     \dot the  x_8  = \cos x_7 x_11 - \sin x_7 x_12                           
      #     \dot psi  x_9  = (\sin x_7/\cos x_8)*x_11 + (\cos x_7/\cos x_8) x_12   <---------
      #     \dot wphi x_10 = x_11 x_12 (I_y - I_z)/I_x + L/I_x u_2
      #     \dot wthe x_11 = x_10 x_12 (I_z - I_x)/I_y + L/I_y u_3
      #     \dot wpsi x_12 = x_10 x_11 (I_x - I_y)/I_z + 1/I_z u_4

    ####################### RECURSIVE FUNC ####################

    def RK4(ALL_x,dtt,opt_a,opt_b):

        k1 = F(ALL_x,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k2)
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k1);
        ALL_tmp[:,[6,7,8]] = p_corr(ALL_tmp[:,[6,7,8]]);

        k2 = F(ALL_tmp,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k3)
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k2);
        ALL_tmp[:,[6,7,8]] = p_corr(ALL_tmp[:,[6,7,8]]);

        k3 = F(ALL_tmp,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k4)
        ALL_tmp = ALL_x + np.multiply(dtt,k3);
        ALL_tmp[:,[6,7,8]] = p_corr(ALL_tmp[:,[6,7,8]]);

        k4 = F(ALL_tmp,opt_a,opt_b);  #### !!!

        Snx = ALL_x + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4)); #np.multiply(dtt,k1)
        Snx[:,[6,7,8]] = p_corr(Snx[:,[6,7,8]]);
        return Snx;

    perms = list(itertools.product([-1,1], repeat=num_ac))
    true_ac_list = [];
    for i in range(len(perms)): #2**num_actions
        ac_tuple = perms[i];
        ac_list = [(tmp1==1)*tmp3 +  (tmp1==-1)*tmp2 for tmp1,tmp2,tmp3 in zip(ac_tuple,min_list,max_list)]; 
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
        values = V_0(next_states[:,[0,1,2]]);
        
        for params in F_PI:
            for ind in range(len(params)): #Reload pi*(x,t+dt) parameters
                sess.run(theta[ind].assign(params[ind]));

            hots = sess.run(Tt,{states:ConvCosSin(next_states)});
            opt_a = Hot_to_Cold(hots,true_ac_list)            
            for _ in range(subSamples):
                next_states = RK4(next_states,dt/float(subSamples),opt_a,None);
                values = np.min((values,V_0(next_states[:,[0,1,2]])),axis=0);
        
        values_ = V_0(next_states[:,[0,1,2]]);
        compare_vals_ = values_.reshape([-1,ALL_x.shape[0]]).T;         #Changed to values instead of values_
        index_best_a_ = compare_vals_.argmin(axis=1)                    #Changed to ARGMIN
        values_ = np.min(compare_vals_,axis=1,keepdims=True);
        
#        filterr = np.min(compare_vals_,axis=1) < 0.0
#        index_best_a_ = index_best_a_[filterr]
#        values_ = values_[filterr]
#        print("States filtered out: "+str(len(filterr)-np.sum(filterr)))
        
        for ind in range(len(current_params)): #Reload pi*(x,t+dt) parameters
            sess.run(theta[ind].assign(current_params[ind]));
        
        return sess.run(make_hot,{hot_input:index_best_a_}),values_,0#filterr

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
#        return traj,V_0(next_states[:,[0,2]]),actions;
        

    def ConvCosSin(ALL_x):
        cos_phi = np.cos(ALL_x[:,6,None])
        sin_phi = np.sin(ALL_x[:,6,None])
        cos_the = np.cos(ALL_x[:,7,None])
        sin_the = np.sin(ALL_x[:,7,None])
        cos_psi = np.cos(ALL_x[:,8,None]) 
        sin_psi = np.sin(ALL_x[:,8,None])   
        pos = ALL_x[:,[0,1,2]]/5.0;
        vel = ALL_x[:,[3,4,5]]/10.0;
        arate = ALL_x[:,[9,10,11]]/30.0;
        ret_val = np.concatenate((pos,vel,arate,cos_phi,sin_phi,cos_the,sin_the,cos_psi,sin_psi),axis=1)
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
    
#    act_color = ['r','g','b','y'];
#    if(imp == 1.0):
#        ALL_PI = pickle.load( open( "policies6D_C&D_h30_h30.pkl", "rb" ) );
#        while (imp == 1.0):
#            state_get = input('State: ');
#            sub_smpl = input('SUBSAMPLING: ');
#            pause_len = input('Pause: ')
#            s_left = input("How many steps left to go (max. "+str(len(ALL_PI))+")? -> ")
#            traj,VAL,act = getTraj(state_get,F_PI=ALL_PI,subSamples=sub_smpl,StepsLeft=s_left,Noise=False);
#            act.append(act[-1]);
#            all_to = np.concatenate(traj);
#            plt.scatter(all_to[:,[0]],all_to[:,[2]],c=[act_color[i] for i in act])
#            #plt.colorbar()
#            plt.pause(pause_len)
#            print(str(VAL));
    
    for i in xrange(iters):
        
        if(np.mod(i,renew) == 0 and i is not 0):       
            
            ALL_PI.insert(0,sess.run(theta))            
            
#            fig = plt.figure(1)
#            plt.clf();
#            _,nn_vals,_ = getTraj(grid_check,ALL_PI,20)
#            fi = (np.abs(nn_vals) < 0.05)
#            mini_reach_ = grid_check[fi[:,0]]
#            ax = fig.add_subplot(111, projection='3d')
#            ax.scatter(mini_reach_[:,0], mini_reach_[:,2], mini_reach_[:,4]);            
#            plt.pause(0.25);            

#            plt.figure(2) #TODO: Figure out why facing up vs facing down has same action... -> Solved: colors in a scatter plot only depend on the labels
#            plt.clf();
#            ALL_xx = np.array([[0.0,0.0,0.0,0.0,0.0,0.0],
#                               [0.0,0.0,1.0,0.0,np.pi/4,0.0],
#                               [0.0,0.0,1.0,0.0,np.pi/2 - 0.3,0.0],
#                               [0.0,0.0,1.0,0.0,np.pi/2 + 0.3,0.0],
#                               [0.0,0.0,1.0,0.0,np.pi/2 + 0.7,0.0],
#                               [0.0,0.0,1.0,0.0,np.pi,0.0]]);
#            for tmmp in range(ALL_xx.shape[0]):                   
#                traj,_,act = getTraj(ALL_xx[[tmmp],:],F_PI=ALL_PI,subSamples=10);
#                act.append(act[-1]);
#                all_to = np.concatenate(traj);
#                plt.scatter(all_to[:,[0]],all_to[:,[2]],c=act);                   
#            plt.pause(0.25)                             


            t = time.time()
            ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]-3));
            ALL_x[:,[3,4,5]] = ALL_x[:,[3,4,5]]
            ALL_x[:,[6,7]] = np.mod(ALL_x[:,[6,7]]*np.pi/20.0,2.0*np.pi);
            ALL_x[:,[8]] = ALL_x[:,[8]]*np.pi/5.0 + np.pi;
            ALL_x[:,[9,10,11]] = ALL_x[:,[9,10,11]]
            PI,_,filterr = getPI(ALL_x,ALL_PI,subSamples=1);
            #ALL_x = ALL_x[filterr]
            pre_ALL_x = ConvCosSin(ALL_x);
            elapsed = time.time() - t
            print("Compute Data Time = "+str(elapsed))
            
            ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls/100,layers[0]-3));
            ALL_x_[:,[3,4,5]] = ALL_x_[:,[3,4,5]]*2.0
            ALL_x_[:,[6,7]] = np.mod(ALL_x_[:,[6,7]]*np.pi/20.0,2.0*np.pi);
            ALL_x_[:,[8]] = ALL_x_[:,[8]]*np.pi/5.0 + np.pi;
            ALL_x_[:,[9,10,11]] = ALL_x_[:,[9,10,11]]
            PI_,_,filterr = getPI(ALL_x_,ALL_PI,subSamples=1);
            #ALL_x_ = ALL_x_[filterr]
            pre_ALL_x_ = ConvCosSin(ALL_x_);

            t = t - dt; 
            print('Again.')
            
        elif(np.mod(i,renew) == 0 and i is 0):

#            sess.run(set_to_zero);
            t = time.time()
            ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]-3));
            ALL_x[:,[3,4,5]] = ALL_x[:,[3,4,5]]
            ALL_x[:,[6,7]] = np.mod(ALL_x[:,[6,7]]*np.pi/20.0,2.0*np.pi);
            ALL_x[:,[8]] = ALL_x[:,[8]]*np.pi/5.0 + np.pi;
            ALL_x[:,[9,10,11]] = ALL_x[:,[9,10,11]]
            PI,_,filterr = getPI(ALL_x,F_PI=[],subSamples=1);
            #ALL_x = ALL_x[filterr]
            pre_ALL_x = ConvCosSin(ALL_x);
            elapsed = time.time() - t
            print("Compute Data Time = "+str(elapsed))
            
            ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls/100,layers[0]-3));
            ALL_x_[:,[3,4,5]] = ALL_x_[:,[3,4,5]]
            ALL_x_[:,[6,7]] = np.mod(ALL_x_[:,[6,7]]*np.pi/20.0,2.0*np.pi);
            ALL_x_[:,[8]] = ALL_x_[:,[8]]*np.pi/5.0 + np.pi;
            ALL_x_[:,[9,10,11]] = ALL_x_[:,[9,10,11]]
            PI_,_,filterr = getPI(ALL_x_,F_PI=[],subSamples=1);
            #ALL_x_ = ALL_x_[filterr]
            pre_ALL_x_ = ConvCosSin(ALL_x_);           
#            sess.run(set_to_not_zero);

            

        # |||||||||||| ----  PRINT ----- |||||||||||| 

        if(np.mod(i,200) == 0):

            #xel = sess.run(L,{states:ALL_x,y:PI});
            #test_e = sess.run(L,{states:ALL_x_,y:PI_});
            train_acc = sess.run(accuracy,{states:pre_ALL_x,y:PI});
            test_acc = sess.run(accuracy,{states:pre_ALL_x_,y:PI_});            
            #o = np.random.randint(len(ALL_x));
            print str(i) + ") | TR_ACC = " + str(train_acc) + " | TE_ACC = " + str(test_acc) + " | Lerning Rate = " + str(nunu)
            #print str(i) + ") | XEL = " + str(xel) + " | Test_E = " + str(test_e) + " | Lerning Rate = " + str(nunu)
            #print str(PI[[o],:]) + " || " + str(sess.run(l_r[-1],{states:ALL_x[[o],:]})) #+ " || " + str(sess.run(gvs[-1],{states:ALL_x,y:PI}))
            
        nunu = 0.01#/(np.sqrt(np.mod(i,renew))+1.0)#lr_schedule.value(i);
        #nunu = ler_r/(np.mod(i,renew)+1.0);
        tmp = np.random.randint(len(ALL_x), size=bts);
        sess.run(train_step, feed_dict={states:pre_ALL_x[tmp],y:PI[tmp],nu:nunu});
        #tmp = np.random.randint(len(reach100s), size=bts);
        #sess.run(train_step, feed_dict={states:reach100s[tmp,:-1],y:reach100s[tmp,-1,None],nu:nunu});

    pickle.dump(ALL_PI,open( "policies6Dreach_h50.pkl", "wb" ));
#    while True:
#        state_get = input('State: ');
#        if(state_get == 0):
#            break;
#        _,VAL = getPI(state_get,ALL_PI);
#        print(str(VAL));

num_ac = 4;
layers1 = [15,80,80,2**num_ac];
t_hor = -0.5;

main(layers1,t_hor,0,500000,5000,0.001,0.95,99,5000,0.0,0);
