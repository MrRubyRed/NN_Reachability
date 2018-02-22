# -*-: coding: utf-8 -*-
"""
Parent: CopyCopyNew_6D.py
In this script you implemented a rechability algorithm which:
- Takes in a  NN which computes a policy directly.( X -> NN -> softmax(output) which selects the actions. 
- The policy parameters are stored and used for subsequent training.

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

def main(layers,t_hor,ind,nrolls,bts,ler_r,mom,teps,renew,imp,q,num_ac):
    
    # Constants
    #   The choices of n0, d1, d0 actually results in a very large
    #   steady state error in the pitch/roll; this seems to be
    #   expected according to Pat's report
    n0 = 10     # Angular dynamics parameters
    d1 = 8
    d0 = 10
    
    kT = 0.91   # Thrust coefficient (vertical direction)
    grav = 9.81    # Acceleration due to gravity (for convenience)
    m = 1.3     # Mass

    # Quad Params
    max_list = [10*np.pi/180.0,  10*np.pi/180.0, 2.0*grav];
    min_list = [-10*np.pi/180.0, -10*np.pi/180.0, 0.0];

    print 'Starting worker-' + str(ind)

    f = 1;
    Nx = 100*f + 1;
    minn = [-5.0,-10.0,   0.0,-10.0,-5.0,-10.0,   0.0,-10.0,-5.0,-10.0];
    maxx = [ 5.0, 10.0,2*np.pi,10.0, 5.0, 10.0,2*np.pi,10.0, 5.0, 10.0];
    
    X = np.linspace(minn[0],maxx[0],Nx);
    Y = np.linspace(minn[2],maxx[2],Nx);
    Z = np.linspace(minn[4],maxx[4],Nx);
    X_,Y_,Z_ = np.meshgrid(X, Y, Z);    
    X,Y = np.meshgrid(X, Y);
    XX = np.reshape(X,[-1,1]);
    YY = np.reshape(Y,[-1,1]);
    XX_ = np.reshape(X_,[-1,1]);
    YY_ = np.reshape(Y_,[-1,1]);
    ZZ_ = np.reshape(Z_,[-1,1]); 
    grid_check = np.concatenate((XX_,-1.5*np.ones(XX_.shape),np.zeros(XX_.shape),np.zeros(XX_.shape),
                                 YY_,-1.8*np.ones(XX_.shape),np.zeros(XX_.shape),np.zeros(XX_.shape),
                                 ZZ_,1.2*np.zeros(XX_.shape)),axis=1);
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
    dt = 0.05
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

    def F(ALL_x,opt_a,opt_b):
       
       col1 = ALL_x[:,1,None];
       col2 = grav*np.tan(ALL_x[:,2,None]);
       col3 = -d1*ALL_x[:,[2]] + ALL_x[:,[3]];
       col4 = -d0*ALL_x[:,[2]] + n0*opt_a[:,0,None]
       
       col5 = ALL_x[:,5,None];
       col6 = grav*np.cos(ALL_x[:,6,None]);
       col7 = -d1*ALL_x[:,[6]] + ALL_x[:,[7]];
       col8 = -d0*ALL_x[:,[6]] + n0*opt_a[:,1,None]
       
       col9 = ALL_x[:,9,None]
       col10 = kT*opt_a[:,2,None]
       
       return np.concatenate((col1,col2,col3,col4,col5,col6,col7,col8,col9,col10),axis=1);

  #     Dynamics of the 10D Quadrotor
  #         x_dot  \dot x_1 = x_2
  #         vx_dot \dot x_2 = g * tan(x_3)
  #         tx_dot \dot x_3 = -d1 * x_3 + x_4
  #         wx_dot \dot x_4 = -d0 * x_3 + n0 * u1
  #         y_dot  \dot x_5 = x_6
  #         vy_dot \dot x_6 = g * tan(x_7)
  #         ty_dot \dot x_7 = -d1 * x_7 + x_8
  #         wy_dot \dot x_8 = -d0 * x_7 + n0 * u2
  #         z_dot  \dot x_9 = x_10
  #         vz_dot \dot x_10 = kT * u3 - g
  #              uMin <= [u1; u2; u3] <= uMax
  #              dMin <= [d1; d2; d3] <= dMax 

    ####################### RECURSIVE FUNC ####################

    def RK4(ALL_x,dtt,opt_a,opt_b):

        k1 = F(ALL_x,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k2)
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k1);
        ALL_tmp[:,[2,6]] = p_corr(ALL_tmp[:,[2,6]]);

        k2 = F(ALL_tmp,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k3)
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k2);
        ALL_tmp[:,[2,6]] = p_corr(ALL_tmp[:,[2,6]]);

        k3 = F(ALL_tmp,opt_a,opt_b);  #### !!!
        # ~~~~ Compute optimal input (k4)
        ALL_tmp = ALL_x + np.multiply(dtt,k3);
        ALL_tmp[:,[2,6]] = p_corr(ALL_tmp[:,[2,6]]);

        k4 = F(ALL_tmp,opt_a,opt_b);  #### !!!

        Snx = ALL_x + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4)); #np.multiply(dtt,k1)
        Snx[:,[2,6]] = p_corr(Snx[:,[2,6]]);
        return Snx;

    perms = list(itertools.product([-1,1], repeat=num_ac))
    true_ac_list = [];
    for i in range(len(perms)): #2**num_actions
        ac_tuple = perms[i];
        ac_list = [(tmp1==1)*tmp3 +  (tmp1==-1)*tmp2 for tmp1,tmp2,tmp3 in zip(ac_tuple,min_list,max_list)]; #ASSUMING: aMax = -aMin
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
        
        for params in F_PI:
            for ind in range(len(params)): #Reload pi*(x,t+dt) parameters
                sess.run(theta[ind].assign(params[ind]));

            hots = sess.run(Tt,{states:ConvCosSin(next_states)});
            opt_a = Hot_to_Cold(hots,true_ac_list)            
            for _ in range(subSamples):
                next_states = RK4(next_states,dt/float(subSamples),opt_a,None);
        
        values_ = V_0(next_states[:,[0,4,8]]);
        compare_vals_ = values_.reshape([-1,ALL_x.shape[0]]).T;
        index_best_a_ = compare_vals_.argmin(axis=1)                    #Changed to ARGMAX
        values_ = np.min(compare_vals_,axis=1,keepdims=True);
        
        for ind in range(len(current_params)): #Reload pi*(x,t+dt) parameters
            sess.run(theta[ind].assign(current_params[ind]));
        
        return sess.run(make_hot,{hot_input:index_best_a_}),values_

    def getTraj(ALL_x,F_PI=[],subSamples=1,Noise = False):

        current_params = sess.run(theta);
        
        next_states = ALL_x;
        traj = [next_states];
        actions = [];
              
        for params in F_PI:
            for ind in range(len(params)): #Reload pi*(x,t+dt) parameters
                sess.run(theta[ind].assign(params[ind]));
            
            hots = sess.run(Tt,{states:ConvCosSin(next_states)});
            opt_a = Hot_to_Cold(hots,true_ac_list)
            for _ in range(subSamples):
                next_states = RK4(next_states,dt/float(subSamples),opt_a,None);
                if Noise:
                    next_states = next_states + np.random.normal(size=next_states.shape)*0.01
                traj.append(next_states);
                actions.append(hots.argmax(axis=1)[0]);
                #values = np.min((values,V_0(next_states[:,[0,1]])),axis=0);    

        for ind in range(len(current_params)): #Reload pi*(x,t+dt) parameters
            sess.run(theta[ind].assign(current_params[ind]));
                        
        return traj,V_0(next_states[:,[0,4,8]]),actions; 

    def ConvCosSin(ALL_x):
        sin_phi = np.sin(ALL_x[:,[2,6]])
        cos_phi = np.cos(ALL_x[:,[2,6]])
        pos = ALL_x[:,[0,4,8]]/5.0;
        vel = ALL_x[:,[1,5,9]]/10.0;
        arate = ALL_x[:,[3,7]]/30.0;
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
    
    if(imp == 1.0):
        ALL_PI = pickle.load( open( "policies6D_h20_h20.pkl", "rb" ) );
    while (imp == 1.0):
        state_get = input('State: ');
        sub_smpl = input('SUBSAMPLING: ');
        pause_len = input('Pause: ')
        traj,VAL,act = getTraj(state_get,F_PI=ALL_PI,subSamples=sub_smpl,Noise=False);
        act.append(act[-1]);
        all_to = np.concatenate(traj);
        plt.scatter(all_to[:,[0]],all_to[:,[2]],c=act)
        #plt.colorbar()
        plt.pause(pause_len)
        print(str(VAL));
#        print(str(traj));
    
    for i in xrange(iters):
        
        if(np.mod(i,renew) == 0 and i is not 0):       
            
            ALL_PI.insert(0,sess.run(theta))            
            
            fig = plt.figure(1)
            plt.clf();
            _,nn_vals,_ = getTraj(grid_check,ALL_PI,20)
            fi = (np.abs(nn_vals) < 0.05)
            mini_reach_ = grid_check[fi[:,0]]
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(mini_reach_[:,0], mini_reach_[:,4], mini_reach_[:,8]);            
            plt.pause(0.25);            

#            plt.figure(2) #TODO: Figure out why facing up vs facing down has same action...
#            plt.clf();
#            ALL_xx = np.array([[0.0,0.0,1.0,0.0,0.0,0.0],
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

            t = time.time()
            ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]-2));
            ALL_x[:,[1,5,9]] = ALL_x[:,[1,5,9]]*2.0;
            ALL_x[:,[2,6]] = ALL_x[:,[2,6]]*np.pi/5.0 + np.pi;
            ALL_x[:,[3,7]] = ALL_x[:,[3,7]]*6.0;          
            PI,_ = getPI(ALL_x,F_PI=ALL_PI,subSamples=3);
            pre_ALL_x = ConvCosSin(ALL_x);
            elapsed = time.time() - t
            print("Compute Data Time = "+str(elapsed))
            
            ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls/100,layers[0]-2));
            ALL_x_[:,[1,5,9]] = ALL_x_[:,[1,5,9]]*2.0;
            ALL_x_[:,[2,6]] = ALL_x_[:,[2,6]]*np.pi/5.0 + np.pi;
            ALL_x_[:,[3,7]] = ALL_x_[:,[3,7]]*6.0;    
            PI_,_ = getPI(ALL_x_,F_PI=ALL_PI,subSamples=3);
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
            ALL_x = np.random.uniform(-5.0,5.0,(nrolls,layers[0]-2));
            ALL_x[:,[1,5,9]] = ALL_x[:,[1,5,9]]*2.0;
            ALL_x[:,[2,6]] = ALL_x[:,[2,6]]*np.pi/5.0 + np.pi;
            ALL_x[:,[3,7]] = ALL_x[:,[3,7]]*6.0;          
            PI,_ = getPI(ALL_x,F_PI=[],subSamples=3);
            pre_ALL_x = ConvCosSin(ALL_x);
            elapsed = time.time() - t
            print("Compute Data Time = "+str(elapsed))
            
            ALL_x_ = np.random.uniform(-5.0,5.0,(nrolls/100,layers[0]-2));
            ALL_x_[:,[1,5,9]] = ALL_x_[:,[1,5,9]]*2.0;
            ALL_x_[:,[2,6]] = ALL_x_[:,[2,6]]*np.pi/5.0 + np.pi;
            ALL_x_[:,[3,7]] = ALL_x_[:,[3,7]]*6.0;    
            PI_,_ = getPI(ALL_x_,F_PI=[],subSamples=3);
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
            
        nunu = 0.001#/(np.sqrt(np.mod(i,renew))+1.0)#lr_schedule.value(i);
        #nunu = ler_r/(np.mod(i,renew)+1.0);
        tmp = np.random.randint(len(ALL_x), size=bts);
        sess.run(train_step, feed_dict={states:pre_ALL_x[tmp],y:PI[tmp],nu:nunu});
        #tmp = np.random.randint(len(reach100s), size=bts);
        #sess.run(train_step, feed_dict={states:reach100s[tmp,:-1],y:reach100s[tmp,-1,None],nu:nunu});

    pickle.dump(ALL_PI,open( "policies6D_L2_h20_h20.pkl", "wb" ));
#    while True:
#        state_get = input('State: ');
#        if(state_get == 0):
#            break;
#        _,VAL = getPI(state_get,ALL_PI);
#        print(str(VAL));

num_ac = 3;
layers1 = [12,20,20,2**num_ac];
t_hor = -0.5;

main(layers1,t_hor,0,2000000,50000,0.001,0.95,99,5000,0.0,0,num_ac);
