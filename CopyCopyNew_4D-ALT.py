# -*-: coding: utf-8 -*-
"""
In this script you implemented a rechability algorithm which: (MASTER)
- Takes in a  NN which computes a policy directly.( X -> NN -> softmax(output) which selects the actions. 
- The NN parameters are stored and used for subsequent training.
- This script is the  child of CopyCopyNew.py

Created on Thu May 26 13:37:12 2016

@author: vrubies

"""

import numpy as np
import tensorflow as tf
import itertools
from FAuxFuncs4_0 import TransDef
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
from mpl_toolkits.mplot3d import Axes3D
from dqn_utils import PiecewiseSchedule,LinearSchedule,linear_interpolation
import time
import h5py
            

def main(layers,t_hor,ind,nrolls,bts,ler_r,mom,teps,renew,imp,q):
# Quad Params
    aMax = 3.0; 
    aMin = -1.0*aMax;
    wMax = 2*np.pi/10.0;
    wMin = -1.0*wMax; 
    max_list = [wMax,aMax];


    print 'Starting worker-' + str(ind)

    f = 1;
    Nx = 100*f + 1;
    minn = [-5.0,-5.0,0.0,6.0];
    maxx = [ 5.0, 5.0,2*np.pi,12.0];
    
    X = np.linspace(minn[0],maxx[0],Nx);
    Y = np.linspace(minn[1],maxx[1],Nx);
    Z = np.linspace(minn[2],maxx[2],Nx);
    X_,Y_,Z_ = np.meshgrid(X, Y, Z);    
    X,Y = np.meshgrid(X, Y);
    XX = np.reshape(X,[-1,1]);
    YY = np.reshape(Y,[-1,1]);
    XX_ = np.reshape(X_,[-1,1]);
    YY_ = np.reshape(Y_,[-1,1]);
    ZZ_ = np.reshape(Z_,[-1,1]); grid_check = np.concatenate((XX_,YY_,ZZ_,6.0*np.ones(XX_.shape)),axis=1);
    grid_eval = np.concatenate((XX,YY,0.0*np.ones(XX.shape),6.0*np.ones(XX.shape)),axis=1);
    grid_eval_ = np.concatenate((XX,YY,(2.0/3.0)*np.pi*np.ones(XX.shape),6.0*np.ones(XX.shape)),axis=1);
    grid_eval__ = np.concatenate((XX,YY,(4.0/3.0)*np.pi*np.ones(XX.shape),6.0*np.ones(XX.shape)),axis=1);
    grid_evall = np.concatenate((XX,YY,0.0*np.ones(XX.shape),12.0*np.ones(XX.shape)),axis=1);
    grid_evall_ = np.concatenate((XX,YY,(2.0/3.0)*np.pi*np.ones(XX.shape),12.0*np.ones(XX.shape)),axis=1);
    grid_evall__ = np.concatenate((XX,YY,(4.0/3.0)*np.pi*np.ones(XX.shape),12.0*np.ones(XX.shape)),axis=1);    

    reach100s = sio.loadmat('flat_1s.mat'); reach100s = reach100s["M"]; 
    reach100s[:,[1,2]] = reach100s[:,[2,1]];
    reach100s[:,2] = np.mod(reach100s[:,2],2.0*np.pi);
    
    fi = (np.abs(reach100s[:,-1]) < 0.05)
    mini_reach = reach100s[fi]
    fi = (mini_reach[:,3] == 6.0)
    mini_reach = mini_reach[fi]
    #mean_data = np.mean(reach100s[:,:-1],axis=0);
    #std_data = np.std(reach100s[:,:-1],axis=0);

    nofparams = 0;
    for i in xrange(len(layers)-1):
        nofparams += layers[i]*layers[i+1] + layers[i+1];
    print 'Number of Params is: ' + str(nofparams)
    
    H_length = t_hor;#-1.0; #Has to be negative                                 #VA                                                           #VAR
    #center = np.array([[0.0,0.0]])
    center = np.array([[0.0,0.0,0.0,0.0,0.0,0.0]])
    depth = 2.0;
    incl = 1.0;

    ##################### DEFINITIONS #####################
    #layers = [2 + 1,10,1];                                                    #VAR
    #ssize = layers[0] - 1;
    dt = 0.05;                                                                 #VAR
    num_ac = 2;
    iters = int(np.abs(t_hor)/dt)*renew + 1; 
    ##################### INSTANTIATIONS #################
    nu = tf.placeholder(tf.float32, shape=[])                                         #VAR
    states,y,Tt,L,l_r,lb,cross_entropy,train_step = TransDef("Critic",False,layers,num_ac,t_hor,dt,nu,mom);
    ola1 = tf.argmax(Tt,dimension=1)
    ola2 = tf.argmax(y,dimension=1)
    ola3 = tf.equal(ola1,ola2)
    accuracy = tf.reduce_mean(tf.cast(ola3, tf.float32));
    #a_layers = layers;
    #a_layers[-1] = 2; #We have two actions
    #states_,y_,Tt_,l_r_,lb_,reg_ = TransDef("Actor",False,a_layers,depth,incl,center,outp=True);
    
    V_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Critic');
    #A_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Actor');


    set_to_zero = []
    for var  in sorted(V_func_vars,        key=lambda v: v.name):
        set_to_zero.append(var.assign(tf.zeros(tf.shape(var))))
    set_to_zero = tf.group(*set_to_zero)
    
    set_to_not_zero = []
    for var  in sorted(V_func_vars,        key=lambda v: v.name):
        set_to_not_zero.append(var.assign(tf.random_uniform(tf.shape(var),minval=-0.1,maxval=0.1)));
    set_to_not_zero = tf.group(*set_to_not_zero)    



    hot_input = tf.placeholder(tf.int64,shape=(None));   
    make_hot = tf.one_hot(hot_input, 4, on_value=1, off_value=0)

    # INITIALIZE GRAPH
    theta = tf.trainable_variables();
    sess = tf.Session();
    init = tf.initialize_all_variables();
    sess.run(init);

    def V_0(x):
        return np.linalg.norm(x,ord=np.inf,axis=1,keepdims=True) - 2.0
        #return np.linalg.norm(x,axis=1,keepdims=True) - 2.0

    def p_corr(ALL_x):
        ALL_x = np.mod(ALL_x,2.0*np.pi);
        return ALL_x;

    def F(ALL_x,opt_a,opt_b):
       sin_phi = np.around(np.sin(ALL_x[:,2,None]),5); #<+++++ ROUNDOFF ERROR!!!
       cos_phi = np.around(np.cos(ALL_x[:,2,None]),5);

       col1 = np.multiply(ALL_x[:,3,None],cos_phi);
       col2 = np.multiply(ALL_x[:,3,None],sin_phi);
       col3 = opt_a[:,0,None];
       col4 = opt_a[:,1,None];

       return np.concatenate((col1,col2,col3,col4),axis=1);


    ####################### RECURSIVE FUNC ####################

    def RK4(ALL_x,dtt,opt_a,opt_b):

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

        Snx = ALL_x + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4)); #np.multiply(dtt,k1)
        Snx[:,2] = p_corr(Snx[:,2]);
        return Snx;

    perms = list(itertools.product([-1,1], repeat=num_ac))
    
    def Hot_to_Cold(hots,ac_list):
        a = hots.argmax(axis=1);
        a = np.asarray([ac_list[i] for i in a]);
        return a;
    
    def Normalize(ALL_x):
        ALL_x[:,[0,1]] = ALL_x[:,[0,1]]/5.0;
        ALL_x[:,[2]] = (ALL_x[:,[2]] - np.pi)/np.pi
        ALL_x[:,[3]] = (ALL_x[:,[3]] - 9.0)/3.0
        return ALL_x
    
    def getPI(ALL_x,F_PI=[],ret_traj=False,subSamples=1): #Things to keep in MIND: You want the returned value to be the minimum accross a trajectory.

        current_params = sess.run(theta);
        
        if(ret_traj): traj = [];
        #perms = list(itertools.product([-1,1], repeat=num_ac))
        next_states = [];
        true_ac_list = [];
        for i in range(len(perms)): #2**num_actions
            ac_tuple = perms[i];
            ac_list = [np.float(tmp1)*tmp2 for tmp1,tmp2 in zip(ac_tuple,max_list)]; #ASSUMING: aMax = -aMin
            true_ac_list.append(ac_list);
            opt_a = np.asarray(ac_list)*np.ones([ALL_x.shape[0],1]);
            Snx = ALL_x;
            for _ in range(subSamples): Snx = RK4(Snx,dt/float(subSamples),opt_a,None);
            next_states.append(Snx);
        next_states = np.concatenate(next_states,axis=0);
        if(ret_traj): traj.append(next_states);
        values = V_0(next_states[:,[0,1]]);
        
        flag = True;
        for params in F_PI:
            for ind in range(len(params)): #Reload pi*(x,t+dt) parameters
                sess.run(theta[ind].assign(params[ind]));
            
            if(flag):
                hots_ = np.zeros((len(next_states),2**num_ac));
                flag = False;
            else:
                hots_ = np.zeros((len(next_states),2**num_ac));
                
            for _ in range(subSamples):
                hots = sess.run(Tt,{states:ConvCosSin(next_states),actns:hots_});
                opt_a = Hot_to_Cold(hots,true_ac_list)
                next_states = RK4(next_states,dt/float(subSamples),opt_a,None);
                if(ret_traj): traj.append(next_states);
                values = np.min((values,V_0(next_states[:,[0,1]])),axis=0);      
            
        compare_vals = values.reshape([-1,ALL_x.shape[0]]).T
        index_best_a = compare_vals.argmin(axis=1)#.reshape([-1,1]);    #Changed to ARGMAX
        values = np.min(compare_vals,axis=1,keepdims=True);             #Changed to MAX
        final_values = np.min((values,V_0(ALL_x[:,[0,1]])),axis=0)
        
        values_ = V_0(next_states[:,[0,1]]); 
        compare_vals_ = values_.reshape([-1,ALL_x.shape[0]]).T;
        values_ = np.min(compare_vals_,axis=1,keepdims=True);
        
        for ind in range(len(current_params)): #Reload pi*(x,t+dt) parameters
            sess.run(theta[ind].assign(current_params[ind]));         
            
        #return index_best_a,final_values
        if(ret_traj):
            return sess.run(make_hot,{hot_input:index_best_a}),values_,traj
        
        return sess.run(make_hot,{hot_input:index_best_a}),values_#final_values 


    def ConvCosSin(ALL_x):
        sin_phi = np.sin(ALL_x[:,2,None])
        cos_phi = np.cos(ALL_x[:,2,None])
        insertion = np.concatenate((sin_phi,cos_phi),axis=1)
        ret_val = np.insert(ALL_x[:,[0,1,3]],2,insertion.T,axis=1)
        return ret_val
    # *****************************************************************************
    #
    # ============================= MAIN LOOP ====================================
    #                     ( )    
    # *****************************************************************************
    t1 = time.time();
    t = 0.0;
    mse = np.inf;
    k=0; kk = 0; beta=3.0; batch_size = bts; tau = 1000.0; steps = teps;
    ALL_PI = [];
    nunu = lr_schedule.value(k);
    
    if(imp == 1.0):
        ALL_PI = pickle.load( open( "policies8.pkl", "rb" ) );
    while (imp == 1.0):
        state_get = input('State: ');
        sub_smpl = input('SUBSAMPLING: ');
        _,VAL,traj = getPI(state_get,ALL_PI,True,sub_smpl);
        print(str(VAL));
#        print(str(traj));
    
    for i in xrange(iters):
        
        if(np.mod(i,renew) == 0 and i is not 0):       
            
#            plt.figure(1)
#            plt.clf();
#            letsee = sess.run(ola3,{states:pre_ALL_x,y:PI})
#            letsee = np.array([not(i) for i in letsee])
#            ALL_xx = pre_ALL_x[letsee]
#            mhm = ALL_x[letsee]
#            cc = (ALL_xx[:,-1]-6.0)/6.0
#            #plt.scatter(ALL_xx[:,0],ALL_xx[:,1],c=cc) 
#            plt.scatter(mhm[:,2],ALL_xx[:,0],c=cc) 
#            plt.pause(0.01);
            
            fig = plt.figure(1)
            plt.clf();
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(mini_reach[:,0], mini_reach[:,1], mini_reach[:,2]);
            _,nn_vals = getPI(grid_check,ALL_PI)
            fi = (np.abs(nn_vals) < 0.05)
            mini_reach_ = grid_check[fi[:,0]]
            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(mini_reach_[:,0], mini_reach_[:,1], mini_reach_[:,2]);            
            plt.pause(0.01);            
            
            plt.figure(3)
            d = 0.1
            plt.clf();
            plt.title(str([str(i)+" : "+str(perms[i]) for i in range(len(perms))]))
            ALL_xp = np.random.uniform(-6.0,6.0,(nrolls/100,layers[0]-1));
            plt.subplot(2,3,1) #SUBPLOT
            ALL_xp[:,2] = 0.0 + d#np.pi/2.0;
            ALL_xp[:,3] = 9.0;
            letsee_ = sess.run(Tt,{states:ConvCosSin(ALL_xp)});
            letsee_ = letsee_.argmax(axis=1);
            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
            plt.colorbar()
            plt.subplot(2,3,2) #SUBPLOT
            ALL_xp[:,2] = np.pi/2 + d;#2*np.pi/3.0;
            ALL_xp[:,3] = 9.0;
            letsee_ = sess.run(Tt,{states:ConvCosSin(ALL_xp)});
            letsee_ = letsee_.argmax(axis=1);
            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
            plt.colorbar()
            plt.subplot(2,3,3) #SUBPLOT
            ALL_xp[:,2] = np.pi + d;#4*np.pi/3.0;
            ALL_xp[:,3] = 9.0;
            letsee_ = sess.run(Tt,{states:ConvCosSin(ALL_xp)});
            letsee_ = letsee_.argmax(axis=1);
            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
            plt.colorbar()
            plt.subplot(2,3,4) #SUBPLOT
            ALL_xp[:,2] = 0.0 - d#np.pi/2.0;
            ALL_xp[:,3] = 9.0;
            letsee_ = sess.run(Tt,{states:ConvCosSin(ALL_xp)});
            letsee_ = letsee_.argmax(axis=1);
            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
            plt.colorbar()
            plt.subplot(2,3,5) #SUBPLOT
            ALL_xp[:,2] = np.pi/2 - d;#2*np.pi/3.0;
            ALL_xp[:,3] = 9.0;
            letsee_ = sess.run(Tt,{states:ConvCosSin(ALL_xp)});
            letsee_ = letsee_.argmax(axis=1);
            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
            plt.colorbar()
            plt.subplot(2,3,6) #SUBPLOT
            ALL_xp[:,2] = np.pi-d#4*np.pi/3.0;
            ALL_xp[:,3] = 9.0;
            letsee_ = sess.run(Tt,{states:ConvCosSin(ALL_xp)});
            letsee_ = letsee_.argmax(axis=1);
            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
            plt.colorbar()            
            plt.pause(0.01);            
            
            ALL_PI.insert(0,sess.run(theta))            
            
            k = 0;
            ALL_x = np.random.uniform(-6.0,6.0,(nrolls,layers[0]-1));
            ALL_x[:,2] = ALL_x[:,2]*np.pi/6.0 + np.pi;
            ALL_x[:,3] = ALL_x[:,3]*3.0/6.0 + 9.0;
            PI,_ = getPI(ALL_x,ALL_PI);
            pre_ALL_x = ConvCosSin(ALL_x);
            
            ALL_x_ = np.random.uniform(-6.0,6.0,(nrolls/100,layers[0]-1));
            ALL_x_[:,2] = ALL_x_[:,2]*np.pi/6.0 + np.pi;
            ALL_x_[:,3] = ALL_x_[:,3]*3.0/6.0 + 9.0;
            PI_,_ = getPI(ALL_x_,ALL_PI);
            pre_ALL_x_ = ConvCosSin(ALL_x_);

            tmp = np.random.randint(len(reach100s[:,:-1]), size=12000);
            _,ZR = getPI(reach100s[tmp,:-1],ALL_PI)
            #ZR = sess.run(Tt,{states:reach100s[:,:-1]});
            error1 = ZR - reach100s[tmp,-1,None];
            
           
            plt.figure(2)
            _,Z000 = getPI(grid_eval,ALL_PI);
            _,Z001 = getPI(grid_eval_,ALL_PI);
            _,Z002 = getPI(grid_eval__,ALL_PI);            
            Z000 = np.reshape(Z000,X.shape);
            Z001 = np.reshape(Z001,X.shape);
            Z002 = np.reshape(Z002,X.shape);
            #filter_in = (Z000 <= 0.05) #& (Z000 >= 0.05);
            filter_out = (Z000 > 0.00) #| (Z000 < -0.05);       
            filter_out_ = (Z001 > 0.00) #| (Z000 < -0.05);       
            filter_out__ = (Z002 > 0.00) #| (Z000 < -0.05);       
            #Z000[filter_in] = 1.0;
            Z000[filter_out] = 0.0;
            Z001[filter_out_] = 0.0;
            Z002[filter_out__] = 0.0;

            _,Z000l = getPI(grid_evall,ALL_PI);
            _,Z001l = getPI(grid_evall_,ALL_PI);
            _,Z002l = getPI(grid_evall__,ALL_PI);             
            Z000l = np.reshape(Z000l,X.shape);
            Z001l = np.reshape(Z001l,X.shape);
            Z002l = np.reshape(Z002l,X.shape);
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
            plt.plot([30*f, 30*f], [30*f, 70*f], 'r-', lw=1)
            plt.plot([30*f, 70*f], [70*f, 70*f], 'r-', lw=1)
            plt.plot([70*f, 70*f], [70*f, 30*f], 'r-', lw=1)
            plt.plot([70*f, 30*f], [30*f, 30*f], 'r-', lw=1)
            plt.subplot(2,3,2)
            plt.imshow(Z001,cmap='gray');
            plt.plot([30*f, 30*f], [30*f, 70*f], 'r-', lw=1)
            plt.plot([30*f, 70*f], [70*f, 70*f], 'r-', lw=1)
            plt.plot([70*f, 70*f], [70*f, 30*f], 'r-', lw=1)
            plt.plot([70*f, 30*f], [30*f, 30*f], 'r-', lw=1)
            plt.subplot(2,3,3)
            plt.imshow(Z002,cmap='gray');
            plt.plot([30*f, 30*f], [30*f, 70*f], 'r-', lw=1)
            plt.plot([30*f, 70*f], [70*f, 70*f], 'r-', lw=1)
            plt.plot([70*f, 70*f], [70*f, 30*f], 'r-', lw=1)
            plt.plot([70*f, 30*f], [30*f, 30*f], 'r-', lw=1)
            plt.subplot(2,3,4)
            plt.imshow(Z000l,cmap='gray');
            plt.plot([30*f, 30*f], [30*f, 70*f], 'r-', lw=1)
            plt.plot([30*f, 70*f], [70*f, 70*f], 'r-', lw=1)
            plt.plot([70*f, 70*f], [70*f, 30*f], 'r-', lw=1)
            plt.plot([70*f, 30*f], [30*f, 30*f], 'r-', lw=1)
            plt.subplot(2,3,5)
            plt.imshow(Z001l,cmap='gray');
            plt.plot([30*f, 30*f], [30*f, 70*f], 'r-', lw=1)
            plt.plot([30*f, 70*f], [70*f, 70*f], 'r-', lw=1)
            plt.plot([70*f, 70*f], [70*f, 30*f], 'r-', lw=1)
            plt.plot([70*f, 30*f], [30*f, 30*f], 'r-', lw=1)
            plt.subplot(2,3,6)
            plt.imshow(Z002l,cmap='gray'); 
            plt.plot([30*f, 30*f], [30*f, 70*f], 'r-', lw=1)
            plt.plot([30*f, 70*f], [70*f, 70*f], 'r-', lw=1)
            plt.plot([70*f, 70*f], [70*f, 30*f], 'r-', lw=1)
            plt.plot([70*f, 30*f], [30*f, 30*f], 'r-', lw=1)
            plt.pause(0.01);

            t = t - dt;  
            #sess.run(set_to_not_zero);
            print str(t) + " || " + str(np.max(np.abs(error1))) + " , " + str(np.mean(np.abs(error1))) + "|ITR=" + str(i)                                                #VAR         
            
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
            
            k = 0;
#            sess.run(set_to_zero);
            ALL_x = np.random.uniform(-6.0,6.0,(nrolls,layers[0]-1));
            ALL_x[:,2] = ALL_x[:,2]*np.pi/6.0 + np.pi;
            ALL_x[:,3] = ALL_x[:,3]*3.0/6.0 + 9.0;
            PI,_ = getPI(ALL_x); 
            pre_ALL_x = ConvCosSin(ALL_x);
            
#            load = sio.loadmat('opt_a_0.5.mat');
#            load = load['data_s']
#            ALL_x = load[:,:-1]
#            ALL_x[:,2] = ALL_x[:,2] + np.pi;
#            pre_ALL_x = ConvCosSin(ALL_x);
#            acts = load[:,-1,None]
#            PI = sess.run(make_hot,{hot_input:acts[:,0]})
            
            ALL_x_ = np.random.uniform(-6.0,6.0,(nrolls/100,layers[0]-1));
            ALL_x_[:,2] = ALL_x_[:,2]*np.pi/6.0 + np.pi;
            ALL_x_[:,3] = ALL_x_[:,3]*3.0/6.0 + 9.0;
            PI_,_ = getPI(ALL_x_);
            pre_ALL_x_ = ConvCosSin(ALL_x_);            
#            sess.run(set_to_not_zero);
            
            plt.figure(4)
            #plt.clf();
            plt.title(str([str(i)+" : "+str(perms[i]) for i in range(len(perms))]))
            b_sele = (np.abs(ALL_x[:,-1]-7.5) < 0.1); 
            ALL_xp = ALL_x[b_sele]; 
            letsee_ = PI[b_sele];
            b_sele = (ALL_xp[:,2] < 0.1);
            ALL_xp = ALL_xp[b_sele]; 
            letsee_ = letsee_[b_sele];  
            #plt.subplot(2,3,1) #SUBPLOT
            letsee_ = letsee_.argmax(axis=1);
            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
            plt.colorbar()
#            plt.subplot(2,3,2) #SUBPLOT
#            ALL_xp[:,2] = np.pi/2.0;
#            ALL_xp[:,3] = 6.0;
#            letsee_ = sess.run(Tt,{states:ALL_xp});
#            letsee_ = letsee_.argmax(axis=1);
#            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
#            plt.colorbar()
#            plt.subplot(2,3,3) #SUBPLOT
#            ALL_xp[:,2] = np.pi;
#            ALL_xp[:,3] = 6.0;
#            letsee_ = sess.run(Tt,{states:ALL_xp});
#            letsee_ = letsee_.argmax(axis=1);
#            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
#            plt.colorbar()
#            plt.subplot(2,3,4) #SUBPLOT
#            ALL_xp[:,2] = 0.0#np.pi/2.0;
#            ALL_xp[:,3] = 12.0;
#            letsee_ = sess.run(Tt,{states:ALL_xp});
#            letsee_ = letsee_.argmax(axis=1);
#            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
#            plt.colorbar()
#            plt.subplot(2,3,5) #SUBPLOT
#            ALL_xp[:,2] = np.pi/2.0;
#            ALL_xp[:,3] = 12.0;
#            letsee_ = sess.run(Tt,{states:ALL_xp});
#            letsee_ = letsee_.argmax(axis=1);
#            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
#            plt.colorbar()
#            plt.subplot(2,3,6) #SUBPLOT
#            ALL_xp[:,2] = np.pi;
#            ALL_xp[:,3] = 12.0;
#            letsee_ = sess.run(Tt,{states:ALL_xp});
#            letsee_ = letsee_.argmax(axis=1);
#            plt.scatter(ALL_xp[:,0],ALL_xp[:,1],c=letsee_)
#            plt.colorbar()            
            plt.pause(0.01); 
            

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

    pickle.dump(ALL_PI,open( "policies8.pkl", "wb" ));
#    while True:
#        state_get = input('State: ');
#        if(state_get == 0):
#            break;
#        _,VAL = getPI(state_get,ALL_PI);
#        print(str(VAL));

num_ac = 2;
layers1 = [5,20,20,2**num_ac]; #[5,20,20,20,2**num_ac],[5,30,25,2**num_ac],[5,20,20,2**num_ac],[5,50,50,2**num_ac],[5,50,50,2**num_ac]
t_hor = -0.25;

main(layers1,t_hor,0,2000000,50000,0.001,0.95,99,5000,0.0,0);
