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
from FAuxFuncs3_0 import TransDef
import pickle
            

def main(layers,t_hor,num_ac,nrolls,bts,ler_r,mom,renew):
    # Dynamics Parameters
    aMax = 3.0; 
    aMin = -1.0*aMax;
    wMax = 2*np.pi/10.0;
    wMin = -1.0*wMax; 
    max_list = [wMax,aMax];
    min_list = [wMin,aMin];

    dt = 0.05;
    iters = int(np.abs(t_hor)/dt)*renew + 1; 
    
    
    nofparams = 0;
    for i in xrange(len(layers)-1):
        nofparams += layers[i]*layers[i+1] + layers[i+1];
    print 'Number of Params is: ' + str(nofparams)    
    
    ##################### INSTANTIATIONS #################
    states,y,Tt,L,l_r,lb,reg, cross_entropy = TransDef("Policy",False,layers);
    tmp_1 = tf.argmax(Tt,dimension=1)
    tmp_2 = tf.argmax(y,dimension=1)
    tmp_3 = tf.equal(tmp_1,tmp_2)
    accuracy = tf.reduce_mean(tf.cast(tmp_3, tf.float32));
    
    nu = tf.placeholder(tf.float32, shape=[])

    train_step = tf.train.RMSPropOptimizer(learning_rate=nu,momentum=mom).minimize(L);

    hot_input = tf.placeholder(tf.int64,shape=(None));   
    make_hot = tf.one_hot(hot_input, 4, on_value=1, off_value=0)

    # INITIALIZE GRAPH
    theta = tf.trainable_variables();
    sess = tf.Session();
    init = tf.initialize_all_variables();
    sess.run(init);

    # Indicator/Distace function
    def V_0(x):
        return np.linalg.norm(x,ord=np.inf,axis=1,keepdims=True) - 2.0

    #Phi_correction (corrects anges to the correct range)
    def p_corr(ALL_x):
        ALL_x = np.mod(ALL_x,2.0*np.pi);
        return ALL_x;

    #Dynamics
    def F(ALL_x,opt_a,opt_b):
       sin_phi = np.around(np.sin(ALL_x[:,2,None]),5);
       cos_phi = np.around(np.cos(ALL_x[:,2,None]),5);

       col1 = np.multiply(ALL_x[:,3,None],cos_phi);
       col2 = np.multiply(ALL_x[:,3,None],sin_phi);
       col3 = opt_a[:,0,None];
       col4 = opt_a[:,1,None];

       return np.concatenate((col1,col2,col3,col4),axis=1);


    ####################### RECURSIVE FUNC ####################

    def RK4(ALL_x,dtt,opt_a,opt_b):

        k1 = F(ALL_x,opt_a,opt_b);
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k1);
        ALL_tmp[:,2] = p_corr(ALL_tmp[:,2]);

        k2 = F(ALL_tmp,opt_a,opt_b);
        ALL_tmp = ALL_x + np.multiply(dtt/2.0,k2);
        ALL_tmp[:,2] = p_corr(ALL_tmp[:,2]);

        k3 = F(ALL_tmp,opt_a,opt_b);
        ALL_tmp = ALL_x + np.multiply(dtt,k3);
        ALL_tmp[:,2] = p_corr(ALL_tmp[:,2]);

        k4 = F(ALL_tmp,opt_a,opt_b);

        Snx = ALL_x + np.multiply((dtt/6.0),(k1 + 2.0*k2 + 2.0*k3 + k4));
        Snx[:,2] = p_corr(Snx[:,2]);
        return Snx;
     
    #Generate all possible action tuples given the input to the system 
    perms = list(itertools.product([-1,1], repeat=num_ac))
    true_ac_list = [];
    for i in range(len(perms)):
        ac_tuple = perms[i];
        ac_list = [(tmp1==1)*tmp3 +  (tmp1==-1)*tmp2 for tmp1,tmp2,tmp3 in zip(ac_tuple,min_list,max_list)]; 
        true_ac_list.append(ac_list);
    
    #Convert onehot vectors into action vectors
    def Hot_to_Cold(hots,ac_list):
        a = hots.argmax(axis=1);
        a = np.asarray([ac_list[i] for i in a]);
        return a;
    
    # This function gets a set of states ALL_x and computes the optimal action
    # labels for training.
    def getPI(ALL_x,F_PI=[],subSamples=1):

        current_params = sess.run(theta);
        
        next_states = [];
        
        for i in range(len(perms)):
            opt_a = np.asarray(true_ac_list[i])*np.ones([ALL_x.shape[0],1]);
            Snx = ALL_x;
            for _ in range(subSamples): Snx = RK4(Snx,dt/float(subSamples),opt_a,None);
            next_states.append(Snx);
        next_states = np.concatenate(next_states,axis=0);
        
        for params in F_PI:
            for ind in range(len(params)): #Reload pi*(x,t+dt) parameters
                sess.run(theta[ind].assign(params[ind]));

            hots = sess.run(Tt,{states:NormalizeFcn(next_states)});
            opt_a = Hot_to_Cold(hots,true_ac_list)            
            for _ in range(subSamples):
                next_states = RK4(next_states,dt/float(subSamples),opt_a,None);
        
        values_ = V_0(next_states[:,[0,1]]);
        compare_vals_ = values_.reshape([-1,ALL_x.shape[0]]).T;
        index_best_a_ = compare_vals_.argmin(axis=1) #<------- HERE WE CHOOSE THE OPTIMAL ACTION
        values_ = np.min(compare_vals_,axis=1,keepdims=True); #<- Value of the optimal action (how close we are to the origin
        
        for ind in range(len(current_params)):
            sess.run(theta[ind].assign(current_params[ind]));
            
        return sess.run(make_hot,{hot_input:index_best_a_}),values_

    # This function records a trajectory induced by the policy
    def getTraj(ALL_x,F_PI=[],subSamples=1):

        current_params = sess.run(theta);
        
        next_states = ALL_x;
        traj = [next_states];
        actions = [];
              
        for params in F_PI:
            for ind in range(len(params)):
                sess.run(theta[ind].assign(params[ind]));

            hots = sess.run(Tt,{states:NormalizeFcn(next_states)});
            opt_a = Hot_to_Cold(hots,true_ac_list)            
            for _ in range(subSamples):
                next_states = RK4(next_states,dt/float(subSamples),opt_a,None);
                traj.append(next_states);
                actions.append(hots.argmax(axis=1)[0]);   

        for ind in range(len(current_params)):
            sess.run(theta[ind].assign(current_params[ind]));
                        
        return traj,V_0(next_states[:,[0,1]]),actions; 

    # This function normalizes the inputs to the NN
    def NormalizeFcn(ALL_x):
        sin_phi = np.sin(ALL_x[:,2,None])
        cos_phi = np.cos(ALL_x[:,2,None])
        insertion = np.concatenate((sin_phi,cos_phi),axis=1)
        ret_val = np.insert(ALL_x[:,[0,1,3]],2,insertion.T,axis=1)
        return ret_val
    # *****************************************************************************
    #
    # ============================= MAIN LOOP ====================================
    #                       
    # *****************************************************************************
    ALL_PI = []; nunu = 0.001;

    
    for i in xrange(iters):
        
        if(np.mod(i,renew) == 0 and i is not 0):       

            ALL_PI.insert(0,sess.run(theta)) 
            
            ALL_x = np.random.uniform(-6.0,6.0,(nrolls,layers[0]-1));
            ALL_x[:,2] = ALL_x[:,2]*np.pi/6.0 + np.pi;
            ALL_x[:,3] = ALL_x[:,3]*3.0/6.0 + 9.0;
            PI,_ = getPI(ALL_x,ALL_PI);
            pre_ALL_x = NormalizeFcn(ALL_x);
            
            ALL_x_ = np.random.uniform(-6.0,6.0,(nrolls/100,layers[0]-1));
            ALL_x_[:,2] = ALL_x_[:,2]*np.pi/6.0 + np.pi;
            ALL_x_[:,3] = ALL_x_[:,3]*3.0/6.0 + 9.0;
            PI_,_ = getPI(ALL_x_,ALL_PI);
            pre_ALL_x_ = NormalizeFcn(ALL_x_);

        elif(np.mod(i,renew) == 0 and i is 0):         
            
            ALL_x = np.random.uniform(-6.0,6.0,(nrolls,layers[0]-1));
            ALL_x[:,2] = ALL_x[:,2]*np.pi/6.0 + np.pi;
            ALL_x[:,3] = ALL_x[:,3]*3.0/6.0 + 9.0;
            PI,_ = getPI(ALL_x); 
            pre_ALL_x = NormalizeFcn(ALL_x);
            
            ALL_x_ = np.random.uniform(-6.0,6.0,(nrolls/100,layers[0]-1));
            ALL_x_[:,2] = ALL_x_[:,2]*np.pi/6.0 + np.pi;
            ALL_x_[:,3] = ALL_x_[:,3]*3.0/6.0 + 9.0;
            PI_,_ = getPI(ALL_x_);
            pre_ALL_x_ = NormalizeFcn(ALL_x_);            

        # Show accuracy after 200 steps...
        if(np.mod(i,200) == 0):

            train_acc = sess.run(accuracy,{states:pre_ALL_x,y:PI});
            test_acc = sess.run(accuracy,{states:pre_ALL_x_,y:PI_});
            print str(i) + ") | TR_ACC = " + str(train_acc) + " | TE_ACC = " + str(test_acc) + " | Lerning Rate = " + str(nunu)
            
        tmp = np.random.randint(len(ALL_x), size=bts);
        sess.run(train_step, feed_dict={states:pre_ALL_x[tmp],y:PI[tmp],nu:nunu});

    pickle.dump(ALL_PI,open( "policies4D_reach_h20_h20.pkl", "wb" ));

num_ac = 2;
layers1 = [5,20,20,2**num_ac];
t_hor = -0.5;

main(layers1,t_hor,num_ac,2000000,50000,0.001,0.95,5000); #main(layers,t_hor,num_ac,nrolls,bts,ler_r,mom,renew)
