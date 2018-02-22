# -*- coding: utf-8 -*-
"""

Created on Thu May 26 14:13:20 2016

@author: Vicenc Rubies Royo
"""

import tensorflow as tf
import numpy as np

def lrelu(x):
  return tf.nn.relu(x) - 0.01*tf.nn.relu(-x)

def getPI(sess,ALL_x,extra_args,F_PI=[], F_PI_=[], subSamples=1): #Things to keep in MIND: You want the returned value to be the minimum accross a trajectory.

    perms = extra_args["perms"]
    perms_ = extra_args["perms_"]
    true_ac_list

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

def MakeFNN(scope=None, reuse=None, lsizes = None):
    with tf.variable_scope(scope, reuse=reuse):
        states = tf.placeholder(tf.float32,shape=(None,lsizes[0]),name="states");
        y = tf.placeholder(tf.float32,shape=(None,lsizes[-1]),name="y");   
    
        lw = [];
        lb = [];
        l = [];
        reg = 0.0;
        for i in xrange(len(lsizes) - 1):
            lw.append(0.1*tf.Variable(tf.random_uniform([lsizes[i],lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="H"+str(i)));
            lb.append(0.1*tf.Variable(tf.random_uniform([1,lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="B"+str(i)));
            reg = reg + tf.reduce_sum(tf.abs(lw[-1])) + tf.reduce_sum(tf.abs(lb[-1]));
            
        l.append(lrelu(tf.add(tf.matmul(states,lw[0]), lb[0])))
        for i in xrange(len(lw)-2):
            l.append(lrelu(tf.add(tf.matmul(l[-1],lw[i+1]), lb[i+1])));
        last_ba = tf.add(tf.matmul(l[-1],lw[-1]), lb[-1],name="A_end");
       
        l.append(tf.nn.softmax(last_ba));
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=last_ba,labels=y)
        L = tf.reduce_mean(cross_entropy)
        
        PI = l[-1];
        
    return states,y,PI,L,l,lb,reg,cross_entropy
