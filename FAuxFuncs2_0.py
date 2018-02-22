# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:13:20 2016

@author: cusgadmin
"""

import tensorflow as tf

def TransDef(scope=None, reuse=None, lsizes = None, depth=None, incl=None, center=None, outp=False):
    with tf.variable_scope(scope, reuse=reuse):
        states = tf.placeholder(tf.float32,shape=(None,lsizes[0]),name="states");
        s_vec = tf.slice(states, [0,0], [-1,2]);
        y = tf.placeholder(tf.float32,shape=(None,1),name="y");   
    
        lw = [];
        lb = [];
        l = [];
        reg = 0.0;
        for i in xrange(len(lsizes) - 1):
            lw.append(0.1*tf.Variable(tf.random_uniform([lsizes[i],lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="H"+str(i)));
            lb.append(0.1*tf.Variable(tf.random_uniform([1,lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="B"+str(i)));
            reg = reg + tf.reduce_sum(tf.abs(lw[-1])) + tf.reduce_sum(tf.abs(lb[-1]));
            
        l.append(tf.nn.sigmoid(tf.add(tf.matmul(states,lw[0]), lb[0]),name="A"+str(0)))
        for i in xrange(len(lw)-2):
            l.append(tf.nn.sigmoid(tf.add(tf.matmul(l[-1],lw[i+1]), lb[i+1]),name="A"+str(i)));
        
        if(outp==True): 
            l.append(tf.nn.tanh(tf.add(tf.matmul(l[-1],lw[-1]), lb[-1],name="A_end")));
            
            #V_x0 = incl*tf.sqrt(tf.reduce_sum(tf.square(s_vec),1,keep_dims=True)) - depth;
            V = l[-1];
            #V = tf.reduce_max(tf.abs(s_vec + tf.mul(tf.tanh(t_vec),l[-1])),1,keep_dims=True) - depth;
            
        else: 
            l.append(tf.add(tf.matmul(l[-1],lw[-1]), lb[-1],name="A_end"));
         
            #V_x0 = incl*tf.sqrt(tf.reduce_sum(tf.square(s_vec),1,keep_dims=True)) - depth;
            V_x0 = tf.reduce_max(tf.abs(s_vec),1,keep_dims=True) - depth;
            V = V_x0 + l[-1];
            #V = l[-1]
            #V = tf.reduce_max(tf.abs(s_vec + tf.mul(tf.tanh(t_vec),l[-1])),1,keep_dims=True) - depth;
    
    return states,y,V,l,lb,reg
