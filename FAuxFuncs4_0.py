# -*- coding: utf-8 -*-
"""
- Helper Function for CopyCopyNew_.py

Created on Thu May 26 14:13:20 2016

@author: cusgadmin
"""

import tensorflow as tf

def TransDef_(scope=None, reuse=None, lsizes = None, depth=None, incl=None, center=None, outp=False):
    with tf.variable_scope(scope, reuse=reuse):
        states = tf.placeholder(tf.float32,shape=(None,lsizes[0]),name="states");
        pos_vec = tf.slice(states, [0,0], [-1,2]); 
        y = tf.placeholder(tf.float32,shape=(None,1),name="y");   
    
        lw = [];
        lb = [];
        l = [];
        reg = 0.0;
        for i in xrange(len(lsizes) - 1):
            lw.append(0.1*tf.Variable(tf.random_uniform([lsizes[i],lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="H"+str(i)));
            lb.append(0.1*tf.Variable(tf.random_uniform([1,lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="B"+str(i)));
            reg = reg + tf.reduce_sum(tf.abs(lw[-1])) + tf.reduce_sum(tf.abs(lb[-1]));
            
        l.append(tf.nn.tanh(tf.add(tf.matmul(states,lw[0]), lb[0]),name="A"+str(0)))
        for i in xrange(len(lw)-2):
            l.append(tf.nn.tanh(tf.add(tf.matmul(l[-1],lw[i+1]), lb[i+1]),name="A"+str(i)));
        
        l.append(tf.add(tf.matmul(l[-1],lw[-1]), lb[-1],name="A_end"))

        V_x0 = tf.reduce_max(tf.abs(pos_vec*5.0),1,keep_dims=True) - 1.0;
        V = V_x0 + l[-1];
        
        L = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y,V)),1,keep_dims=True)))

        
    return states,y,V,L,lw,lb,reg
#
#
## -*- coding: utf-8 -*-
#"""
#- Helper Function for CopyCopyNew_.py
#
#Created on Thu May 26 14:13:20 2016
#
#@author: cusgadmin
#"""
#
#import tensorflow as tf
#
#def TransDef(scope=None, reuse=None, lsizes = None, num_ac=0, hor=0.0, dt=0.01, nu=0.1, mom=0.9):
#    with tf.variable_scope(scope, reuse=reuse):
#        states = tf.placeholder(tf.float32,shape=(None,lsizes[0]),name="states");
#        y = tf.placeholder(tf.float32,shape=(None,lsizes[-1]),name="y");   
#    
#        lw = [[]]*int(hor/dt);
#        lb = [[]]*int(hor/dt);
#        l = [[]]*int(hor/dt);
#        last_ba = [];
#        cross_entropy = [];
#        L = [];
#        PI = [];
#        train_vars = [];
#        train_step = [];
#        
#        for k in xrange(-1*int(hor/dt)):
#            with tf.variable_scope("Vars_Pol_"+str(k)):
#                for i in xrange(len(lsizes) - 1):
#                    lw[k].append(0.1*tf.Variable(tf.random_uniform([lsizes[i] + (i==0)*(k!=0)*(2**num_ac),lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="H"+str(i)));
#                    lb[k].append(0.1*tf.Variable(tf.random_uniform([1,lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="B"+str(i)));
#                    
#                if(k == 0):    
#                    l[k].append(tf.nn.relu(tf.add(tf.matmul(states,lw[k][0]), lb[k][0]),name="A"+str(0)))
#                else:
#                    l[k].append(tf.nn.relu(tf.add(tf.matmul(tf.concat(1,[states,l[k-1][-1]]),lw[k][0]), lb[k][0]),name="A"+str(0)))
#                
#                for i in xrange(len(lw[k])-2):
#                    l[k].append(tf.nn.relu(tf.add(tf.matmul(l[k][-1],lw[k][i+1]), lb[k][i+1]),name="A"+str(i)));
#                
#                last_ba.append(tf.add(tf.matmul(l[k][-1],lw[k][-1]), lb[k][-1]));
#                l[k].append(tf.nn.softmax(last_ba[-1]));
#                
#                cross_entropy.append(tf.nn.softmax_cross_entropy_with_logits(logits=last_ba[-1],labels=y));
#    
#                L.append(tf.reduce_mean(cross_entropy[-1]))
#            
#                PI.append(l[k][-1])
#            
#            train_vars.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Vars_Pol_"+str(k)))
#            train_step.append(tf.train.RMSPropOptimizer(learning_rate=nu,momentum=mom).minimize(L[-1],var_list=train_vars[-1]))
#        
#    return states,y,PI,L,l,lb,cross_entropy,train_step
