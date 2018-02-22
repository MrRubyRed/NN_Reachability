# -*- coding: utf-8 -*-
"""
- Helper Function for CopyCopyNew_.py

Created on Thu May 26 14:13:20 2016

@author: cusgadmin
"""

import tensorflow as tf

def lrelu(x):
  return tf.nn.relu(x) - 0.01*tf.nn.relu(-x)

def TransDef(scope=None, reuse=None, lsizes = None, depth=None, incl=None, center=None, outp=False):
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
            
#        l.append(tf.nn.tanh(tf.add(tf.matmul(states,lw[0]), lb[0])))
#        for i in xrange(len(lw)-2):
#            l.append(tf.nn.tanh(tf.add(tf.matmul(l[-1],lw[i+1]), lb[i+1])));
        
        last_ba = tf.add(tf.matmul(l[-1],lw[-1]), lb[-1],name="A_end");
        #l.append(tf.nn.sigmoid(last_ba));
        l.append(tf.nn.softmax(last_ba));
        #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=last_ba,targets=y)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=last_ba,labels=y)
        #cross_entropy = -(tf.mul(y,tf.log(l[-1])) + tf.mul(1.0-y,tf.log(1.0-l[-1])))
        #cross_entropy = tf.maximum(tf.log(l[-1]), 0) - tf.log(l[-1]) * y + tf.log(1 + tf.exp(-tf.abs(tf.log(l[-1]))))
        L = tf.reduce_mean(cross_entropy)#tf.reduce_sum(cross_entropy,reduction_indices=1));
        
        #V_x0 = incl*tf.sqrt(tf.reduce_sum(tf.square(s_vec),1,keep_dims=True)) - depth;
        PI = l[-1];
        
    return states,y,PI,L,l,lb,reg,cross_entropy

def TransDef_(scope=None, reuse=None, lsizes = None, sub_sys=None):
    with tf.variable_scope(scope, reuse=reuse):
        states = tf.placeholder(tf.float32,shape=(None,lsizes[0]),name="states");
        sub_sys1 = tf.slice(states, [0,0], [-1,4]);
        sub_sys2 = tf.slice(states, [0,4], [-1,4]);
        sub_sys3 = tf.slice(states, [0,8], [-1,2]);
        sys1 = tf.concat((sub_sys1,sub_sys3),axis=1) #TODO
        sys2 = tf.concat((sub_sys2,sub_sys3),axis=1)
        y = tf.placeholder(tf.float32,shape=(None,lsizes[-1]),name="y");   
   
        lsizes_ = lsizes[:]
        lsizes_[0] = lsizes_[0]-4

        lw = []
        lb = [];
        l = [];
        reg = 0.0;
        for i in xrange(len(lsizes_) - 1):
            lw.append(0.1*tf.Variable(tf.random_uniform([lsizes_[i],lsizes_[i + 1]],-1.0,1.0,dtype=tf.float32)));
            lb.append(0.1*tf.Variable(tf.random_uniform([1,lsizes_[i + 1]],-1.0,1.0,dtype=tf.float32)));
            reg = reg + tf.reduce_sum(tf.abs(lw[-1])) + tf.reduce_sum(tf.abs(lb[-1]));
            
        l.append(lrelu(tf.add(tf.matmul(sys1,lw[0]), lb[0])))
        for i in xrange(len(lw)-2):
            l.append(lrelu(tf.add(tf.matmul(l[-1],lw[i+1]), lb[i+1])));

        last_ba = lrelu(tf.add(tf.matmul(l[-1],lw[-1]), lb[-1]))

        lw_ = [];
        lb_ = [];
        l_ = [];
        reg_ = 0.0;
        for i in xrange(len(lsizes_) - 1):
            lw_.append(0.1*tf.Variable(tf.random_uniform([lsizes_[i],lsizes_[i + 1]],-1.0,1.0,dtype=tf.float32)));
            lb_.append(0.1*tf.Variable(tf.random_uniform([1,lsizes_[i + 1]],-1.0,1.0,dtype=tf.float32)));
            reg_ = reg_ + tf.reduce_sum(tf.abs(lw_[-1])) + tf.reduce_sum(tf.abs(lb_[-1]));
            
        l_.append(lrelu(tf.add(tf.matmul(sys2,lw_[0]), lb_[0])))
        for i in xrange(len(lw_)-2):
            l_.append(lrelu(tf.add(tf.matmul(l_[-1],lw_[i+1]), lb_[i+1])));

        last_ba_ = lrelu(tf.add(tf.matmul(l_[-1],lw_[-1]), lb_[-1]))

        mergenet = tf.concat((last_ba,last_ba_),axis=1)
        final_ba = tf.add(tf.matmul(mergenet,0.1*tf.Variable(tf.random_uniform([2*lsizes[-1],lsizes[-1]],-1.0,1.0,dtype=tf.float32))),0.1*tf.Variable(tf.random_uniform([1,lsizes[-1]],-1.0,1.0,dtype=tf.float32)))
        #l.append(tf.nn.sigmoid(last_ba));
        l.append(tf.nn.softmax(final_ba));
        #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=last_ba,targets=y)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=final_ba,labels=y)
        #cross_entropy = -(tf.mul(y,tf.log(l[-1])) + tf.mul(1.0-y,tf.log(1.0-l[-1])))
        #cross_entropy = tf.maximum(tf.log(l[-1]), 0) - tf.log(l[-1]) * y + tf.log(1 + tf.exp(-tf.abs(tf.log(l[-1]))))
        L = tf.reduce_mean(cross_entropy)#tf.reduce_sum(cross_entropy,reduction_indices=1));
        
        #V_x0 = incl*tf.sqrt(tf.reduce_sum(tf.square(s_vec),1,keep_dims=True)) - depth;
        PI = l[-1];
        
    return states,y,PI,L,l,lb,reg,cross_entropy
