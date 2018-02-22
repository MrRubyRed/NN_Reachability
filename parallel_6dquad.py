# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:57:17 2016

@author: cusgadmin
"""

from multiprocessing import Process, Queue
import tensorflow as tf
import numpy as np
from FAuxFuncs2_0 import TransDef
import matplotlib.pyplot as plt

import time
import FMLHigh6quad_rec

depth = 1.0;
def V_true(x,t):
        return np.linalg.norm(np.exp(-t)*x) - depth;

# [3+1,50,30,1]works well
layers1 = [4+1,50,45,1];
layers2 = [4+1,50,45,1];
layers3 = [4+1,50,45,1];
layers4 = [4+1,50,45,1];
layers5 = [4+1,50,45,1];
layers6 = [4+1,50,45,1];
layers7 = [4+1,50,45,1];
layers8 = [4+1,50,45,1];
t_hor = -1.0;
q = Queue();


t1 = time.time()
p1 = Process(target=FMLHigh6quad_rec.main,args=(layers1,t_hor,0,1000,10,0.001,0.999,200,1000,0.0,q,))
p2 = Process(target=FMLHigh6quad_rec.main,args=(layers2,t_hor,1,1000,10,0.001,0.999,200,1000,0.0,q,))
p3 = Process(target=FMLHigh6quad_rec.main,args=(layers3,t_hor,2,1000,10,0.001,0.999,200,1000,0.0,q,))
p4 = Process(target=FMLHigh6quad_rec.main,args=(layers4,t_hor,3,1000,10,0.001,0.999,200,1000,0.0,q,))
p5 = Process(target=FMLHigh6quad_rec.main,args=(layers5,t_hor,4,1000,10,0.001,0.999,200,1000,0.0,q,))
p6 = Process(target=FMLHigh6quad_rec.main,args=(layers6,t_hor,5,1000,10,0.001,0.999,200,1000,0.0,q,))
p7 = Process(target=FMLHigh6quad_rec.main,args=(layers7,t_hor,6,1000,10,0.001,0.999,200,1000,0.0,q,))
p8 = Process(target=FMLHigh6quad_rec.main,args=(layers8,t_hor,7,1000,10,0.001,0.999,200,1000,0.0,q,))
p1.start()
p2.start()
p3.start()
p4.start()
p5.start()
p6.start()
p7.start()
p8.start()
p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6.join()
p7.join()
p8.join()
t2 = time.time()
print "time spent: {0:.2f}".format(t2 - t1)
