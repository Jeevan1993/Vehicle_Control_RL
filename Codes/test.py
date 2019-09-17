#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 02:57:30 2019

@author: jeevan
"""
import numpy as np
import matplotlib.pyplot as plt
m=np.empty((100,100))
m.fill(0)
cx=50
cy=50
l=6

#for l in range(10):
r=1.4*np.pi/5
#r=0
#    vert = [[cx + l / 2, cy + l / 2],
#        [cx - l / 2, cy + l / 2],
#        [cx - l / 2, cy - l / 2],
#        [cx + l / 2, cy - l / 2]]
vert=[]
for i in range(-l,l,1):
    for j in range(-l,l,1):
        print([cx+i,cy+j])
        vert.append([cx+i,cy+j])
#        m[cx+i,cy+j]=1
#print(len(vert))
r_xys = []

for x, y in vert:
    tempX = x - cx
    tempY = y - cy
    # apply rotation
    rotatedX = tempX * np.cos(r) - tempY * np.sin(r)
    rotatedY = tempX * np.sin(r) + tempY * np.cos(r)
    # rotated x y
    x = rotatedX + cx
    y = rotatedY + cy
    r_xys += [[x, y]]

for i in range(len(r_xys)):
#    print(r_xys[i])
#    print(np.array(r_xys[i]).astype(int))
    
    x,y=np.array(r_xys[i]).astype(int)
    m[x,y]=1

    
    
    
    
print(np.array(r_xys[0]).astype(int))
#x,y=np.array(r_xys[0]).astype(int)
#m[[x,y]]=1
#m[np.array(r_xys[0]).astype(int)=1
plt.imshow(m)
plt.show()





















