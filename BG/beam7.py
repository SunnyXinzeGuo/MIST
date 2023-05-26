#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:52:25 2022

@author: sunnyguo
"""

# beam gain 7
import numpy as np

long = np.array(range(0,360))
lat = np.array(range(0,181))
freq = np.array(range(50,101))

A = 10*(freq/75)
x0 = 180
y0 = 90
sigx = 90
sigy = 45

def model(x,y,freq):
    return A*np.exp((-(x-x0)**2/(2*sigx**2))-((y-y0)**2/(2*(sigy**2))))

lat_ = np.exp(-((lat-y0)**2/(2*(sigy**2))))
surface = np.tile(lat_, (360,1))
long_ = np.exp((-(long-x0)**2/(2*sigx**2)))
weight1 = np.tile(long_, (181,1))
weight1 = weight1.T
plane = weight1*surface
plane=plane.T
cube = np.tile(plane, (51,1,1))

A_ = np.tile(A, (360,181,1))
A_=A_.T

cube = cube*A_
print(cube.shape)
