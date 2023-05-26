#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 07:12:25 2022

@author: sunnyguo
"""

import numpy as np

def bc_generator(para):
    #initial parameters: A, x0, y0, sigx, sigy, longitude, latitudem frequency 
    #azimuth is longtitude; elevation is latitude
    azimuth = np.array(range(0,360))
    elevation = np.array(range(-90,91))
    freq = np.array(range(50,101))
    A, sigx, sigy = para[0]*(freq/75), para[1], para[2]  
    x0, y0 = 0, 90

    elev = np.exp(-((elevation-y0)**2/(2*(sigy**2))))
    surface = np.tile(elev, (360,1))
    azi = np.exp((-(azimuth-x0)**2/(2*sigx**2)))
    weight1 = np.tile(azi, (181,1))
    weight1 = weight1.T
    plane = weight1*surface
    plane=plane.T
    cube = np.tile(plane, (51,1,1))
    A_ = np.tile(A, (360,181,1))
    A_=A_.T
    cube = cube*A_
    return cube

def bc_flat(bc):
    return np.ndarray.flatten(bc)

def new_pg_collector(A,sigx,sigy):
    #parameters are in the form (A,sigx,sigy)
    Anew = np.repeat(A,len(sigx)*len(sigy))
    sigxn = np.repeat(sigx, len(sigy))
    sigxnew = np.tile(sigxn,len(A))
    sigynew = np.tile(sigy, len(A)*len(sigx))
    parameters = np.column_stack((Anew,sigxnew,sigynew))
    freq = np.array(range(50,101))
    frequency = np.repeat(freq,65160)
    ele = np.array(range(-90,91))
    elev = np.repeat(ele,360)
    elevation = np.tile(elev,51)
    azi = np.array(range(0,360))
    azimuth = np.tile(azi, 9231)
    coordinates = np.column_stack((frequency,elevation,azimuth))
    for i in range(len(parameters)):
        beamcube = bc_generator(parameters[i])
        beamcube1D = bc_flat(beamcube)
        if i == 0:
            gains = beamcube1D
        else:
            gains = np.column_stack((gains, beamcube1D))
    return parameters, gains, coordinates


#_____________________________________________________________________________

def vec2mat_nd(vec,dims,ind):
    ndim=len(dims)
    while ind<0:
        ind+=ndim

    mat=np.empty(dims)-1
    if ind==0:
        left=''
    else:
        left=':,'*(ind)    
    if ind==ndim-1:
        right=''
    else:
        right=',:'*(ndim-ind-1)
    for i in range(len(vec)):
        myslice=left+repr(i)+right
        to_exec='mat['+myslice+']=vec['+repr(i)+']'
        exec(to_exec)
    return mat

def grid2mat(x):
    ndim=len(x)
    dims=[None]*ndim
    for i in range(ndim):
        dims[i]=len(x[i])
    n=np.product(dims)
    mat=np.zeros([n,ndim])
    for i in range(ndim):
        mat[:,i]=np.ravel(vec2mat_nd(x[i],dims,i))
    return mat

def make_A(x,ord):
    #print(x.shape)
    n=x.shape[0]
    # print('ord is ',ord)
    ord=np.asarray(ord,dtype='int')
    m=np.prod(ord+1)
    
    ndim=x.shape[1]
    A=np.ones([n,m])
    for i in range(m):
        mypows=np.unravel_index(i,ord+1)
        for j in range(ndim):
            A[:,i]=A[:,i]*x[:,j]**mypows[j]
    return A

def polyfitnd(x,y,ord):

    A=make_A(x,ord)
    lhs=A.T@A
    rhs=A.T@y
    fitp=np.linalg.inv(lhs)@rhs
    return fitp

def polyfitnd_grid(x,y,ord=None):

    if ord is None:
        #ndim=len(x)
        #dims=[None]*ndim
        #for i in range(ndim):
        #    ord[i]=len(x[i])-1
        ord=[len(z)-1 for z in x]
        print(ord)
    xx=grid2mat(x)
    return polyfitnd(xx,y,ord)

    dims=y.shape
    ndim=len(dims)
    assert(len(x)==ndim)
    for i in range(ndim):
        assert(len(x[i])==dims[i])
    #assert(len(y.shape)==ndim)
    #assert(np.sum(np.abs(dims-y.shape))==0)  #fails if x and y are different dimensions
             
    if ord is None:
        ord=dims
    if isinstance(ord,int):
        ord=np.asarray(np.ones(ndim)*ord,dtype='int')
    assert(len(ord)==ndim)
    print(ord)

    vecs=[None]*ndim
    for i in range(ndim):
        vecs[i]=vec2mat_nd(x[i],dims,i)

    return vecs
                          
def interp_weights(x,targ,ord):
    xx=np.vstack([x,targ])
    # print('xx shape is ',xx.shape)
    allA=make_A(xx,ord)
    # print('allA shape is ',allA.shape)
    A=allA[:-1,:]
    vec=allA[-1,:]
    lhs=A.T@A
    wts=vec@(np.linalg.inv(lhs)@(A.T))
    return wts

def interp_weights_grid(x,targ,ord=None):
    if ord is None:
        ord=[len(y)-1 for y in x]
    xx=grid2mat(x)
    wts=interp_weights(xx,targ,ord)
    return np.reshape(wts,np.asarray(ord,dtype='int')+1)
#return interp_weights(xx,targ,ord)

#---------------------------------------------------------------------------
def abm_interpolator(parameters, gains, interpolation_parameters, poly):
    
    # This function interpolates new beam models. As with pg_collector,
    # see the README for more detailed information on its use, or to my report
    # listed at the top. The current version (Version 3.0) does not rely upon the SVD method
    # as detailed in the report, but rather on polynd as written above by
    # Prof. Jon Sievers. This version simply calls the functions that he 
    # has written in polynd. The arguments of abm_interpolator are the same
    # as in the report, with addition of a new argument, "poly", described here:
        
    # New argument: poly - a 1xD list containing the order of the polynomial
    # used for interpolation for each dimension/parameter. Note that D follows
    # that reference in the report listed at the top of the script.
    
    # Ex) If I have D = 3 (i.e. I have three distinct parameters, as in blade 
    # length, blade separation, and soil conductivity) and I want a linear 
    # interpolation, poly = [1,1,1]. If instead I wanted a second-order 
    # polynomial in each parameter, I would set poly = [2,2,2].
    
    #-----------------------------------------------------------------------    
    # Version 3.0
    
    # This is the current version. It relies upon polynd as written by
    # Prof. Jon Sievers above.
    
    wts = interp_weights(parameters, interpolation_parameters, poly)
    interpolated_gains = gains @ wts
    
    return interpolated_gains

def read_n_interp(A, sigx, sigy, all_interpolation_parameters, poly):
    
    # This function calls both pg_collector and abm_interpolator at the 
    # same time. "all_interpolation_parameters" is a 2D array of parameters, 
    # each row being a different set of parameters you'd like to interpolate.
        
    parameters, gains, coordinates = new_pg_collector(A,sigx,sigy)
    
    # all_interpolated_gains is an array whose rows will be filled with the
    # interpolated gains corresponding to each row in all_interpolation_parameters. 
    # It has shape ((all_interpolation_parameters[0], len(gains))
    
    all_interpolated_gains = np.zeros((all_interpolation_parameters.shape[0], len(gains)))
    
    # for i in range(all_interpolation_parameters.shape[0]):
    #     all_interpolated_gains[i] = abm_interpolator(parameters, gains, all_interpolation_parameters[i], poly)
        
    for i in range(all_interpolation_parameters.shape[0]):
        wts = interp_weights(parameters, all_interpolation_parameters[i], poly)
        all_interpolated_gains[i] = gains @ wts
        
    return parameters, gains, coordinates, all_interpolated_gains
#____________________________________________________________________________
# Other functions written by Sunny
def percentage(x,y):
    return 100*abs(x-y)/y

def table_cube(table):
    cube = table.reshape((51,181,360))
    return cube


