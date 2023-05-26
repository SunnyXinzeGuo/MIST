#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:23:52 2022

@author: sunnyguo
"""

import numpy as np
# from scipy.interpolate import interpn
import datetime
from glob import glob
import os
from scipy.stats import mode

#---------------------------------------------------------------------------

def read_beam_FEKO(filename, AZ_antenna_axis):

     """
     This function reads '.out' files from FEKO.
     They must have a spatial resolution of (dAZ, dEL) of 1deg x 1deg.
     @author: Dr. Raul Monsalve
     Note that lines 24 - 155 are solely the work of Dr. Monsalve.
     """

     f_list = []
     d_list = []
     with open(filename) as fobject:
         NEXT = 0
         for line in fobject:
             # Extracting frequency
             if 'FREQ' in line:
                 print(line)
                 for i in range(len(line)):
                     if line[i]=='=':
                         x = float(line[i+1::])
                         f_list.append(x)

             # Extracting beam
             if (NEXT == 1) and (line == "\n"):
                 NEXT = 0

             elif ('THETA' in line) and ('PHI' in line) and ('ETHETA' not in line) and ('angular' not in line) and ('grid' not in line) and (NEXT == 0):
                 NEXT = 1
                 #print(line)

             elif (NEXT == 1) and ('THETA' not in line):
                 line_array  = line.split()
                 line_array2 = np.array(line_array[0:11])
                 data = line_array2.astype(np.float)

                 d_list.append(data)

     # Frequency array
     f_array = np.array(f_list)
     f = np.unique(f_array)

     # Data list to array
     d_array = np.zeros((len(d_list), len(d_list[0])))
     #print(d_array.shape)
     for i in range(len(d_list)):
         d_array[i,:] = d_list[i]

     # Theta and Phi
     theta = np.unique(d_array[:,0])
     phi   = np.unique(d_array[:,1])

     # Gain
     gain_1 = 10**(d_array[:,8]/10)
     gain_2 = gain_1.reshape((len(f), int(len(gain_1)/(len(f)*len(theta))), len(theta)))
     gain   = np.transpose(gain_2, (0,2,1))

     # E_theta
     Et_1 = d_array[:,2] * ( np.cos((np.pi/180)*d_array[:,3]) + 1j*np.sin((np.pi/180)*d_array[:,3]) )
     Et_2 = Et_1.reshape((len(f), int(len(Et_1)/(len(f)*len(theta))), len(theta)))
     Et   = np.transpose(Et_2, (0,2,1))

     # E_phi
     Ep_1 = d_array[:,4] * ( np.cos((np.pi/180)*d_array[:,5]) + 1j*np.sin((np.pi/180)*d_array[:,5]) )
     Ep_2 = Ep_1.reshape((len(f), int(len(Ep_1)/(len(f)*len(theta))), len(theta)))
     Ep   = np.transpose(Ep_2, (0,2,1))


     # Change from antenna coordinates (theta, phi) to local coordinates (AZ, EL)
     #
#--------------------------------------------------------------------------
     if np.max(theta) <= 90:
         EL = np.copy(theta)       # We do not change theta, but instead we flip the gain below
     elif np.max(theta) > 90:
         EL = theta - 90

     AZ = np.copy(phi)

     gain = np.fliplr(gain)    # shifting from theta to EL
     Et   = np.fliplr(Et)      # shifting from theta to EL
     Ep   = np.fliplr(Ep)      # shifting from theta to EL

     # Shifting beam relative to true AZ (referenced at due North)
     # Due to angle of orientation of excited antenna panels relative todue North
     #
#---------------------------------------------------------------------------
     print('AZ_antenna_axis = ' + str(AZ_antenna_axis) + ' deg')

     # Right now, this only works if the resolution in azimuth (phi) is 1 degree. FIX this in the future. Make it more general.
     if phi[1]-phi[0] == 1:

         if AZ_antenna_axis < 0:
             AZ_index          = -AZ_antenna_axis
             g1                = gain[:,:,AZ_index::]
             g2                = gain[:,:,0:AZ_index]
             gain_shifted      = np.append(g1, g2, axis=2)

             Et1               = Et[:,:,AZ_index::]
             Et2               = Et[:,:,0:AZ_index]
             Et_shifted        = np.append(Et1, Et2, axis=2)

             Ep1               = Ep[:,:,AZ_index::]
             Ep2               = Ep[:,:,0:AZ_index]
             Ep_shifted        = np.append(Ep1, Ep2, axis=2)

         elif AZ_antenna_axis > 0:
             AZ_index          = AZ_antenna_axis
             g1                = gain[:,:,0:(-AZ_index)]
             g2                = gain[:,:,(360-AZ_index)::]
             gain_shifted      = np.append(g2, g1, axis=2)

             Et1               = Et[:,:,0:(-AZ_index)]
             Et2               = Et[:,:,(360-AZ_index)::]
             Et_shifted        = np.append(Et2, Et1, axis=2)

             Ep1               = Ep[:,:,0:(-AZ_index)]
             Ep2               = Ep[:,:,(360-AZ_index)::]
             Ep_shifted        = np.append(Ep2, Ep1, axis=2)

         elif AZ_antenna_axis == 0:
             gain_shifted      = np.copy(gain)
             Et_shifted        = np.copy(Et)
             Ep_shifted        = np.copy(Ep)

     else:
         print('-------------------')
         print('ERROR: The beam file does not have a resolution of 1 degree in AZ (phi).')
         print('-------------------')
         return 0,0,0,0,0,0

     return f, AZ, EL, Et_shifted, Ep_shifted, gain_shifted

#---------------------------------------------------------------------------
def pg_collector(filepath):
    
    # This function accesses a filepath containing your simulated files (make
    # sure that only simulations are contained in this filepath!). It will
    # access each file name and read off the parameters contained in the
    # name. It will also read the gain values of each file. It returns to you 
    # a 2D array of parameters, the gains associated with each point, 
    # and the "antenna cube points" comprising the coordinates of those gain values. 
    
    # See the README for more comprehensive information regarding this function.
    
    t1 = datetime.datetime.now()
    
    os.chdir(filepath)
    
    all_files = glob('*')
    
    # Note that glob('*') collects -all- the files in a directory. It is
    # important that only the files you want to read are here. If there's
    # some other file that can't be read by read_FEKO_beam, there will be 
    # an error message returned.
    
    # Determine how many files are in the directory:
        
    number_of_files = len(all_files)
    
    # Create some empty arrays to contain data/information:
        
    user_info = []
    n_params_per_file = []
    parameters = []
    gains = []
    
    for file in all_files:
        
        # Split the file to gather its information.
        
        file_info = file.split('-')
        n_params = len(file_info) - 1
        n_params_per_file.append(n_params)
        
        # file_info can then be indexed to gather the necessary infromation.
        # First remove the antenna type/user info and put it in a separate 
        # list, then call the leftover parameter_info since it only contains 
        # parameters.
        
        user_info.append(file_info[0])
        parameter_info = file_info[1:] 
            
        # This loop turns the list of parameters into a list of floats. It is
        # generalized such that anything other than numbers and decimal points
        # are trimmed from the string. It then changes the remaining decimal
        # (ex: '0.20') from a string to a float.
        
        for i in range(n_params):
            parameter_info[i] = ''.join(c for c in parameter_info[i] if c.isdigit() or c == '.')
            extra_period_pos = parameter_info[i].find('.', parameter_info[i].find('.') + 1)
            if extra_period_pos != -1:
                parameter_info[i] = parameter_info[i][:extra_period_pos]
        parameter_info[i] = float(parameter_info[i])
    
        parameters.append(parameter_info)
        
        # Now we can read the data from the file:
        
        AZ_antenna_axis = 0
        f, AZ, EL, Et_shifted, Ep_shifted, gain_shifted = read_beam_FEKO(file, AZ_antenna_axis)
        gain = gain_shifted.flatten()
        gains.append(gain)
        
    # The following goes through the array containing the number of 
    # parameters for each file, and checks to make sure they are all the
    # same. If the number of parameters per file is constant, no message
    # is returned. If a file has more/less parameters than the mode of the 
    # number of parameters array, an error message will be printed; it will
    # tell you which file is problematic, and how many more/less parameters
    # that file name has than the mode.
    
    if len(np.unique(n_params_per_file)) != 1:
        most_common_param = int(mode(n_params_per_file)[0])
        for i in range(len(n_params_per_file)):
            if n_params_per_file[i] > most_common_param:
                print('{} {} {} {} {} {}'.format('ParameterLengthError:', 
                                           'File', all_files[i], 'contains', 
                                           n_params_per_file[i] - most_common_param, 
                                           'more parameter(s) than the other files.'))
            if n_params_per_file[i] < most_common_param:
                print('{} {} {} {} {} {}'.format('ParameterLengthError:', 
                                           'File', all_files[i], 'contains', 
                                           most_common_param - n_params_per_file[i], 
                                           'less parameter(s) than the other files.'))
    
    # Collect the parameter points and gains: 
        
    parameters = np.array(parameters)
    parameters = parameters.reshape((number_of_files, n_params))
    parameters = parameters.astype(np.float)
    gains = np.array(gains)
    gains = gains.T
    
    # Gather antenna "cube" coordinates. These coordinates are generalized.
    
    f_cube = np.zeros((len(f), len(EL), len(AZ)))
    
    for i in range(0, len(f)):
        f_cube[i, :, :] = (i+40)*np.ones((len(EL), len(AZ)))*1e6
        
    f_cube = f_cube.flatten()
        
    az_cube = np.zeros((len(f), len(EL), len(AZ)))
    
    for i in range(0, len(AZ)):
        az_cube[:, :, i] = i*np.ones((len(f), len(EL)))
        
    az_cube = az_cube.flatten()
        
    el_cube = np.zeros((len(f), len(EL), len(AZ)))
    
    for i in range(0, len(EL)):
        el_cube[:, i, :] = i*np.ones((len(f), len(AZ)))
        
    el_cube = el_cube.flatten()
    
    coordinates = np.column_stack((f_cube, az_cube, el_cube))
    
    t2 = datetime.datetime.now()
    
    print('Read files in:', t2 - t1)
    
    return parameters, gains, coordinates

#---------------------------------------------------------------------------
# polynd
# Routine to fit n-dimensional polynomials
# @author: Prof. Jon Sievers

# It is noted that lines 296 - 400 are solely the work of Prof. Sievers.

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
    
    # Version 2.2
    
    # Note that this version is outdated, and kept here for documentation 
    # purposes. This version also relies upon old notation, which does not
    # adhere to the up-to-date notation.
    
    # This requires that the points lie on a grid. The minimum number
    # of distinct parameters for each parameter is 2. The minimum number
    # of files you can have is then 2^D, where D is the dimensionality
    # of your parameters (D = how many parameters you have).
    
    # D = points.shape[1] # Parameter space dimension
    
    # t1 = datetime.datetime.now()
    # npt = 2*points.shape[0]
    
    # grid_points = []
    
    # for i in range(D):
    #     unique_param_vals = np.unique(points[:, i])
    #     grid_points.append(unique_param_vals)
            
    # n = int(points.shape[0]**(1/D)) # distinct number of parameters
    
    # list_of_n = list(np.repeat(n, D))
                                      
    # nn = gains.shape[0] // npt # // means floor division
    # tmp = gains[::nn,:] # gather first nn rows of gain_values
    # npt = tmp.shape[0] # number of rows in tmp
    # vals = np.zeros(npt) # array to hold interpolated values
    
    # # Interpolate a small number of gains:
        
    # for i in range(npt):
    #     vals[i]=interpn(grid_points, np.reshape(tmp[i, :], list_of_n), Point)
        
    # # This is the SVD way of getting weights out of the
    # # sub-sampled interpolation. Note that "@" is matrix multiplication.
    
    # u, s, v = np.linalg.svd(tmp, 0)
    # thresh = 1e-8*np.max(s)
    # ind = s > thresh
    # u = u[:, ind]
    # s = s[ind]
    # v = v[ind, :]
    # weights = v.T @ np.diag(1.0/s) @ (u.T @ vals) # vals is observed/interpolated data points

    # # Use the weights to recover the full interpolated beam model:

    # output_gains = gains @ weights
    
    # # Interpolate a small random sample to compare with the SVD method to
    # # make sure it is extracting the interpolated values correctly (self-
    # # checker):
        
    # rand_samp_indices = np.unique(np.random.randint(0, output_gains.shape[0], size = npt))
        
    # rand_samp_interpn_gains = []
    # rand_samp_SVD_gains = []
    
    # for i in rand_samp_indices:
    #     rand_samp_interpn_gains.append(interpn(grid_points, np.reshape(gains[i, :], list_of_n), Point))
    #     rand_samp_SVD_gains.append(output_gains[i])
    
    # rand_samp_interpn_gains = np.array(rand_samp_interpn_gains)
    # rand_samp_interpn_gains = rand_samp_interpn_gains.flatten()
    # rand_samp_SVD_gains = np.array(rand_samp_SVD_gains)
    
    # def rms(interpn, SVD):
    #     ith_sum_term = ((SVD - interpn)/interpn)**2
    #     return 100*np.sqrt((1/int(len(interpn)))*np.sum(ith_sum_term))
    
    # t2 = datetime.datetime.now()
    
    # print('Interpolated a new beam model in:', t2 - t1)
    # print('RMS% Agreement Between SVD and interpn: ', rms(rand_samp_interpn_gains, rand_samp_SVD_gains))
    
    # output_gains = np.reshape(output_gains, (len(gains), 1))
    
    # return output_gains
    
    #-----------------------------------------------------------------------
    
    # Version 3.0
    
    # This is the current version. It relies upon polynd as written by
    # Prof. Jon Sievers above.
    
    wts = interp_weights(parameters, interpolation_parameters, poly)
    interpolated_gains = gains @ wts
    
    return interpolated_gains

def read_n_interp(filepath, all_interpolation_parameters, poly):
    
    # This function calls both pg_collector and abm_interpolator at the 
    # same time. "all_interpolation_parameters" is a 2D array of parameters, 
    # each row being a different set of parameters you'd like to interpolate.
    
    t1 = datetime.datetime.now()
    
    parameters, gains, coordinates = pg_collector(filepath)
    
    # all_interpolated_gains is an array whose rows will be filled with the
    # interpolated gains corresponding to each row in all_interpolation_parameters. 
    # It has shape ((all_interpolation_parameters[0], len(gains))
    
    all_interpolated_gains = np.zeros((all_interpolation_parameters.shape[0], len(gains)))
    
    # for i in range(all_interpolation_parameters.shape[0]):
    #     all_interpolated_gains[i] = abm_interpolator(parameters, gains, all_interpolation_parameters[i], poly)
        
    for i in range(all_interpolation_parameters.shape[0]):
        wts = interp_weights(parameters, all_interpolation_parameters[i], poly)
        all_interpolated_gains[i] = gains @ wts
        
    t2 = datetime.datetime.now()
    print('Read and interpolated in:', t2 - t1)
    return parameters, gains, coordinates, all_interpolated_gains

#__________________________________________________________________________________________








