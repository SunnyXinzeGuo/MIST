import numpy as np
def read_beam_FEKO(filename, AZ_antenna_axis):
	
	"""
	This function reads '.out' files from FEKO.
	They must have a spatial resolution of (dAZ, dEL) of 1deg x 1deg.
	
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
	# --------------------------------------------------------------------------
	if np.max(theta) <= 90:
		EL = np.copy(theta)       # We do not change theta, but instead we flip the gain below
	elif np.max(theta) > 90:
		EL = theta - 90
		
	AZ = np.copy(phi)

	gain = np.fliplr(gain)    # shifting from theta to EL
	Et   = np.fliplr(Et)      # shifting from theta to EL
	Ep   = np.fliplr(Ep)      # shifting from theta to EL






	# Shifting beam relative to true AZ (referenced at due North)
	# Due to angle of orientation of excited antenna panels relative to due North
	# ---------------------------------------------------------------------------
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


