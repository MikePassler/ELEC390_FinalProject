import numpy as np
import h5py

#dummy data 
train = np.random.random(size = (100,33))
test = np.random.random(size = (100,333))
# Will need to load 5 seperate csv files testing, training, and 3 csv files from group members.

hf = h5py.File('hdf5_data.h5', 'w')


g1 = hf.create_group('data')
g2 = hf.create_group('Mike')
g3 = hf.create_group('Nasser')
g4 = hf.create_group('Mattew')

g1.create_dataset('training_data',data=train)
g1.create_dataset('testing_data', data=test)


#
#g2.create_dataset('Mike_data', data=)



