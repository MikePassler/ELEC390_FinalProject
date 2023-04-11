import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:/Users/micha/ELEC390_FINALPROJECT/test_train.csv')

# Plot the x, y, and z acceleration versus time
print(dataset.columns)
plt.plot(dataset['Time (s)'], dataset['Acceleration x (m/s^2)'], label='X acceleration')
plt.plot(dataset['Time (s)'], dataset['Acceleration y (m/s^2)'], label='Y acceleration')
plt.plot(dataset['Time (s)'], dataset['Acceleration z (m/s^2)'], label='Z acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Acceleration vs Time')
plt.legend()
plt.show()