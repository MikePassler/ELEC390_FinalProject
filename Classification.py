import pandas as pd
import numpy as np
import h5py
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from scipy.stats import skew, kurtosis

mike_data = pd.read_csv('C:/Users/micha/ELEC390_FINALPROJECT/Mike_data.csv')
Nass_data = pd.read_csv('C:/Users/micha/ELEC390_FINALPROJECT/Nass_data.csv')
df = pd.read_csv('C:/Users/micha/ELEC390_FINALPROJECT/full_data.csv')
Matt_data = pd.read_csv('C:/Users/micha/ELEC390_FINALPROJECT/Matt_data.csv')
print(df)
# Create a list of DataFrame segments, where each segment has 496 rows
windows = [df[i:i+496] for i in range(0, len(df), 496)]

# Add a "window" column to each segment with a unique window number
for i, window in enumerate(windows):
    window['window'] = int(i)

# Combine all segments back into a single DataFrame
df = pd.concat(windows)
df['window'] = df['window'].astype(int)
print(df)


windowSize = 3 # set the window size for the moving average filter
data = pd.DataFrame(df.rolling(window=windowSize).mean()) # apply the rolling function with the specified window size to compute the moving average

print(data)

indexes = np.where(data.isnull())
print("NaNs found:", indexes[0].size)
indexes = np.where(data == '-')
print("'-'s found:", indexes[0].size)

values = {'Time (s)':0,
          'Acceleration x (m/s^2)' : 0,
          'Acceleration y (m/s^2)' : 0,
          'Acceleration z (m/s^2)' : 0, 
          'Absolute acceleration (m/s^2)': 0,
          'window' : 0
}
data.fillna(value=values, inplace=True)

data = data.drop(['Time (s)','Acceleration x (m/s^2)', 'Acceleration y (m/s^2)','Absolute acceleration (m/s^2)'], axis = 1)
data['window'] = data['window'].astype(int)
print(data)

# group by window and calculate mean, std, max, and min of each group
grouped_data = data.groupby('window')['Acceleration z (m/s^2)'].agg(['mean', 'std', 'max', 'min', 'median', 'var',  lambda x: skew(x), lambda x: kurtosis(x), lambda x: x.value_counts().index[0]]).rename(columns={'<lambda_0>': 'skewness', '<lambda_1>': 'kurtosis', '<lambda_2>': 'mode'})

# reorder the columns

grouped_data['range'] = grouped_data['max'] - grouped_data['min']
# add a column for window values
#grouped_data['windows'] = grouped_data.index

# reorder the columns
grouped_data = grouped_data[[ 'mean', 'std', 'max', 'min', 'median', 'var', 'range', 'skewness', 'kurtosis', 'mode']]

# display the result
grouped_data = grouped_data.reset_index(drop=True)
print(grouped_data)

grouped_labels = data.groupby('window')['Movement'].agg(['mean'])
grouped_labels['windows'] = grouped_labels.index
grouped_labels = grouped_labels[['windows', 'mean']]
grouped_labels = grouped_labels.iloc[:, 1:]
grouped_labels['mean'] = grouped_labels['mean'].round()
#grouped_labels.drop(['window'])

grouped_labels = grouped_labels.reset_index(drop=True)

#grouped_data["mean"] = grouped_data["mean"].apply(lambda x: 1 if x >= 2.6 else 0)
labels = grouped_labels['mean']
print(labels)

X_train, X_test, y_train, y_test = train_test_split(grouped_data, labels, test_size=0.1, shuffle=True, random_state=0)

hf = h5py.File('hdf5_data.h5', 'w')

g1 = hf.create_group('data')
g2 = hf.create_group('Mike')
g3 = hf.create_group('Nasser')
g4 = hf.create_group('Mattew')

g1.create_dataset('training_data',data=X_train)
g1.create_dataset('testing_data', data=X_test)

g2.create_dataset('Mike_Data', data=mike_data)
g3.create_dataset('Nass_Data', data=Nass_data)
g4.create_dataset('Matt_data', data=Matt_data)

scaler = StandardScaler()

l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)

#print(y_clf_prob)

acc = accuracy_score(y_test, y_pred)
print('accuracy is:', acc)


f1 = f1_score(y_test, y_pred, average='weighted')
print(f1)

fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:,1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

auc = roc_auc_score(y_test, y_clf_prob[:,1])
print('the AUC is: ', auc)