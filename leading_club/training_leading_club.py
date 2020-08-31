print("process started looding .....")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import os

#splitting the data
from sklearn.model_selection import train_test_split

#scalling the data
from sklearn.preprocessing import MinMaxScaler

#tencerflow import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

#to stop over fitting
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.metrics import classification_report,confusion_matrix

"""
feature engeneering
"""

df = pd.read_csv('./data/lending_club_loan_two.csv')

df = df.drop('emp_length',axis=1)
df = df.drop('emp_title',axis=1)
df = df.drop('title', axis=1)
df = df.drop('grade', axis=1)
df = df.drop('issue_d', axis=1)

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']

def fill_mod_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x:fill_mod_acc(x['total_acc'],x['mort_acc']),axis=1)

df = df.dropna()

df['term'] = df['term'].apply(lambda term: int(term[:3]))

df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')
df['address'] = df['address'].apply(lambda address: int(address[-5:]))
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda earliest_cr_line: int(earliest_cr_line[-4:]))

dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose',
                             'sub_grade','home_ownership','address','earliest_cr_line']], drop_first=True)
df = pd.concat([df.drop(['verification_status', 'application_type','initial_list_status','purpose','sub_grade',
                         'home_ownership','address','earliest_cr_line'],axis=1), dummies],axis=1)

X = df.drop('loan_status', axis=1).values
df['loan_status'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
Y = df['loan_status'].values

#splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,random_state=101)

#scalling the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#3d finaly output
model = Sequential()
model.add(Dense(72, activation='relu'))
#dropping the nurons randmoly
model.add(Dropout(0.5))
model.add(Dense(36, activation='relu'))
#dropoutlayer randomly
model.add(Dropout(0.5))
model.add(Dense(17, activation='relu'))
#dropoutlayer randomly
model.add(Dropout(0.5))
#BINARY CLASSIFICATION
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=25)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

userinput = int(input("----------------------------------------------------------------------------\n"
                          "use the below command to build or load your model \n"
                          "1. build your model (recommended if your are running for the first-time) \n"
                          "2. load your model (recommended if already building is done) \n"
                          "----------------------------------------------------------------------------\n"
                          "Answer : "))

if userinput == 1:
    model.fit(x=X_train, y=Y_train, epochs=10, validation_data=(X_test,Y_test), callbacks=[early_stop,cp_callback])
    loss = pd.DataFrame(model.history.history)
    loss.plot()
    plt.show()
elif userinput == 2:
    model.load_weights(checkpoint_path)
    prediction_model = model.predict_classes(X_test)
    print(classification_report(Y_test, prediction_model))
    print(confusion_matrix(Y_test, prediction_model))
else:
    print("choose correction option ")

