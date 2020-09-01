print("program started .......\n"
      "imports in process ...")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

print("reading data .......")
df = pd.read_csv('./data/kc_house_data.csv')
print("done :)")

print("feature engineering .......")
df = df.drop('id',axis=1)
df = df.sort_values('price',ascending=False).iloc[216:]
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda date:date.year)
df['month'] = df['date'].apply(lambda date:date.month)
df = df.drop('date',axis=1)
df = df.drop('zipcode', axis=1)
X = df.drop('price', axis=1).values
Y = df['price'].values
print("done :)")

print("splitting data .......")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=101)
print("done :)")

print("scaling the data ..... ")
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
print("done :)")

print("creating model")
models = Sequential()
models.add(Dense(19, activation='relu'))
models.add(Dense(19, activation='relu'))
models.add(Dense(19, activation='relu'))
models.add(Dense(19, activation='relu'))
models.add(Dense(1))
models.compile(optimizer='adam', loss='mse')

earlystop = EarlyStopping(monitor='val_loss',patience=2)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
print("done :)")

userinput = int(input("----------------------------------------------------------------------------\n"
                          "use the below command to build or load your model \n"
                      "note \n"
                          "1. build your model (recommended if your are running for the first-time) \n"
                          "2. load your model (recommended if already building is done) \n"
                          "----------------------------------------------------------------------------\n"
                          "Answer : "))

if userinput == 1:
    print("training started")
    models.fit(x=X_train,y=Y_train,validation_data=(X_test,Y_test),
               batch_size=128, epochs=600,callbacks=[cp_callback,earlystop])
    predictions = models.predict(X_test)
    loss_df = pd.DataFrame(models.history.history)
    print(np.sqrt(mean_squared_error(Y_test, predictions)))
    print(mean_absolute_error(Y_test, predictions))
    print(explained_variance_score(Y_test, predictions))
    plt.scatter(predictions,Y_test), plt.plot(Y_test,Y_test,'r')
    plt.show()

elif userinput == 2:
    models.load_weights(checkpoint_path)
    print("This is the real-estate data of the 30000+ houses in USA with its price and features of the house\n"
    "The data has been analysed and ready for a prediction u can use the below code to predict \n"
    "a house price of your wish\n "
    "NOTE:\n"
    "1. GIVE A NUMBER OF YOUR CHOISE RANGE 1 to 20000 \n "
          "2. the data is from the data folder cross check the data for better understanding\n"
          "--------------------------------------------------------------------------------------\n")
    for i in range(10):
        userinput = int(input("select a new house range(0:20000):"))
        single_house = df["price"].values[userinput]

        single_houses = df.drop('price',axis=1).iloc[userinput]
        single_houses = scaler.transform(single_houses.values.reshape(-1,19))
        print("----------------------------------------------------------------------------\n"
              "this is the original price of the house: "+str(df["price"].values[1])+"\n"
                "there are the bedrooms: "+str(df["bedrooms"].values[userinput])+"\n"
                "this is the prediction price: "+str(models.predict(single_houses)[0][0])+
              "\n----------------------------------------------------------------------------\n")


