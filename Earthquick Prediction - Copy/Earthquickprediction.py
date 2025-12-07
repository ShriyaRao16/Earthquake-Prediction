import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("dataset.csv")
data.columns
data = data[['Date','Time','Latitude','Longitude','Depth','Magnitude']]
print(data.head())

import datetime
import time
timestamps = []
for d, t in zip(data['Date'],data['Time']):
    try:
        ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
        timestamps.append(ts.timestamp())
    except (ValueError, OSError):
        #print( 'ValueError or OSError:')
        timestamps.append('ValueError')
timestamps=pd.Series(timestamps)
data['Timestamp']=timestamps.values
final_data=data.drop(['Date','Time'],axis=1)
final_data=final_data[final_data['Timestamp']!='ValueError']
final_data['Timestamp'] = final_data['Timestamp'].astype(float)
print(final_data.head())
from mpl_toolkits.basemap import Basemap

m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

longitudes = final_data["Longitude"].tolist()
latitudes = final_data["Latitude"].tolist()
#m = Basemap(width=12000000,height=9000000,projection='lcc',
            #resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)
x,y = m(longitudes,latitudes)

fig = plt.figure(figsize=(12,10))
plt.title("All affected areas")
m.plot(x, y, "o", markersize = 2, color = 'blue')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.show()

#splitting the Dataset
X = final_data[['Timestamp', 'Latitude', 'Longitude']]
y = final_data[['Magnitude', 'Depth']]
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from keras.models import Sequential
from keras.layers import Dense

def create_model(**kwargs):
    neurons = kwargs.get('neurons', 16)
    activation = kwargs.get('activation', 'relu')
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(3,)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(2))

    model.compile(optimizer='SGD', loss='mse', metrics=['mae'])

    return model
from scikeras.wrappers import KerasRegressor

model = KerasRegressor(build_fn=create_model, verbose=0)

# neurons = [16, 64, 128, 256]
neurons = [16]
# batch_size = [10, 20, 50, 100]
batch_size = [10]
epochs = [10]
# activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'exponential']
activation = ['sigmoid', 'relu']
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
optimizer = ['SGD', 'Adadelta']
loss = ['squared_hinge']

best_overall_score = float('-inf')
best_overall_params = None

from sklearn.model_selection import cross_val_score

for act in activation:
    for neur in neurons:
        for bs in batch_size:
            for ep in epochs:
                model = KerasRegressor(build_fn=lambda: create_model(neurons=neur, activation=act), verbose=0)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
                mean_score = scores.mean()
                
                if mean_score > best_overall_score:
                    best_overall_score = mean_score
                    best_overall_params = {'activation': act, 'neurons': neur, 'batch_size': bs, 'epochs': ep}
                
                print(f"For activation {act}, neurons {neur}, batch_size {bs}, epochs {ep}: Score: {mean_score}")

print(f"Overall Best: {best_overall_score} using {best_overall_params}")
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(2))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, batch_size=10, epochs=20, verbose=1, validation_data=(X_test, y_test))

[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {},  accuracy = {}".format(test_loss, test_acc))

import smtplib
from email.mime.text import MIMEText

def send_email(recipient, subject, message):
    sender_email = "sendersmail@gmail.com"
    sender_password = "abcd 1234"  # Use an app password for better security

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient, msg.as_string())

    print("Email sent!")

# Make predictions on test data
predictions = model.predict(X_test)

# Inverse transform predictions to get actual values
predictions = scaler_y.inverse_transform(predictions)
actual = scaler_y.inverse_transform(y_test)

# Example: Take the first prediction for demonstration
pred_magnitude, pred_depth = predictions[0]
act_magnitude, act_depth = actual[0]

# Format the email message with prediction results
message = f"Earthquake Prediction Results:\n\nPredicted Magnitude: {pred_magnitude:.2f}\nPredicted Depth: {pred_depth:.2f} km\n\nActual Magnitude: {act_magnitude:.2f}\nActual Depth: {act_depth:.2f} km\n\nModel Evaluation: Loss = {test_loss:.4f}, MAE = {test_acc:.4f}"

# Send the email with prediction results
send_email("recieversmail@gmail.com", "Earthquake Prediction ⚠️", message)


