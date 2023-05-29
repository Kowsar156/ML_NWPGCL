import pandas
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

df = pandas.read_excel("Group_1.xlsx")

X = df[['temperature_ambient', 'temperature_coolant', 'voltage_direct', 'voltage_quadrature', 'current_direct', 'current_quadrature', 'voltage_module', 'current_module']]
Y = df['temperature_stator_tooth']

train_X = X[:90]
train_Y = Y[:90]

test_X = X[90:]
test_Y = Y[90:]

model = linear_model.LinearRegression()
model.fit(train_X, train_Y)

prediction_Y = model.predict(test_X)
mse = mean_squared_error(test_Y, prediction_Y)
print("Mean Squared Error: ", mse)

prediction = model.predict([[-0.603,-0.393,-0.359,-0.235,0.0834,0.231,1.26,1.19]])
print(prediction)