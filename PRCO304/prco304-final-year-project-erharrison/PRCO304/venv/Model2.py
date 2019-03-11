import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

seed = 10
numpy.random.seed(seed) # for reproducibility

scaler = StandardScaler() # scaling beacuse mlp is sensitive to feature scaling

dataset = numpy.loadtxt(r"C:\Users\emily\Documents\GitHub\prco304-final-year-project-erharrison\PRCO304\prco304-final-year-project-erharrison\Data.csv", delimiter=",")
inputX, outputY = dataset[:,0:78], dataset[:128]

X_train, X_test, y_train, y_test = train_test_split(inputX, outputY)

scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(77,15,77),max_iter=500)

mlp.fit(X_train,y_train)

MLPClassifier(activation='relu', alpha=0.0001, batch_size='len(X_test)', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(77,15,77), learning_rate='constant',
              learning_rate_init=0.001, max_iter=500, momentum=0.9,
              nesterovs_momentum=True, power_t=0.5, random_state=None,
              shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
              verbose=2, warm_start=False)



# calculate predictions
predictions = mlp.predict(X_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))