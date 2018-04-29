# New Customer Prediction

from sklearn.externals import joblib

new_data = []
new_data.append(int(input("Is the customer from Spain? 0 for no, 1 for yes: ")))
new_data.append(int(input("Is the customer from Germany? 0 for no, 1 for yes: ")))
new_data.append(int(input("Enter the customer's credit score: ")))
new_data.append(int(input("What is the customer's gender? 0 for Female, 1 for Male: ")))
new_data.append(int(input("Enter the customer's age: ")))
new_data.append(int(input("Enter the customer's tenure (in years): ")))
new_data.append(float(input("Enter the customer's balance: ")))
new_data.append(int(input("Enter the customer's number of products: ")))
new_data.append(int(input("Does the customer have a credit card? 0 for no, 1 for yes: ")))
new_data.append(int(input("Is the customer an active member? 0 for no, 1 for yes: ")))
new_data.append(int(input("Enter the customer's estimated salary: ")))

scaler_filename = input("\nEnter filename of scaler to load (required): ")
sc = joblib.load(scaler_filename)
new_data = sc.transform([new_data])

from keras import models

trained_filename = input("Enter filename of trained model to load (required): ")
classifier = models.load_model(trained_filename)
print("The predicted probability of the customer leaving is {}%".format(classifier.predict(new_data)[0][0]*100))
