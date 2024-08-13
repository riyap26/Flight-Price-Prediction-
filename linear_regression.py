import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt

time_list = []
accuracy_list = []
for i in range(1000, 100000, 1000) :
    strart_time = time.time()
    data = pd.read_csv('test.csv', nrows=i)

    # Step 2: Define features and the target variable
    X = data.drop(columns=['baseFare', 'legId', 'fareBasisCode', 'totalFare', 'segmentsDepartureTimeEpochSeconds', 'segmentsDepartureTimeRaw', 'segmentsArrivalTimeEpochSeconds', 'segmentsArrivalTimeRaw', 'segmentsArrivalAirportCode', 'segmentsDepartureAirportCode', 'segmentsAirlineName', 'segmentsAirlineCode', 'segmentsEquipmentDescription', 'segmentsDurationInSeconds', 'segmentsDistance', 'segmentsCabinCode'])
    y = data['baseFare']

    # Step 3: Preprocess the data
    categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numeric_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ])

    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Implement the KNN regression algorithm
    knn = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors as needed

    # Preprocess and fit the model
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    knn.fit(X_train_preprocessed, y_train)

    # Step 6: Make predictions
    y_pred = knn.predict(X_test_preprocessed)

    # Step 7: Evaluate the model
    r2 = r2_score(y_test, y_pred)
    accuracy_percentage = r2 * 100
    end_time = time.time()
    print(f'Accuracy Percentage: {accuracy_percentage:.2f}% ' f"Runtime of the program is {end_time - strart_time} " f"Num {i}")
    time_list.append(i)
    accuracy_list.append(accuracy_percentage)

plt.plot(time_list, accuracy_list)
plt.xlabel('Time (seconds)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Time')
plt.show()
