import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# data = pd.read_csv('student-mat.csv', sep=';')
#
# print(data.head())
#
# data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#
# predict = "G3"
#
# X = np.array(data.drop([predict], 1))
# y = np.array(data[predict])
#
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
#
# linear = linear_model.LinearRegression()
#
# linear.fit(x_train, y_train)
# acc = linear.score(x_test, y_test)
# print(acc)
#
# # print('Coefficient: \n', linear.coef_)
# # print('Intercept: \n', linear.intercept_)
#
# predictions = linear.predict(x_test) #prediction is y
#
# for x in range(len(predictions)):
#     # x_test[x]
#     # print('I have predicted that the student will score', predictions[x] , 'but the actual score is', y_test[x])
#     print('I have predicted that the student will score {:.0f}'.format(predictions[x]), 'and the actual score is {:.0f}'.format(y_test[x]))

import pandas as pd
import numpy as np

np.random.seed(1)

def data():
    for i in range(len(data3)):
        if data3.loc[i, 'Calls'] <= 10:
            data2 = pd.DataFrame({
                "Calls": np.random.randint(low=1, high=10, size=10),
                "Credit bought per day": np.random.randint(low=5, high=20, size=10),
                "Time taken per call": np.random.randint(low=1, high=15, size=10),
            })
            return data2
        elif data3.loc[i, 'Calls'] <= 20:
            data2 = pd.DataFrame({
                "Calls": np.random.randint(low=11, high=20, size=10),
                "Credit bought per day": np.random.randint(low=10, high=50, size=10),
                "Time taken per call": np.random.randint(low=5, high=60, size=10),
            })
            return data2
        else:
            data2 = pd.DataFrame({
                "Calls": np.random.randint(low=21, high=50, size=10),
                "Credit bought per day": np.random.randint(low=50, high=200, size=10),
                "Time taken per call": np.random.randint(low=10, high=100, size=10),
            })
            return data2


data3 = pd.DataFrame({"Calls": np.random.randint(low=1, high=50, size=10)})

# def bundle():
#     for i in range(len(data3)):
#             if data3.loc[i, 'Calls'] > 10:
#                 data3.loc[i, 'Bundle'] = 5
#
#             data3.loc[i, 'Bundle'] = 5
#             if data3.loc[i, 'Calls'] > 20:
#                 data3.loc[i, 'Bundle'] = 20
#             if data3.loc[i, 'Calls'] < 50:
#                 data3.loc[i, 'Bundle'] = 60
# bundle()
print(data3)
print(data())


