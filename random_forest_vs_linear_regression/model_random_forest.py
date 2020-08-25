import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# For splitting dataset
from sklearn.model_selection import ShuffleSplit, train_test_split, cross_val_score

# Import sklearn models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

Forest = RandomForestRegressor(random_state=0, max_depth=20, n_estimators=150)
lr = LinearRegression()


def rmsle(y, y_):
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")

train_data["year"] = train_data.datetime.apply(lambda x: x.split()[0].split("-")[0])
train_data["month"] = train_data.datetime.apply(lambda x: x.split()[0].split("-")[1])
train_data["day"] = train_data.datetime.apply(lambda x: x.split()[0].split("-")[2])
train_data["hour"] = train_data.datetime.apply(lambda x: x.split()[1].split(":")[0])
train_data = train_data.drop('datetime', axis=1)

test_data["year"] = test_data.datetime.apply(lambda x: x.split()[0].split("-")[0])
test_data["month"] = test_data.datetime.apply(lambda x: x.split()[0].split("-")[1])
test_data["day"] = test_data.datetime.apply(lambda x: x.split()[0].split("-")[2])
test_data["hour"] = test_data.datetime.apply(lambda x: x.split()[1].split(":")[0])
test_features = test_data.drop('datetime', axis=1)

df = pd.DataFrame()
df['c'] = train_data['count']
df['hour'] = train_data['hour']

target = train_data['count']
features = train_data.drop(['casual', 'registered', 'count'], axis=1)

scaled_features = MinMaxScaler().fit_transform(features)
scaled_test_features = MinMaxScaler().fit_transform(test_features)

X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.5, random_state=42)


def train_and_predict_model(model, model_name, X_train, X_test, y_train, y_test, selected_cols):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_score = r2_score(y_train, train_pred)

    test_score = r2_score(y_test, test_pred)

    root_mean_squared_log_error = rmsle(y_test, test_pred)

    print("r2_score des données de train de {} = {}".format(model_name, train_score))
    print("r2_score des données de tests {} = {}".format(model_name, test_score))
    print("Root Mean Squared Log Error de {} = {}".format(model_name, root_mean_squared_log_error))
    print("cross_val_score de {} = {}".format(model_name, cross_val_score(model, features, target, cv=10).mean()))

    plt.scatter(y_test, test_pred)
    plt.title("Nb de vélo loué en fonction de la prédiction (set test)")
    plt.legend(model_name)
    plt.show()

    plt.scatter(y_train, train_pred)
    plt.title("Nb de vélo loué en fonction de la prédiction (set train)")
    plt.legend(model_name)
    plt.show()

    return model


# total_features = list(features.columns)
selected_features = ['hour', 'temp', 'workingday', 'hum']

model = train_and_predict_model(Forest, 'Forest', X_train, X_test, y_train, y_test, selected_features)
model = train_and_predict_model(lr, 'LinearRegression', X_train, X_test, y_train, y_test, selected_features)
