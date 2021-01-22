import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def trainBestXGBmodel(X, y):
    # seperating data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    num_boost_round = 999
    # Hyperparameter Selection/Optimization Required, default values below
    params = {
        # Parameters that we are going to tune, these are defaults
        'max_depth': 6,
        # maximum number of nodes allowed from the root to the farthest leaf of a tree. , the higher the more likely we overfit
        'min_child_weight': 1,
        # is the minimum weight  required in order to create a new node in the tree. A smaller min_child_weight is more likely to overfit.
        'eta': .3,
        # the shrinkage of the weights associated to features after each round, in other words it defines the amount of "correction" we make at each step
        'subsample': 1,
        # the fraction of observations (the rows) to subsample at each step. By default it is set to 1 meaning that we use all rows.
        'colsample_bytree': 1,
        # the fraction of features (the columns) to use. By default it is set to 1 meaning that we will use all features.
        # Other parameters
        'objective': 'reg:squarederror',
    }

    # setting the evaluation metric to mean squared error
    params['eval_metric'] = "rmse"


    # setting up model hyperparameter crossvalidation
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=123,
        nfold=5,
        metrics={'rmse'},
        early_stopping_rounds=10
    )
    #testing default hyperparameters
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        early_stopping_rounds=10  # number of rounds of testing without improvement
    )

    ParamsBestScore = model.best_score
    ParamsBestNumOfBoostRounds = model.best_iteration + 1

    print(ParamsBestScore)
    print(ParamsBestNumOfBoostRounds)
    # setting up intervals for testing different values of maxdepth and maxchild in range

    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(6, 16)
        for min_child_weight in range(1, 10)
    ]

    # search for initial best hyperparameters and rmse score

    # Define initial best params and MSE
    min_rmse = float("Inf")
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(
            max_depth,
            min_child_weight))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=123,
            nfold=5,
            metrics={'rmse'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_rmse = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (max_depth, min_child_weight)
        params['max_depth'] = best_params[0]
        params['min_child_weight'] = best_params[1]
    print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))

    gridsearch_params = [
        (subsample)
        for subsample in [i / 10. for i in range(1, 11)] #bound [0,1]
    ]

    min_rmse = float("Inf")
    best_params = None
    for subsample in reversed(gridsearch_params):
        print("CV with subsample={}".format(
            subsample,
            ))
        # We update our parameters
        params['subsample'] = subsample

        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=123,
            nfold=5,
            metrics={'rmse'},
            early_stopping_rounds=10
        )
        # Update best score
        mean_rmse = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        print("\tMSE {} for {} rounds".format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (subsample)
        params['subsample'] = best_params

    print("Best params: {}, RMSE: {}".format(best_params, min_rmse))

    min_rmse = float("Inf")
    best_params = None
    for eta in [.3, .2, .1, .05, .01, .005]:
        print("CV with eta={}".format(eta))
        # We update our parameters
        params['eta'] = eta
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=123,
            nfold=5,
            metrics=['rmse'],
            early_stopping_rounds=10
        )
        # Update best score
        mean_rmse = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        print("\trmse {} for {} rounds\n".format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = eta
        params['eta'] = best_params
    print("Best params: {}, rmse: {}".format(best_params, min_rmse))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        early_stopping_rounds=10  # number of rounds of testing without improvement
    )

    ParamsBestScore = model.best_score
    ParamsBestNumOfBoostRounds = model.best_iteration + 1

    print(ParamsBestScore)
    print(ParamsBestNumOfBoostRounds)

    num_boost_round = model.best_iteration + 1
    best_model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")]
    )
    best_model.save_model("best.model")
    return params

def importantFeatures(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    loaded_model = xgb.Booster()
    loaded_model.load_model("best.model")

    importance = loaded_model.get_score(importance_type='gain')
    for i in range (0, len(importance.keys())):
        importance[X.columns[i]] = importance[f'f{i}']
        del importance[f'f{i}']

    important_features = pd.DataFrame({
        'Feature': importance.keys(),
        'Importance': importance.values()}) \
        .sort_values('Importance', ascending=True)

    plt.figure(figsize=(18, 9))
    plt.style.use('fivethirtyeight')
    plt.barh(important_features.Feature, important_features.Importance)
    plt.barh(important_features.Feature, important_features.Importance, color="#e377c2")
    plt.suptitle('Feature Important for BTCUSDT using Gradient Boosting')
    plt.xlabel('Feature Importance', fontsize=13)
    plt.ylabel('Feature')
    plt.savefig("importance.jpg")
    plt.show()


def predictReturn(data):
    loaded_model = xgb.Booster()
    loaded_model.load_model("best.model")
    return loaded_model.predict(data)


