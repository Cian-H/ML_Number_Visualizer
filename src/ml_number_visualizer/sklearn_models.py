import pickle

from loguru import logger
from sklearn.ensemble import RandomForestClassifier

from ml_number_visualizer.utils import extract_numpy_data


def train_sklearn(train_loader, val_loader, test_loader):
    logger.info("Fetching dataset for Scikit-Learn...")
    X_train, y_train = extract_numpy_data(train_loader)
    _X_val, _y_val = extract_numpy_data(val_loader)
    _X_test, _y_test = extract_numpy_data(test_loader)

    logger.info("Training Scikit-Learn Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=15)
    rf_model.fit(X_train, y_train)
    with open("./models/sklearn_rf.pkl", "wb") as f:
        pickle.dump(rf_model, f)
