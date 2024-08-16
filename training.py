import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from utils import create_dictionary, create_features, process_text

# Hyperparameters
TEST_SIZE = 0.2
RAMDOM_SEED = 42

# Load data
DATA_PATH = "data\\2cls_spam_text_cls.csv"
df = pd.read_csv(DATA_PATH)

# Preprocessing
messages = df["Message"].values.tolist()
labels = df["Category"].values.tolist()

messages = [process_text(message) for message in messages]
dictionary = create_dictionary(messages)
features = np.array([create_features(message, dictionary)
                    for message in messages])

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=TEST_SIZE, random_state=RAMDOM_SEED)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Save model
if not os.path.exists("models"):
    os.makedirs("models")

with open("models/model.pkl", "wb") as f:
    pickle.dump(gnb, f)

# Save dictionary
with open("models/dictionary.pkl", "wb") as f:
    pickle.dump(dictionary, f)
