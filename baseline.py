#! /usr/bin/env python

import tensorflow as tf
import data_helpers
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn import metrics
# Parameters
from sklearn.cross_validation import train_test_split
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(x_text,
y)
# Build vocabulary
yy=[]
for i in y:
    if i[0]==0:
        yy.append(1)
    if i[0]==1:
        yy.append(0)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(x_text,
yy)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
xgbc=XGBClassifier()
xgbc.fit(X_train,y_train)
pres=xgbc.predict(X_test)
print metrics.accuracy_score(y_test, pres)

