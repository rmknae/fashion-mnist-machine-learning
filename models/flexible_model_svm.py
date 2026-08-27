''' Author:2023-EE-5 '''
''' FLEXIBLE MODEL: SVM '''

'''
Data Loading and Preprocessing
'''

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load full 70000 images
print("Loading Fashion-MNIST...")
fashion = fetch_openml('Fashion-MNIST',version=1,as_frame=False)

X = fashion.data        
y = fashion.target.astype(int)  

print(f"Full dataset: {X.shape}")
print(f"Labels: {y.shape}")

# Class names
class_names = [
    'T-shirt', 'Trouser', 'Pullover',
    'Dress',   'Coat',    'Sandal',
    'Shirt',   'Sneaker', 'Bag',
    'Ankle Boot'
]

# Split 80/20 :gives 56000 dev and 14000 test
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nDev set:  {X_dev.shape}")
print(f"Test set: {X_test.shape}")
print(f"y_dev:    {y_dev.shape}")
print(f"y_test:   {y_test.shape}")


# Normalize (Preprocessing)
X_dev_flat  = X_dev  / 255.0
X_test_flat = X_test / 255.0

print(f"Dev flat:  {X_dev_flat.shape}")
print(f"Test flat: {X_test_flat.shape}")

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("Training SVM...")

'''
SVC() creates a Support Vector Machine classifier.
Learned from: Scikit-learn SVC API reference:all parameters explained
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

PARAMETER CHOICES:

kernel='rbf':
RBF maps data into a higher-dimensional space so the SVM can draw non-linear
boundaries between classes. I chose RBF because Fashion-MNIST classes like
Shirt/Coat/Pullover cannot be separated with a straight line they 
need curved boundaries only RBF can produce.

C=10:
Learned from: Scikit-learn: C parameter and regularization tradeoff
https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use

gamma='scale':
Gamma controls how far one training sample's influence reaches.
'scale' automatically sets gamma = 1 / (n_features * X.var()),
which adjusts to the actual spread of the data. I chose 'scale' because 
the scikit-learn docs recommend it as the safer default over 
'auto' since it accounts for feature variance.
Important after normalizing pixels to 0-1.

decision_function_shape='ovr':
OVR = One-vs-Rest: trains 10 binary classifiers (one per class),
then picks the class with the highest confidence score.
I chose OVR because it is simpler to interpret
each classifier asks "is this class X or not?", which maps
cleanly to Fashion-MNIST's 10 independent clothing categories.
'''
svm = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    decision_function_shape='ovr',
    random_state=42
)

# STEP 3 : Train the model

'''
fit() finds the optimal hyperplane (support vectors) that
best separates each class from the rest.
X_dev_flat = flattened pixel features (n_samples, 784)
y_dev      = correct class labels

Learned from: Scikit-learn SVM :how fit() works with RBF kernel
https://scikit-learn.org/stable/modules/svm.html#svc
'''
svm.fit(X_dev_flat, y_dev)
print("Training done")


# STEP 4 :Make predictions on test data

'''
predict() applies the learned decision boundaries to unseen images.
Returns the predicted class label for each test sample.

Learned from: Scikit-learn SVC API :predict method
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict
'''
svm_pred = svm.predict(X_test_flat)


# STEP 5 :Evaluate model performance

'''
accuracy_score() = correct predictions / total predictions.
I used this as the first quick check it tells me overall 
how often the model is right.

Learned from: Scikit-learn:accuracy score definition
https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
'''
svm_acc = accuracy_score(y_test, svm_pred)

'''
f1_score() with average='macro'
Computes F1 per class then averages equally across all 10 classes.
I chose macro (not weighted) because each clothing
category has equal support (1400 samples each), so macro and
weighted give the same result and macro is more standard to report.

Learned from: Scikit-learn : macro vs weighted F1 averaging
https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification
'''
svm_f1 = f1_score(
    y_test,
    svm_pred,
    average='macro'
)

# STEP 6 : Cross Validation

'''
cross_val_score() splits the data into cv=5 folds,
trains on 4 and tests on 1 each time, repeating 5 times.
As a student I used this to check if my C=10 result was
consistent or just lucky on one particular split.
A small CV std (like 0.003) confirms the model is stable.

Learned from: Scikit-learn : cross_val_score and k-fold CV
https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics
'''
svm_cv = cross_val_score(
    svm,                # model
    X_dev_flat,         # input features
    y_dev,              # labels
    cv=5,               # 5-fold cross validation
    scoring='accuracy'  # evaluation metric
)


# STEP 7 :Print final results

print(f"\n=== SVM RESULTS ===")

'''
Overall test accuracy
'''
print(f"Accuracy:  {svm_acc:.4f}")

'''
Macro F1-score across all 10 classes
'''
print(f"Macro F1:  {svm_f1:.4f}")

'''
Mean accuracy across all 5 CV folds
'''
print(f"CV Mean:   {svm_cv.mean():.4f}")
print(f"CV Std:    {svm_cv.std():.4f}")


# STEP 8 :Detailed per-class evaluation

print("\nPer class:")

print(
    classification_report(
        y_test,
        svm_pred,
        target_names=class_names
    )
)