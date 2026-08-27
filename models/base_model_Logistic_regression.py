''' Author:2023-EE-5   ''' 
'''  BASELINE MODEL: LR   ''' 
        
''' 
Data Loading and Preprocessing
sklearn used only for dataloading, all other logic implemented from scratch
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

#From scratch no built in libraries are used
import numpy as np

class LogisticRegression:
 
    def __init__(self):
        
        ''' 
        Setting up the weights. W is 784x10 since we have 784 pixels and 
        10 clothing classes in Fashion-MNIST. The score function is just f = Wx +b
        Took Help from: https://cs231n.github.io/linear-classify/#score-function
        ''' 
        self.W = None
        
        ''' 
        Adding bias to shift the decision boundaries.This helps the model fit 
        better without relying entirely on the input features.
        Learned from: 5:role of bias
        https://www.deeplearningbook.org/contents/ml.html
        ''' 
        self.b = None
 
    def softmax(self, z):
 
        ''' 
        I spent so long debugging NaN values before I realized I needed to subtract 
        the max value from the scores. This trick stops the exponentials
        causing an overflow error. softmax(z) == softmax(z - max(z))
        Took help from: https://cs231n.github.io/linear-classify/#softmax
        '''
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        
        # σ(z)_i = exp(z_i) / Σ exp(z_j)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
 
    def train(self, X, y, lr=0.1, epochs=500):
 
        n = X.shape[0]
        y = y.astype(int)
 
        ''' 
        Initializing weights to zero here so I can see the gradient updates 
        do all the heavy lifting from scratch. 
        Took help from (weight initialization section):
        https://cs231n.github.io/neural-networks-2/#init
        '''
        self.W = np.zeros((784, 10)) 
        self.b = np.zeros(10)
 
        for epoch in range(epochs):
            
            ''' 
            Calculating the raw scores for all samples at once. Matrix 
            multiplication saves so much time here compared to writing out loops.
            Learned from: CS231n score function f(x; W, b) = Wx + b
            https://cs231n.github.io/linear-classify/#score-function
            ''' 
            scores = X @ self.W + self.b
 
            probs = self.softmax(scores)
 
            grad = probs.copy()
 
            '''
            To get the full gradient of the loss, 
            you just subtract 1 from the probability of the correct class. 
            d(Loss)/d(score_correct) = p_correct - 1 (softmax + cross-entropy combined)
            https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            '''
            grad[np.arange(n), y] -= 1
 
            '''
            dW = X^T · grad / n — gradient of loss w.r.t. weights, averaged over batch
            Learned from: CS231n optimization: weight update rule W += -lr * dW
            https://cs231n.github.io/optimization-1/
            '''
            dW = X.T @ grad / n
            db = np.mean(grad, axis=0)
 
            '''
            Taking the actual gradient descent step.
            '''
            self.W -= lr * dW
            self.b -= lr * db
 
        print("Training Complete!")
 
 

# PREDICT FUNCTION

    def predict(self, X):
 
        scores = X @ self.W + self.b
        probs  = self.softmax(scores)
 
        '''
        Using argmax to just pick the class with the highest probability. 
        https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        '''
        return np.argmax(probs, axis=1)
 
 

# ACCURACY FUNCTION
 
def get_accuracy(y_true, y_pred):
 
    '''
    I picked up: comparing the arrays gives booleans, and taking 
    the mean directly gives the fraction of correct predictions
    '''
    return np.mean(y_true == y_pred)
 
 

# F1 SCORE FUNCTION

 
def get_f1(y_true, y_pred):
 
    all_scores = []
 
    for c in range(10):
 
        '''
        Manually calculating True Positives, False Positives, and False Negatives. 
        '''
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
 
        '''
        Added small if-statements to prevent division by zero errors when 
        precision or recall is 0.
        '''
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall    = tp / (tp + fn) if tp + fn > 0 else 0
 
        '''
        Calculating the F1 score using the harmonic mean. 
        Learned from: Wikipedia F-score :derivation of harmonic mean formula
        https://en.wikipedia.org/wiki/F-score
        '''
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
 
        all_scores.append(f1)
 
    '''
    Averaging the F1 scores across all 10 classes to get the Macro F1. 
    This treats every class equally, whether it's shirts or sneakers.
    Learned from: Scikit-learn multiclass metrics: macro vs micro vs weighted averaging
    https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification
    '''
    macro_f1 = np.mean(all_scores)
 
    return macro_f1, all_scores
 
 
# CLASSIFICATION REPORT

 
def print_report(y_true, y_pred):
 
    '''
    Simple printing
    '''
 
    macro_f1, f1_scores = get_f1(y_true, y_pred)
 
    # Collect per-class precision, recall, f1, support
    precisions = []
    recalls    = []
    supports   = []
 
    for c in range(10):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
 
        precisions.append(tp / (tp + fp) if tp + fp > 0 else 0)
        recalls.append(   tp / (tp + fn) if tp + fn > 0 else 0)
        supports.append(  int(np.sum(y_true == c)))
 
    total     = len(y_true)
    accuracy  = get_accuracy(y_true, y_pred)
 
    macro_p   = np.mean(precisions)
    macro_r   = np.mean(recalls)
    w_p       = np.sum([precisions[c] * supports[c] for c in range(10)]) / total
    w_r       = np.sum([recalls[c]    * supports[c] for c in range(10)]) / total
    w_f1      = np.sum([f1_scores[c]  * supports[c] for c in range(10)]) / total
 
    # Header
    print("\n=== LOGISTIC REGRESSION RESULTS ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
 
    # Per-class table — same column layout as SVM output
    print(f"\nPer class:\n")
    print(f"{'':>15}{'precision':>12}{'recall':>9}{'f1-score':>10}{'support':>10}")
    print()
    for i, name in enumerate(class_names):
        print(f"{name:>15}{precisions[i]:>12.2f}{recalls[i]:>9.2f}{f1_scores[i]:>10.2f}{supports[i]:>10}")
 
    print()
    print(f"{'accuracy':>15}{'':>12}{'':>9}{accuracy:>10.2f}{total:>10}")
    print(f"{'macro avg':>15}{macro_p:>12.2f}{macro_r:>9.2f}{macro_f1:>10.2f}{total:>10}")
    print(f"{'weighted avg':>15}{w_p:>12.2f}{w_r:>9.2f}{w_f1:>10.2f}{total:>10}")
 
    return f1_scores
 
 
# CROSS VALIDATION (K-Fold)

 
def cross_validation(X, y, k=5):
 
    '''
    Splitting the data for k-fold cross-validation. Holding out one chunk 
    at a time to make sure the model isn't just memorizing the training set.
    Learned from: Scikit-learn K-Fold guide
    https://scikit-learn.org/stable/modules/cross_validation.html#k-fold
    '''
    fold_size = len(X) // k
 
    acc_scores = []
 
    for i in range(k):
 
        print(f"\nFold {i+1}")
 
        start = i * fold_size
        end   = (i + 1) * fold_size
 
        X_val  = X[start:end]
        y_val  = y[start:end]

        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])
 
        model = LogisticRegression()
 
        '''
        I experimented with the epochs here. 300 was too little and the 
        accuracy was still increasing, but going past 500 just wasted time 
        without much gain.Took help from:
        https://cs231n.github.io/neural-networks-3/#babysit
        '''
        model.train(X_train, y_train, lr=0.1, epochs=500)
 
        pred = model.predict(X_val)
        acc  = get_accuracy(y_val, pred)
 
        print(f"Accuracy = {acc:.4f}")
        acc_scores.append(acc)
 
    mean_acc = np.mean(acc_scores)
 
    '''
    Checking the standard deviation to see how stable the model is 
    across the different data splits.
    Learned from: Scikit-learn CV:why reporting mean ± std is standard practice
    https://scikit-learn.org/stable/modules/cross_validation.html
    '''
    std_acc = np.std(acc_scores)
 
    return mean_acc, std_acc
 

# TRAIN MODEL

 
print("Training Model...")
 
model = LogisticRegression()
 
'''
Learned from: CS231n hyperparameter tuning :lr too high = diverge, too low = too slow
https://cs231n.github.io/neural-networks-3/#hyper
'''
 
model.train(X_dev_flat, y_dev, lr=0.1, epochs=500)
 
print("Model Trained Successfully!")
 
 
# TEST MODEL
 
y_true = y_test.astype(int)
y_pred = model.predict(X_test_flat)
 
accuracy         = get_accuracy(y_true, y_pred)
macro_f1, all_f1 = get_f1(y_true, y_pred)
 
 
# CROSS VALIDATION
 
print("\nRunning Cross Validation...")
 
'''
Going with k=5 folds since that seems to be the standard choice 
in most scikit-learn tutorials I read.Learned from: Scikit-learn CV guide: 
k=5 or k=10 are standard choices
https://scikit-learn.org/stable/modules/cross_validation.html
'''
cv_mean, cv_std = cross_validation(
    X_dev_flat,
    y_dev.astype(int)
)
 
 
all_f1 = print_report(y_true, y_pred)
 
print(f"CV Mean:   {cv_mean:.4f}")
print(f"CV Std:    {cv_std:.4f}")