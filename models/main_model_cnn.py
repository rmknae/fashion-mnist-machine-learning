# ===================================================
# CNN FROM SCRATCH
# Author : Member 1  Rameen (2023-EE-3)
# Rameen (2023-EE-3)
# ===================================================
#   1. Two conv layers are used in this code
#   2. 16 filters per convolution layer are used      
#   3. Momentum SGD is used because it has faster convergence
#   4. Learning rate decay because fine-tunes near end
#   5. Dropout on FC1  
#   6. 40 training epochs were used
#
# Architecture:
#   Input (N, 28, 28, 1)
#   -> Conv1 (16 filters, 3x3) -> ReLU -> MaxPool(2x2)  shape: (N,13,13,16)
#   -> Conv2 (16 filters, 3x3) -> ReLU                  shape: (N,11,11,16)
#   -> Flatten                                           shape: (N,1936)
#   -> FC1 (256 units) -> ReLU -> Dropout
#   -> FC2 (10 units)  -> Softmax
# Flatten then normalize

# Run this first
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("All imports done!")

# Load full 70000 images
print("Loading Fashion-MNIST...")
fashion = fetch_openml('Fashion-MNIST',
                        version=1,
                        as_frame=False)

X = fashion.data        # 70000 x 784
y = fashion.target.astype(int)  # 70000 labels

print(f"Full dataset: {X.shape}")
print(f"Labels: {y.shape}")

# Class names
class_names = [
    'T-shirt', 'Trouser', 'Pullover',
    'Dress',   'Coat',    'Sandal',
    'Shirt',   'Sneaker', 'Bag',
    'Ankle Boot'
]

# SPLITTING INTO DEV AND TEST SETS
# I split the data into 80% development and 20% test
# This gives 56,000 images for training/validation and 14,000 for final testing
# random_state=42 makes sure I get the same split every time I run the code
# stratify=y makes sure each class gets the same proportion in both sets

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


# NORMALIZING THE PIXEL VALUES
# Original pixel values go from 0 to 255
# I divide by 255 so all values are now between 0.0 and 1.0
# This is important because:
#    Large input numbers make gradient descent behave badly
#    The network learns much faster when inputs are on the same small scale

X_dev_flat  = X_dev  / 255.0
X_test_flat = X_test / 255.0
print(X_dev_flat.shape)   # should be (N, 784)
print(X_dev_flat.max())   # should be 1.0
print(X_dev_flat.min())  # should be 0
print(f"Dev flat:  {X_dev_flat.shape}")
print(f"Test flat: {X_test_flat.shape}")
print(f"Min: {X_dev_flat.min():.1f}")
print(f"Max: {X_dev_flat.max():.1f}")
print("Ready to train")


# ===================================================
# STEP 1 — ReLU Activation Function
# ===================================================
# ReLU stands for Rectified Linear Unit
# The rule is very simple: if the value is positive, keep it. If negative, make it 0.
#
# Why I chose ReLU instead of sigmoid or tanh:
#   - Sigmoid and tanh both squish values into a small range (0-1 or -1 to 1)
#   - When we send gradients backwards through many layers, they keep getting smaller
#   - This is called the vanishing gradient problem as early layers stop learning
#   - ReLU does not squish positive values, so gradients flow properly
#   - I watched YouTube video series and read cs231n to confirm this choice
#
# Reference: https://cs231n.github.io/neural-networks-1/#actfun

def relu(x):
    return np.maximum(0, x)


# WHY we do this backward step:
# In backpropagation, we need to send the gradient back through ReLU
# so earlier layers can learn from the error.
# ReLU rule:
#   If input > 0 → output = input (neuron is active, so gradient flows)
#   If input <= 0 → output = 0 (neuron is inactive, so no learning happens there)
# WHY we block gradient when input <= 0:
# Because those neurons did not contribute to the forward output
# This helps the model focus only on useful activated features.
# Reference: https://cs231n.github.io/optimization-2/

def relu_backward(d_out, cache_x):
    d_x = d_out.copy()
    d_x[cache_x <= 0] = 0
    return d_x


# ===================================================
# STEP 2 — Softmax Function
# ===================================================
# Softmax converts the raw output scores of the network into probabilities
# All 10 outputs become positive numbers that add up to exactly 1.0
# The class with the highest probability is the model's prediction
#
# Problem I ran into:
#   When I first wrote this, my code was giving NaN (not a number) for every prediction
#   The reason was that np.exp() makes very large numbers when input is large
#   For example np.exp(1000) gives infinity, and infinity/infinity = NaN
#
# Fix I used:
#   Before applying exp(), I subtract the maximum value in each row
#   This makes all values <= 0, so exp() gives at most 1.0
#   The final probabilities stay exactly the same mathematically, but no overflow happens
#   I found this fix by researching online and asking ChatGPT to explain it
#
# Reference: softmax was also explained in class lectures
 
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ===================================================
# STEP 3 — Dropout (Regularization)
# ===================================================
# learnt that we also need to do regularization in cnn from youtube video series. 
# and confirmed how to do that from the cite mentioned 

# Dropout is a regularization technique used to reduce overfitting.
# During training, it randomly deactivates neurons with probability p,
# forcing the network to learn more robust and distributed features
# instead of relying on specific neurons.
#
# It works like training many smaller subnetworks together, which improves
# generalization similar to L1/L2 regularization but by randomly removing
# activations instead of penalizing weights.
#
# During testing, all neurons are active and outputs are scaled to match
# training-time expectations.
# Ref: https://cs231n.github.io/neural-networks-2/#reg

def dropout_forward(x, p=0.5, train=True):
    if not train:
        # At test time: scale down to match training expectation
        return x * (1 - p), None
    # Randomly keep each neuron with probability (1-p)
    mask = (np.random.rand(*x.shape) > p).astype(float)
    # Inverted dropout: divide by (1-p) so we don't need to scale at test time
    mask /= (1 - p)
    return x * mask, mask


def dropout_backward(d_out, mask):
    # Gradient only flows through neurons that weren't dropped
    return d_out * mask


# ===================================================
# STEP 4 — Convolutional Layer (Forward Pass)
# ===================================================
# A convolutional layer slides small filters (kernels) across the image
# Each filter is a small grid (3x3 in my case) that looks for a specific pattern
# The filter slides over every possible position and computes a dot product
# The result is a feature map which shows where in the image that specific pattern was found
#
# Important things I learned:
#   - The filters are NOT hand-coded. They start random and the training learns what to detect
#   - First conv layer learns simple things like edges and corners
#   - Second conv layer combines those to detect more complex shapes like curves or parts of clothing
#   - Output size formula (no padding, stride=1): H_out = H - filter_size + 1
#     So 28x28 image with 3x3 filter gives 26x26 output
#
# I watched the deeplizard convolution visualization to understand this visually:
# https://deeplizard.com/resource/pavq7noze2
# Reference: https://cs231n.github.io/convolutional-networks/#conv
 
def conv_forward(X, W_conv, b_conv):
    N, H, W, C_in = X.shape
    f, _, _, C_out = W_conv.shape

    H_out = H - f + 1
    W_out = W - f + 1

    out = np.zeros((N, H_out, W_out, C_out))

    for h in range(H_out):
        for w in range(W_out):
            # Extract patch and flatten for fast matrix multiply
            patch      = X[:, h:h+f, w:w+f, :]   # (N, f, f, C_in)
            patch_flat = patch.reshape(N, -1)       # (N, f*f*C_in)
            W_flat     = W_conv.reshape(-1, C_out)  # (f*f*C_in, C_out)
            out[:, h, w, :] = patch_flat @ W_flat + b_conv

    cache = (X, W_conv, b_conv)
    return out, cache


# ===================================================
# STEP 5 — Convolutional Layer (Backward Pass)
# ===================================================
# In the backward pass I need three gradients:
#   dX     — gradient with respect to the input (sent to previous layer)
#   dW     — gradient with respect to filter weights (used to update filters)
#   db     — gradient with respect to biases (used to update biases)
#
# The logic reverses the forward loop:
#   For each position (h, w) where the filter was applied:
#     - The gradient of the output at that position tells us how wrong we were
#     - We spread that gradient back into the input region that contributed
#     - We also accumulate it into the filter weight gradients
#
# Reference: https://cs231n.github.io/convolutional-networks/
 
def conv_backward(d_out, cache):
    X, W_conv, b_conv  = cache
    N, H, W, C_in      = X.shape
    f, _, _, C_out      = W_conv.shape
    _, H_out, W_out, _  = d_out.shape

    dX = np.zeros_like(X)
    dW = np.zeros_like(W_conv)
    db = np.sum(d_out, axis=(0, 1, 2))  # sum over N, H, W dims

    for h in range(H_out):
        for w in range(W_out):
            patch      = X[:, h:h+f, w:w+f, :]
            patch_flat = patch.reshape(N, -1)
            d          = d_out[:, h, w, :]           # (N, C_out)

            dW += (patch_flat.T @ d).reshape(W_conv.shape)
            dX[:, h:h+f, w:w+f, :] += (
                d @ W_conv.reshape(-1, C_out).T
            ).reshape(N, f, f, C_in)

    return dX, dW, db


# ===================================================
# STEP 6 — Max Pooling (Forward Pass)
# ===================================================
# Max pooling reduces the spatial size of the feature maps
# It looks at small windows (2x2 in my case) and keeps only the maximum value
#
# Why I use max pooling:
#   - It reduces computation for the layers that come after
#   - It makes the model slightly tolerant to small shifts in the image
#   - After conv1 (output: 26x26), pooling gives 13x13
#
# Reference: https://cs231n.github.io/convolutional-networks/#pool
 
def maxpool_forward(X, pool_size=2):
    N, H, W, C = X.shape
    p = pool_size

    H_out = H // p
    W_out = W // p

    out = np.zeros((N, H_out, W_out, C))

    for h in range(H_out):
        for w in range(W_out):
            window = X[:, h*p:(h+1)*p, w*p:(w+1)*p, :]
            out[:, h, w, :] = np.max(window, axis=(1, 2))

    cache = (X, pool_size)
    return out, cache


# ===================================================
# STEP 7 — Max Pooling (Backward Pass)
# ===================================================
# During forward pass, only the maximum value in each window was kept
# So during backward pass, only that position should receive the gradient
# All other positions in the window get gradient = 0
# (because they did not contribute to the output at all)
#
# Reference: https://cs231n.github.io/convolutional-networks/#pool
 
def maxpool_backward(d_out, cache):
    X, p = cache
    N, H, W, C = X.shape
    H_out = H // p
    W_out = W // p

    dX = np.zeros_like(X)

    for h in range(H_out):
        for w in range(W_out):
            window      = X[:, h*p:(h+1)*p, w*p:(w+1)*p, :]
            window_flat = window.reshape(N, p*p, C)

            # Find which position in each window had the max
            # Ref: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
            max_idx = np.argmax(window_flat, axis=1)  # (N, C)
            max_h   = max_idx // p
            max_w   = max_idx %  p

            for n in range(N):
                for c in range(C):
                    mh = max_h[n, c]
                    mw = max_w[n, c]
                    dX[n, h*p + mh, w*p + mw, c] += d_out[n, h, w, c]

    return dX


# ===================================================
# STEP 8 — Fully Connected Layer
# ===================================================
# After the conv and pool layers, I flatten everything into a 1D vector
# Then this fully connected (FC) layer connects every input to every output neuron
# The formula is: output = input @ weights + bias
# This is the same as a standard dense layer in any neural network
#
# Reference: https://cs231n.github.io/linear-classify/#score-function
 
def fc_forward(X, W, b):
    out   = X @ W + b
    cache = (X, W, b)
    return out, cache


# Gradients: dX flows to the previous layer, dW and db update this layer.
# Ref: https://cs231n.github.io/optimization-2/
def fc_backward(d_out, cache):
    X, W, b = cache
    N = X.shape[0]

    dX = d_out @ W.T
    dW = X.T @ d_out
    db = np.mean(d_out, axis=0)

    return dX, dW, db


# ===================================================
# STEP 9 — CNN Class
# ===================================================

class CNN:

    def __init__(self, num_filters=16, filter_size=3,
                 hidden_size=256, num_classes=10, dropout_p=0.4):

        f  = filter_size
        nf = num_filters
        self.dropout_p = dropout_p

        # --- Conv1 weights ---
        # He init: sets scale so activations have unit variance after ReLU.
        # Without this, all filters learn the same thing (symmetry problem).
        # Ref: https://arxiv.org/abs/1502.01852
        fan_in_conv = f * f * 1
        self.W_conv1 = np.random.randn(f, f, 1, nf) * np.sqrt(2.0 / fan_in_conv)
        self.b_conv1 = np.zeros(nf)

        # --- Conv2 weights ---
        # Input channels = nf (output of conv1)
        fan_in_conv2 = f * f * nf
        self.W_conv2 = np.random.randn(f, f, nf, nf) * np.sqrt(2.0 / fan_in_conv2)
        self.b_conv2 = np.zeros(nf)

        # --- Compute flat size after both conv layers + one pool ---
        # Conv1: 28 -> 26, Pool: 26 -> 13, Conv2: 13 -> 11
        after_pool = (28 - f + 1) // 2          # = 13
        after_conv2 = after_pool - f + 1         # = 11
        flat_size = after_conv2 * after_conv2 * nf  # = 11*11*16 = 1936

        # --- FC1 weights ---
        self.W_fc1 = np.random.randn(flat_size, hidden_size) * np.sqrt(2.0 / flat_size)
        self.b_fc1 = np.zeros(hidden_size)

        # --- FC2 weights (output layer) ---
        self.W_fc2 = np.random.randn(hidden_size, num_classes) * np.sqrt(2.0 / hidden_size)
        self.b_fc2 = np.zeros(num_classes)

        # --- Momentum buffers (one per parameter, same shape, start at 0) ---
        # Momentum accumulates past gradients so updates build up in the
        # right direction and don't just bounce around.
        # Ref: https://cs231n.github.io/neural-networks-3/#sgd
        self.v = {k: np.zeros_like(v) for k, v in self._params().items()}

    def _params(self):
        # Convenience: return all parameters as a dict
        return {
            'W_conv1': self.W_conv1, 'b_conv1': self.b_conv1,
            'W_conv2': self.W_conv2, 'b_conv2': self.b_conv2,
            'W_fc1'  : self.W_fc1,   'b_fc1'  : self.b_fc1,
            'W_fc2'  : self.W_fc2,   'b_fc2'  : self.b_fc2,
        }

    # ------------------------------------------
    # FORWARD PASS
    # ------------------------------------------

    def forward(self, X, train=True):
        N = X.shape[0]

        # Reshape flat pixels
        # Ref: https://cs231n.github.io/convolutional-networks/#overview
        X_img = X.reshape(N, 28, 28, 1)

        # Conv1 -> ReLU -> MaxPool
        c1_out, c1_cache = conv_forward(X_img, self.W_conv1, self.b_conv1)
        r1_out           = relu(c1_out)
        p1_out, p1_cache = maxpool_forward(r1_out, pool_size=2)

        # Conv2 -> ReLU 
        c2_out, c2_cache = conv_forward(p1_out, self.W_conv2, self.b_conv2)
        r2_out           = relu(c2_out)

        # Flatten (N,11,11,16) -> (N,1936)
        # Ref: https://cs231n.github.io/convolutional-networks/#convert
        flat_out = r2_out.reshape(N, -1)

        # FC1 -> ReLU -> Dropout
        f1_out, f1_cache = fc_forward(flat_out, self.W_fc1, self.b_fc1)
        r3_out           = relu(f1_out)
        dp_out, dp_mask  = dropout_forward(r3_out, p=self.dropout_p, train=train)

        # FC2 -> Softmax
        f2_out, f2_cache = fc_forward(dp_out, self.W_fc2, self.b_fc2)
        probs            = softmax(f2_out)

        # Save everything needed for backward pass
        self.cache = {
            'X_img'  : X_img,
            'c1_out' : c1_out,   'c1_cache': c1_cache,
            'r1_out' : r1_out,
            'p1_out' : p1_out,   'p1_cache': p1_cache,
            'c2_out' : c2_out,   'c2_cache': c2_cache,
            'r2_out' : r2_out,
            'flat_out': flat_out,
            'f1_out' : f1_out,   'f1_cache': f1_cache,
            'r3_out' : r3_out,
            'dp_mask': dp_mask,
            'f2_cache': f2_cache,
        }

        return probs

    # ------------------------------------------
    # BACKWARD PASS
    # ------------------------------------------

    def backward(self, probs, y):
        N = probs.shape[0]
        c = self.cache

        # Gradient of softmax + cross-entropy combined:
        #   subtract 1 from the correct class score, divide by N.
        # Ref: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        d_scores = probs.copy()
        d_scores[np.arange(N), y.astype(int)] -= 1
        d_scores /= N

        # FC2 backward
        d_dp, dW_fc2, db_fc2 = fc_backward(d_scores, c['f2_cache'])

        # Dropout backward
        d_r3 = dropout_backward(d_dp, c['dp_mask'])

        # ReLU3 backward
        d_f1 = relu_backward(d_r3, c['f1_out'])

        # FC1 backward
        d_flat, dW_fc1, db_fc1 = fc_backward(d_f1, c['f1_cache'])

        # Unflatten back to conv2 output shape
        d_r2 = d_flat.reshape(c['r2_out'].shape)

        # ReLU2 backward
        d_c2 = relu_backward(d_r2, c['c2_out'])

        # Conv2 backward
        d_p1, dW_conv2, db_conv2 = conv_backward(d_c2, c['c2_cache'])

        # MaxPool backward
        d_r1 = maxpool_backward(d_p1, c['p1_cache'])

        # ReLU1 backward
        d_c1 = relu_backward(d_r1, c['c1_out'])

        # Conv1 backward
        _, dW_conv1, db_conv1 = conv_backward(d_c1, c['c1_cache'])

        return {
            'W_conv1': dW_conv1, 'b_conv1': db_conv1,
            'W_conv2': dW_conv2, 'b_conv2': db_conv2,
            'W_fc1'  : dW_fc1,   'b_fc1'  : db_fc1,
            'W_fc2'  : dW_fc2,   'b_fc2'  : db_fc2,
        }

    # ------------------------------------------
    # TRAIN
    # ------------------------------------------

    def train(self, X, y, lr=0.01, epochs=40, batch_size=64,
              momentum=0.9, lr_decay=0.95):
        # momentum: fraction of previous velocity to carry forward.
        #   0.9 is the standard starting value.
        #   Ref: https://cs231n.github.io/neural-networks-3/#sgd
        #
        # lr_decay: multiply LR by this after each epoch.
        #   0.95 means a 5% reduction per epoch — starts fast, ends precise.
        #   Ref: https://cs231n.github.io/neural-networks-3/#anneal
        #
        # epochs=40: loss was still falling at epoch 20 in v1, so we double it.
        #   Ref: https://cs231n.github.io/neural-networks-3/#babysit

        n = X.shape[0]
        y = y.astype(int)

        current_lr = lr

        for epoch in range(epochs):

            # Shuffle every epoch so batches see different data each time.
            # Ref: https://cs231n.github.io/optimization-1/#gd
            idx = np.random.permutation(n)
            X_s, y_s = X[idx], y[idx]

            epoch_loss  = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_b, y_b = X_s[start:end], y_s[start:end]

                # Forward (train=True enables dropout)
                probs = self.forward(X_b, train=True)

                # Cross-entropy loss: -log(probability of correct class)
                # Clip avoids log(0) = -inf
                # Ref: https://cs231n.github.io/linear-classify/#softmax
                loss = np.mean(-np.log(
                    np.clip(probs[np.arange(len(y_b)), y_b], 1e-9, 1.0)
                ))
                epoch_loss  += loss
                num_batches += 1

                grads = self.backward(probs, y_b)

                # Momentum SGD update:
                #   velocity = momentum * velocity - lr * gradient
                #   parameter += velocity
                # Builds up speed in consistent gradient directions.
                # Ref: https://cs231n.github.io/neural-networks-3/#sgd
                for key in grads:
                    self.v[key] = momentum * self.v[key] - current_lr * grads[key]
                    getattr(self, key)[:] += self.v[key]

            # Decay the learning rate after each epoch
            current_lr *= lr_decay

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1:>2}/{epochs}  |  Loss: {avg_loss:.4f}  |  LR: {current_lr:.5f}")

        print("Training Complete!")

    # ------------------------------------------
    # PREDICT
    # ------------------------------------------

    def predict(self, X):
        # train=False because dropout is off at test time
        probs = self.forward(X, train=False)

        # Return column index with highest probability = predicted class
        # Ref: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        return np.argmax(probs, axis=1)


# ===================================================
# ACCURACY
# ===================================================

def get_accuracy(y_true, y_pred):
    # np.mean on a boolean array = fraction of True = fraction correct
    # Ref: https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
    return np.mean(y_true == y_pred)


# ===================================================
# F1 SCORE
# ===================================================

def get_f1(y_true, y_pred):

    all_scores = []

    for c in range(10):
        # TP/FP/FN — the three cells of the confusion matrix we need
        # Ref: https://en.wikipedia.org/wiki/Confusion_matrix
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        # Precision: how many predicted-c are actually c?
        # Recall:    how many actual-c did we catch?
        # Ref: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1 = harmonic mean of precision & recall
        # Punishes imbalance: precision=1, recall=0 -> F1=0
        # Ref: https://en.wikipedia.org/wiki/F-score
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)
        all_scores.append(f1)

    # Macro F1: average per-class F1 (all 10 classes equally weighted)
    # Ref: https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification
    return np.mean(all_scores), all_scores


# ===================================================
# CLASSIFICATION REPORT
# ===================================================

def print_report(y_true, y_pred):

    macro_f1, f1_scores = get_f1(y_true, y_pred)

    precisions, recalls, supports = [], [], []

    for c in range(10):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recalls.append(   tp / (tp + fn) if (tp + fn) > 0 else 0)
        supports.append(  int(np.sum(y_true == c)))

    total    = len(y_true)
    accuracy = get_accuracy(y_true, y_pred)
    macro_p  = np.mean(precisions)
    macro_r  = np.mean(recalls)
    w_p  = np.sum([precisions[c] * supports[c] for c in range(10)]) / total
    w_r  = np.sum([recalls[c]    * supports[c] for c in range(10)]) / total
    w_f1 = np.sum([f1_scores[c]  * supports[c] for c in range(10)]) / total

    # Format spec: :<15 = left-align in 15 chars, :.2f = 2 decimal places
    # Ref: https://docs.python.org/3/library/string.html#format-specification-mini-language
    print("\n=== CNN RESULTS ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
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


# ===================================================
# CROSS VALIDATION 5-fold
# ===================================================

def cross_validation(X, y, k=5):

    # Each fold holds out 1/k of the data for validation.
    # Ref: https://scikit-learn.org/stable/modules/cross_validation.html#k-fold
    fold_size  = len(X) // k
    acc_scores = []

    for i in range(k):

        print(f"\nFold {i+1}")

        start = i * fold_size
        end   = (i + 1) * fold_size

        X_val,  y_val  = X[start:end], y[start:end]

        # Stack the two non-validation pieces into one training set
        # Ref: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])

        model = CNN()

        # 15 epochs per fold: enough to converge, not so many that k=5 takes forever.
        # Ref: https://cs231n.github.io/neural-networks-3/#babysit
        model.train(X_train, y_train, lr=0.01, epochs=15, batch_size=64)

        pred = model.predict(X_val)
        acc  = get_accuracy(y_val, pred)

        print(f"Accuracy = {acc:.4f}")
        acc_scores.append(acc)

    mean_acc = np.mean(acc_scores)
    # High std = model is sensitive to which fold it gets (high variance)
    # Ref: https://scikit-learn.org/stable/modules/cross_validation.html
    std_acc  = np.std(acc_scores)

    return mean_acc, std_acc


# ===================================================
# TRAIN MODEL
# ===================================================

print("Training CNN...")

model = CNN(
    num_filters=16,      # 16 filters which means more feature detectors
    filter_size=3,
    hidden_size=256,     # 256 hidden units which adds capacity
    num_classes=10,
    dropout_p=0.4        # drop 40% of FC1 neurons during training
)

# lr=0.01 with decay: starts fast, slows down as it converges.
# Ref: https://cs231n.github.io/neural-networks-3/#hyper
#
# epochs=40: loss was still decreasing at epoch 20 in v1.
# Ref: https://cs231n.github.io/neural-networks-3/#babysit
#
# batch_size=64: standard trade-off between speed and gradient stability.
# Ref: https://cs231n.github.io/optimization-1/#gd
model.train(X_dev_flat, y_dev, lr=0.01, epochs=40, batch_size=64,
            momentum=0.9, lr_decay=0.95)

print("Model Trained Successfully!")


# ===================================================
# TEST MODEL
# ===================================================

y_true = y_test.astype(int)
y_pred = model.predict(X_test_flat)

accuracy         = get_accuracy(y_true, y_pred)
macro_f1, all_f1 = get_f1(y_true, y_pred)

# ===================================================
# CHECK TRAINED MODEL ACCURACY
# ===================================================

# Step 1: Get predictions from trained model
y_pred = model.predict(X_test_flat)
y_true = y_test.astype(int)

# Step 2: Print full report
print_report(y_true, y_pred)

# ===================================================
# CROSS VALIDATION
# ===================================================

print("\nRunning Cross Validation...")

# k=5: 20% held out each fold. Standard and fast enough for this dataset size.
# Ref: https://scikit-learn.org/stable/modules/cross_validation.html
cv_mean, cv_std = cross_validation(X_dev_flat, y_dev.astype(int))


# ===================================================
# FINAL RESULTS
# ===================================================

all_f1 = print_report(y_true, y_pred)

print(f"CV Mean:   {cv_mean:.4f}")
print(f"CV Std:    {cv_std:.4f}")