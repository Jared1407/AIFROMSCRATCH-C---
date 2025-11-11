#include <iostream>
#include <cmath>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <algorithm>
#include <vector>
#include <random>
#include <numeric>
#include <cstdlib>
#include <map>
using Eigen::MatrixXd;

//---------------- Data Structures ----------------//

// Structure to hold dataset:
struct Dataset {
    Eigen::MatrixXd X;
    Eigen::MatrixXd Y;
};

// Struct to hold model parameters:
struct ModelParameters {
    Eigen::MatrixXd w;
    double b;
};

// Struct to hold training results:
struct TrainingResult {
    ModelParameters params;
    std::vector<double> losses;
};

// Struct to hold prediction results:
struct PredictionResult {
    Eigen::MatrixXd Y_Pred;
    double accuracy;
    double loss;
};

//Struct to hold softmax cross entropy loss:
struct SoftCrossEntropyLoss{
    Eigen::MatrixXd A;
    std::map<std::string, Eigen::MatrixXd> cache;
    double loss;
};


// --------------- Functions ----------------- //

// Linear Regression: Polynomial Transform:
MatrixXd poly_transform(const Eigen::MatrixXd& X, int d){
    /*
    Function to transform scalar values into (d+1)-dimension vectors. 
    Each scalar value x is transformed a vector [1,x,x^2,x^3, ... x^d]. 
    
    Inputs:
        X: vector of m scalar inputs od shape (m, 1) where each row is a scalar input x
        d: number of dimensions
        
    Outputs:
        Phi: Transformed matrix of shape (m, (d+1)) 
    */
    long m = X.rows();
    Eigen::MatrixXd Phi(m, d+1);
    Phi.col(0) = Eigen::VectorXd::Ones(m);  
    for(int i = 1; i <= d; i++){
        Phi.col(i) = X.array().pow(i);
    }
    return Phi;
}

double lin_reg_obj(const Eigen::MatrixXd& Y, const Eigen::MatrixXd& Phi, const Eigen::MatrixXd& Theta){
    /*
    Objective function to estimate loss for the linear regression model.
    Inputs:
        Y: ground truth labels of dimensions (m, 1)
        Phi: Design matrix of dimensions (m, (d+1))
        Theta: Parameters of linear regression of dimensions ((d+1),1)
        
    outputs:
        loss: scalar loss 

    (Y-Phi(X)Theta)T * (Y-Phi(X)Theta)
    */
    Eigen::MatrixXd loss1 = Y - Phi * Theta;
    double loss = loss1.squaredNorm();
    return loss;
}

Eigen::MatrixXd lin_reg_fit(const Eigen::MatrixXd& Phi_X, const Eigen::MatrixXd& Y){
    /*
    A function to estimate the linear regression model parameters using the closed form solution.
    Inputs:
        Phi_X: Design matrix of dimensions (m, (d+1))
        Y: ground truth labels of dimensions (m, 1)
         
    Outputs:
        theta: Parameters of linear regression of dimensions ((d+1),1)
    */
    Eigen::MatrixXd Theta = (Phi_X.transpose() * Phi_X).ldlt().solve(Phi_X.transpose() * Y);
    return Theta;
}

double get_rmse(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y){
    /*
    function to evaluate the goodness of the linear regression model.
    
    Inputs:
        Y_pred: estimated labels of dimensions (m, 1)
        Y: ground truth labels of dimensions (m, 1)
        
    Outputs:
        rmse: root means square error
    */
    long m = Y.rows();
    double se = (Y - Y_pred).squaredNorm();
    double mse = se / m;
    double rmse = std::sqrt(mse);
    return rmse;
}

Eigen::MatrixXd ridge_reg_fit(Eigen::MatrixXd& Phi_X, Eigen::MatrixXd& Y, double lamb_d){
    /*
    A function to estimate the ridge regression model parameters using the closed form solution.
    Inputs:
        Phi_X: Design matrix of dimensions (m, (d+1))
        Y: ground truth labels of dimensions (m, 1)
        lamb_d: regularization parameter
         
    Outputs:                                                                                                                                                                                                                                              
        theta: Parameters of linear regression of dimensions ((d+1),1)
    */
    // Step 1: Get the dimension dplus1 using Phi_X to create the identity matrix I_d 
    long dplus1 = Phi_X.cols();
    Eigen::MatrixXd I_d = Eigen::MatrixXd::Identity(dplus1, dplus1);
    // The equation is: (Phi^T*Phi + lambda^2*I)^-1 * Phi^T * Y
    // Step 2: Estimate the closed form solution similar to *linear_reg_fit* but now include the lamb_d**2*I_d term

    Eigen::MatrixXd Theta = (Phi_X.transpose() * Phi_X + lamb_d * lamb_d * I_d).ldlt().solve(Phi_X.transpose() * Y);
    return Theta;
}

/*
### Cross Validation to Estimate Optimal Lambda for Ridge Regression

In order to avoid overfitting when using a high degree polynomial, we have used **ridge regression**. 
We now need to estimate the optimal value of $\lambda$ using **cross-validation**.

We will obtain a generic value of $\lambda$ using the entire training dataset to validate. 
We will employ the method of **$k$-fold cross validation**, 
where we split the training data into $k$ non-overlapping random subsets. 
In every cycle, for a given value of $\lambda$, $(k-1)$ subsets are used for training the ridge regression model 
and the remaining subset is used for evaluating the goodness of the fit. We estimate the average goodness of 
the fit across all the subsets and select the $lambda$ that results in the best fit.
*/

std::vector<std::vector<int>> k_val_ind(std::vector<int>& index, int k_fold, int seed = 42){
    /*
        Function to split the data into k folds for cross validation. Returns the indices of the data points 
    belonging to every split.
    
    Inputs:
        index: all the indices of the training
        k_fold: number of folds to split the data into
    
    Outputs:
        k_set: list of arrays with indices
    */
    // Step 1: Shuffle the indices randomly
    std::mt19937 g(seed);
    std::shuffle(index.begin(), index.end(), g);

    // Step 2: Split the indices into k folds
    std::vector<std::vector<int>> k_set;
    int fold_size = index.size() / k_fold;
    for(int i = 0; i < k_fold; i++){
        int start_index = i * fold_size;
        int end_index = (i == k_fold - 1) ? index.size() : (i + 1) * fold_size;
        std::vector<int> fold_indices(index.begin() + start_index, index.begin() + end_index);
        k_set.push_back(fold_indices);
    }
    return k_set;

}

// Selects rows from a matrix based on a vector of indices.
Eigen::MatrixXd select_rows(const Eigen::MatrixXd& matrix, const std::vector<int>& indices) {
    Eigen::MatrixXd subset(indices.size(), matrix.cols());
    for (int i = 0; i < indices.size(); ++i) {
        subset.row(i) = matrix.row(indices[i]);
    }
    return subset;
}

std::vector<int> get_training_indices(int total_size, const std::vector<int>& val_indices) {
    std::vector<bool> is_validation(total_size, false);
    for (int idx : val_indices) {
        is_validation[idx] = true;
    }

    std::vector<int> train_indices;
    for (int i = 0; i < total_size; ++i) {
        if (!is_validation[i]) {
            train_indices.push_back(i);
        }
    }
    return train_indices;
}

std::vector<double> k_fold_cv(int k_fold, const Eigen::MatrixXd& train_X, const Eigen::MatrixXd& train_Y, double lamb_d, int d) {
    /*
        Function to implement k-fold cross validation.
    Inputs:
        k_fold: number of validation subsests
        train_X: training data of dimensions (m, 1) 
        train_Y: ground truth training labels
        lamb_d: ridge regularization lambda parameter
        d: polynomial degree
        
    Outputs:
        rmse_list: list of root mean square errors (RMSE) for k_folds 
    */
    int m = train_X.rows();
    std::vector<int> index(m);
    std::iota(index.begin(), index.end(), 0);

    // Get the k sets of validation indices
    std::vector<std::vector<int>> k_set = k_val_ind(index, k_fold);

    // Transform all the data once
    Eigen::MatrixXd Phi_X = poly_transform(train_X, d);

    std::vector<double> rmse_list;
    for (int i = 0; i < k_fold; ++i) {
        // Get the indices for the current validation fold
        const std::vector<int>& val_indices = k_set[i];
        // Get the complementary training indices (the ~ind part)
        std::vector<int> train_indices = get_training_indices(m, val_indices);

        // Create training and validation subsets using the helper functions
        Eigen::MatrixXd Phi_X_train = select_rows(Phi_X, train_indices);
        Eigen::MatrixXd Y_train = select_rows(train_Y, train_indices);

        Eigen::MatrixXd Phi_X_val = select_rows(Phi_X, val_indices);
        Eigen::MatrixXd Y_val = select_rows(train_Y, val_indices);
        
        // Step 1: Estimate theta using the training subset
        Eigen::MatrixXd theta = ridge_reg_fit(Phi_X_train, Y_train, lamb_d);

        // Step 2: Estimate Y_pred over the validation subset
        Eigen::MatrixXd Y_pred = Phi_X_val * theta;

        // Step 3: Determine rmse using Y_pred and the validation labels
        double rmse = get_rmse(Y_pred, Y_val);

        rmse_list.push_back(rmse);
    }
    return rmse_list;
}

std::pair<Eigen::MatrixXd, double> initialize1(int d, int seed = 1){
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd w(d, 1);
    for (int i = 0; i < d; ++i){
        w(i, 0) = 0.01 * dist(gen);
    }
    return {w, 0.0};
}

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& z){
    return 1.0 / (1.0 + (-z.array()).exp());
}

double logistic_loss(const Eigen::MatrixXd& A, const Eigen::MatrixXd& Y){
    /*
    Function to calculate the logistic loss given the predictions and the targets.
    
    Inputs:
        A: Estimated prediction values, A is of dimension (1, m)
        Y: groundtruth labels, Y is of dimension (1, m)
        
    Outputs:
        loss: logistic loss
    */
    long m = A.cols(); // Number of samples

    // Calculate the loss using element-wise array operations
    // Formula: Y * log(A) + (1 - Y) * log(1 - A)
    double sum = (Y.array() * A.array().log() + (1.0 - Y.array()) * (1.0 - A.array()).log()).sum();

    double loss = -(1.0 / m) * sum;

    return loss;
}

std::pair<Eigen::MatrixXd, double> grad_fn(const Eigen::MatrixXd& X, const Eigen::MatrixXd& dZ){
    /*
    Function to calculate the gradients of weights (dw) and biases (db) w.r.t the objective function L.
    
    Inputs:
        X: training data of dimensions (d, m)
        dZ: gradient dL/dZ where L is the logistic loss and Z = w^T*X+b is the input to the sigmoid activation function
            dZ is of dimensions (1, m)
        
    outputs:
        dw: gradient dL/dw - gradient of the weight w.r.t. the logistic loss. It is of dimensions (d,1)
        db: gradient dL/db - gradient of the bias w.r.t. the logistic loss. It is a scalar
    */
    long m = X.cols(); // number of samples
    Eigen::MatrixXd dw = (X * dZ.transpose()) / m; // (d, m) * (m, 1) = (d, 1)
    double db = dZ.sum() / m; // scalar
    return {dw, db};
}

TrainingResult model_fit(Eigen::MatrixXd w, double b, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, 
    double alpha, int n_epochs, bool log = false) {
    /*
    Function to fit a logistic model with the parameters w,b to the training data with labels X and Y.
    
    Inputs:
        w: weight vector of dimensions (d, 1)
        b: scalar bias value
        X: training data of dimensions (d, m)
        Y: training data labels of dimensions (1, m)
        alpha: learning rate
        n_epochs: number of epochs to train the model
        
    Outputs:
        params: a dictionary to hold parameters w and b
        losses: a list train loss at every epoch
        # Implement the steps in the logistic regression using the functions defined earlier.
        # For each iteration of the for loop
            # Step 1: Calculate output Z = w.T*X + b
            # Step 2: Apply sigmoid activation: A = sigmoid(Z)
            # Step 3: Calculate loss = logistic_loss(.) between predicted values A and groundtruth labels Y
            # Step 4: Estimate gradient dZ = A-Y
            # Step 5: Estimate gradients dw and db using grad_fn(.).
            # Step 6: Update parameters w and b using gradients dw, db and learning rate
            #         w = w - alpha * dw
            #         b = b - alpha * db
    */

    // Initialize parameters
   std::vector<double> losses;

    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        // Step 1: Calculate output Z = w^T * X + b (Forward Pass)
        Eigen::MatrixXd Z = (w.transpose() * X).array() + b; // Z is of dimensions (1, m)

        // Step 2: Apply sigmoid activation
        Eigen::MatrixXd A = sigmoid(Z);

        // Step 3: Calculate loss
        double loss = logistic_loss(A, Y);

        // Step 4: Estimate gradient dZ (Backward Pass)
        Eigen::MatrixXd dZ = A - Y;

        // Step 5: Estimate gradients dw and db
        auto gradients = grad_fn(X, dZ);
        Eigen::MatrixXd dw = gradients.first;
        double db = gradients.second;

        // Step 6: Update parameters
        w = w - alpha * dw;
        b = b - alpha * db;

        // Log the loss periodically
        if (epoch % 100 == 0) {
            losses.push_back(loss);
            if (log) {
                std::cout << "After " << epoch << " iterations, Loss = " << loss << std::endl;
            }
        }
    }

    TrainingResult result;
    result.params = {w, b};
    result.losses = losses;
    
    return result;
}

// Model Prediction:
PredictionResult model_predict(ModelParameters params, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int pred_threshold = 0.5) {
    /*
        Function to calculate category predictions on given data and returns the accuracy of the predictions.
    Inputs:
        params: a dictionary to hold parameters w and b
        X: training data of dimensions (d, m)
        Y: training data labels of dimensions (1, m). If not provided, the function merely makes predictions on X
        
    outputs:
        Y_Pred: Predicted class labels for X. Has dimensions (1, m)
        acc: accuracy of prediction over X if Y is provided else, 0 
        loss: loss of prediction over X if Y is provided else, Inf 
    
    */
    Eigen::MatrixXd w = params.w;
    double b = params.b;
    long m = X.cols(); // number of samples

    // Calculate Z using X, w and b
    Eigen::MatrixXd Z = (w.transpose() * X).array() + b;

    // Calculate A using the sigmoid - A is the set of (1,m) probabilities
    Eigen::MatrixXd A = sigmoid(Z);

    // Calculate the prediction labels Y_Pred of size (1,m) using A and pred_threshold
    Eigen::MatrixXd Y_Pred = (A.array() > pred_threshold).cast<double>();

    // Calculate accuracy and loss if Y is provided
    double accuracy = 0.0;
    double loss = std::numeric_limits<double>::infinity();
    if (Y.size() > 0) {
        accuracy = (Y_Pred.array() == Y.array()).cast<double>().mean();
        loss = logistic_loss(A, Y);
    }

    PredictionResult result;

    result.Y_Pred = Y_Pred;
    result.accuracy = accuracy;
    result.loss = loss;

    return result;
}

// Begin setup for Multi-Category Neural Net:
    /*
    Computes relu activation of input Z
    
    Inputs: 
        Z: numpy.ndarray (n, m) which represent 'm' samples each of 'n' dimension
        
    Outputs: 
        A: where A = ReLU(Z) is a numpy.ndarray (n, m) representing 'm' samples each of 'n' dimension
        cache: a dictionary with {"Z", Z}
    */
std::pair<Eigen::MatrixXd, std::map<std::string, Eigen::MatrixXd>> relu(const Eigen::MatrixXd& Z){
    Eigen::MatrixXd A = Z.cwiseMax(0.0);
    std::map<std::string, Eigen::MatrixXd> cache;
    cache["Z"] = Z;
    return {A, cache};
}

// ReLU Gradient:
    /*
    Computes derivative of relu activation
    
    Inputs: 
        dA: derivative from the subsequent layer of dimension (n, m). 
            dA is multiplied elementwise with the gradient of ReLU
        cache: dictionary with {"Z", Z}, where Z was the input 
            to the activation layer during forward propagation
        
    Outputs: 
        dZ: the derivative of dimension (n,m). It is the elementwise 
            product of the derivative of ReLU and dA
    */
Eigen::MatrixXd relu_der(const Eigen::MatrixXd& dA, const std::map<std::string, Eigen::MatrixXd>& cache){
    Eigen::MatrixXd Z = cache.at("Z");
    Eigen::MatrixXd dZ = dA;
    dZ = dZ.array() * (Z.array() > 0).cast<double>();
    return dZ;
}

// Linear Activation: Linear(Z) = Z
    /*
    Computes linear activation of Z
    This function is implemented for completeness
        
    Inputs: 
        Z: numpy.ndarray (n, m) which represent 'm' samples each of 'n' dimension
        
    Outputs: 
        A: where A = Linear(Z) is a numpy.ndarray (n, m) representing 'm' samples each of 'n' dimension
        cache: a dictionary with {"Z", Z}  
    */
std::pair<Eigen::MatrixXd, std::map<std::string, Eigen::MatrixXd>> linear(const Eigen::MatrixXd& Z){
    Eigen::MatrixXd A = Z;
    std::map<std::string, Eigen::MatrixXd> cache;
    cache["Z"] = Z;
    return {A, cache};
}

// Linear Activation Gradient:
    /*
    Computes derivative of linear activation
    This function is implemented for completeness
    
    Inputs: 
        dA: derivative from the subsequent layer of dimension (n, m). 
            dA is multiplied elementwise with the gradient of Linear(.)
        cache: dictionary with {"Z", Z}, where Z was the input 
            to the activation layer during forward propagation
        
    Outputs: 
        dZ: the derivative of dimension (n,m). It is the elementwise 
            product of the derivative of Linear(.) and dA
    */
Eigen::MatrixXd linear_der(const Eigen::MatrixXd& dA,const std::map<std::string, Eigen::MatrixXd>& /*cache*/){
    Eigen::MatrixXd dZ = dA;
    return dZ;
}


//Softmax Cross Entropy Loss:
    /*
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs: 
        Z: numpy.ndarray (n, m)
        Y: numpy.ndarray (1, m) of labels
            when y=[] loss is set to []
    
    Outputs:
        A: numpy.ndarray (n, m) of softmax activations
        cache: a dictionary to store the activations which will be used later to estimate derivatives
        loss: cost of prediction
    */
SoftCrossEntropyLoss softmax_cross_entropy_loss(const Eigen::MatrixXd& Z, const Eigen::MatrixXd& Y = Eigen::MatrixXd()){
    long n = Z.rows(); // number of classes
    long m = Z.cols(); // number of samples
    Eigen::MatrixXd exp_Z = Z.array().exp();
    Eigen::MatrixXd A = exp_Z.array().rowwise() / exp_Z.colwise().sum().array();
    SoftCrossEntropyLoss result;
    result.A = A;
    result.cache["A"] = A;
    result.loss = 0.0;

    if(Y.size() > 0){
        Eigen::MatrixXd logA = A.array().log();
        double sum = (Y.array() * logA.array()).sum();
        result.loss = -(1.0 / m) * sum;
    } else {
        result.loss = std::numeric_limits<double>::infinity();
    }

    return result;
}

// Softmax Cross Entropy Loss Derivative:
    /*
    Computes the derivative of the softmax activation and cross entropy loss

    Inputs: 
        Y: numpy.ndarray (1, m) of labels
        cache: a dictionary with cached activations A of size (n,m)

    Outputs:
        dZ: derivative dL/dZ - a numpy.ndarray of dimensions (n, m) 
    */
Eigen::MatrixXd softmax_cross_entropy_loss_der(const Eigen::MatrixXd& Y, const std::map<std::string, Eigen::MatrixXd>& cache){
    Eigen::MatrixXd A = cache.at("A");
    long m = Y.cols(); // number of samples
    Eigen::MatrixXd dZ = (A - Y) / m;
    return dZ;
}

struct parameters {
    Eigen::MatrixXd W;
    Eigen::MatrixXd b;
};

// Init multi-layer NN:
    /*
    Initializes the parameters of a multi-layer neural network
    
    Inputs:
        net_dims: List containing the dimensions of the network. The values of the array represent the number of nodes in 
        each layer. For Example, if a Neural network contains 784 nodes in the input layer, 800 in the first hidden layer, 
        500 in the secound hidden layer and 10 in the output layer, then net_dims = [784,800,500,10]. 
    
    Outputs:
        parameters: Python Dictionary for storing the Weights and bias of each layer of the network

    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers-1):
        parameters["W"+str(l+1)] = 0.01 * np.random.randn(net_dims[l + 1],net_dims[l])
        parameters["b"+str(l+1)] = np.zeros((net_dims[l + 1],1))

    return parameters
    */
parameters initialize_network(const std::vector<int>& net_dims) {
    std::mt19937 gen(42);
    

    int numLayers = net_dims.size();
    parameters params;
    //std::cout << numLayers << std::endl;

    for(int i  = 0; i < numLayers - 1; i++){
        std::normal_distribution<double> dist(net_dims[i+1],net_dims[i]);
        params.W = Eigen::MatrixXd::Random(net_dims[i+1], net_dims[i]);
        std::cout << "Params W" << i + 1 << ": " << "\n" << params.W.rows() << "," << params.W.cols() << std::endl;
        params.b = Eigen::MatrixXd::Zero(net_dims[i + 1],1);
        std::cout << "Params b" << i + 1 << ": " << "\n" << params.b.rows() << "," << params.b.cols() << std::endl;
    }

    return params;
}
// Linear Forward: 
    /*
    Input A_prev propagates through the layer 
    Z = WA + b is the output of this layer. 

    Inputs: 
        A_prev: numpy.ndarray (n,m) the input to the layer
        W: numpy.ndarray (n_out, n) the weights of the layer
        b: numpy.ndarray (n_out, 1) the bias of the layer

    Outputs:
        Z: where Z = W.A_prev + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache: a dictionary containing the inputs A


    cache = {"A": A_prev}

    Z = np.dot(W,A_prev) + b

    return Z, cache
    
    */
std::pair<Eigen::MatrixXd, std::map<std::string, Eigen::MatrixXd>> linear_forward(const Eigen::MatrixXd &A_prev, const parameters &params){
    std::map<std::string, Eigen::MatrixXd> cache;
    cache["A"] = A_prev;

    Eigen::MatrixXd Z = (params.W * A_prev).colwise() + params.b.col(0);

    return {Z, cache};
}

// Layer Forward:
 /*
    Input A_prev propagates through the layer and the activation

    Inputs: 
        A_prev: numpy.ndarray (n,m) the input to the layer
        W: numpy.ndarray (n_out, n) the weights of the layer
        b: numpy.ndarray (n_out, 1) the bias of the layer
        activation: is the string that specifies the activation function

    Outputs:
        A: = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache: a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
        
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)

    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache
        
    */

std::pair<Eigen::MatrixXd, std::map<std::string, Eigen::MatrixXd>> layer_forward(const Eigen::MatrixXd &A_prev, const parameters &params, const std::string &activation){
    auto linearForward = linear_forward(A_prev, params);
    Eigen::MatrixXd Z = linearForward.first;
    std::map<std::string, Eigen::MatrixXd> lin_cache = linearForward.second;

    Eigen::MatrixXd A;
    std::map<std::string, Eigen::MatrixXd> act_cache;

    if(activation == "relu"){
        auto reluForward = relu(Z);
        A = reluForward.first;
        act_cache = reluForward.second;
    } else if(activation == "linear"){
        auto linearForwardAct = linear(Z);
        A = linearForwardAct.first;
        act_cache = linearForwardAct.second;
    } else {
        throw std::invalid_argument("Unsupported activation function");
    }

    std::map<std::string, Eigen::MatrixXd> cache;
    cache["lin_cache"] = lin_cache.at("A");
    // act_cache contains key "Z"
    cache["act_cache"] = act_cache.at("Z");

    return {A, cache};
}

// Multi-Layer Forward:
    /*
    Forward propgation through the layers of the network

    Inputs: 
        A0: numpy.ndarray (n,m) with n features and m samples
        parameters: dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    
    Outputs:
        AL: numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
            where c is number of categories and m is number of samples
        caches: a dictionary of associated caches of parameters and network inputs

    L = len(parameters)//2  
    A = A0
    caches = []
    
    for l in range(1,L):
        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)
    
    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
    caches.append(cache)
    return AL, caches
    */
std::pair<Eigen::MatrixXd, std::vector<std::map<std::string, Eigen::MatrixXd>>> multi_layer_forward(const Eigen::MatrixXd &A0, const parameters &params){
    int L = params.W.rows(); // Number of layers
    Eigen::MatrixXd A = A0;
    std::vector<std::map<std::string, Eigen::MatrixXd>> caches;

    for(int l = 1; l < L; l++){
        auto layerForward = layer_forward(A, params, "relu");
        A = layerForward.first;
        std::map<std::string, Eigen::MatrixXd> cache = layerForward.second;
        caches.push_back(cache);
    }
    auto layerForward = layer_forward(A, params, "linear");
    Eigen::MatrixXd AL = layerForward.first;
    caches.push_back(layerForward.second);

    return {AL, caches};
}

// Backward Propagation Through a Single Layer:
/*
'''
    Backward prpagation through the linear layer

    Inputs:
        dZ: numpy.ndarray (n,m) derivative dL/dz 
        cache: a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W: numpy.ndarray (n,p)
        b: numpy.ndarray (n, 1)

    Outputs:
        dA_prev: numpy.ndarray (p,m) the derivative to the previous layer
        dW: numpy.ndarray (n,p) the gradient of W 
        db: numpy.ndarray (n, 1) the gradient of b
    '''
    A = cache["A"]
    m = dZ.shape[1]
    
    # Compute dA_prev
    dA_prev = np.dot(W.T, dZ)
    
    # Compute dW
    dW = np.dot(dZ, A.T)
    
    # Compute db (sum across the rows)
    db = np.sum(dZ, axis=1, keepdims=True)

    
    
    return dA_prev, dW, db
*/
struct BackwardResult {
    Eigen::MatrixXd dA_prev;
    Eigen::MatrixXd dW;
    Eigen::MatrixXd db;
};
BackwardResult linear_backward(const Eigen::MatrixXd& dZ, const Eigen::MatrixXd& cache, const parameters& params){
    Eigen::MatrixXd A = cache; // linear_backward still expects the linear cache as a matrix (A)

    // Compute dA_prev
    Eigen::MatrixXd dA_prev = params.W.transpose() * dZ;

    // Compute dW
    Eigen::MatrixXd dW = (dZ * A.transpose());

    // Compute db
    Eigen::MatrixXd db = dZ.rowwise().sum();

    BackwardResult result;
    result.dA_prev = dA_prev;
    result.dW = dW;
    result.db = db;

    return result;
}

// Back Propagation with Activation:
/*
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA: numpy.ndarray (n,m) the derivative to the previous layer
        cache: dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W: numpy.ndarray (n,p)
        b: numpy.ndarray (n, 1)
    
    Outputs:
        dA_prev: numpy.ndarray (p,m) the derivative to the previous layer
        dW: numpy.ndarray (n,p) the gradient of W 
        db: numpy.ndarray (n, 1) the gradient of b
    '''

    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db
*/

BackwardResult layer_backward(const Eigen::MatrixXd& dA, const std::map<std::string, Eigen::MatrixXd>& cache, const parameters& params, const std::string& activation){
    // Separate lin_cache and act_cache from cache

    Eigen::MatrixXd lin_cache = cache.at("lin_cache"); 
    Eigen::MatrixXd act_cache = cache.at("act_cache"); 

    Eigen::MatrixXd dZ;
    if(activation == "relu"){
        // relu_der expects a map with key "Z"
        std::map<std::string, Eigen::MatrixXd> act_cache_map;
        act_cache_map["Z"] = act_cache;
        dZ = relu_der(dA, act_cache_map);
    } else if(activation == "linear"){
        std::map<std::string, Eigen::MatrixXd> act_cache_map;
        act_cache_map["Z"] = act_cache;
        dZ = linear_der(dA, act_cache_map);
    } else {
        throw std::invalid_argument("Unsupported activation function");
    }

    BackwardResult backResult = linear_backward(dZ, lin_cache, params);

    return backResult;
}


// Multi-Layer Backward:
/*
'''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs: 
        dAL: numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches: a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Outputs:
        gradients: dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''

    L = len(caches) 
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                    layer_backward(dA, caches[l-1], \
                    parameters["W"+str(l)],parameters["b"+str(l)],\
                    activation)
        activation = "relu"
    return gradients
*/

std::map<std::string, Eigen::MatrixXd> multi_layer_backward(const Eigen::MatrixXd& dAL, const std::vector<std::map<std::string, Eigen::MatrixXd>>& caches, const parameters& params){
    int L = caches.size();
    std::map<std::string, Eigen::MatrixXd> gradients;
    Eigen::MatrixXd dA = dAL;
    std::string activation = "linear";

    for(int l = L; l >= 1; l--){
        BackwardResult backResult = layer_backward(dA, caches[l - 1], params, activation);
        dA = backResult.dA_prev;
        gradients["dW" + std::to_string(l)] = backResult.dW;
        gradients["db" + std::to_string(l)] = backResult.db;
        activation = "relu";
    }

    return gradients;
}

// Prediction:
/*
'''
    Network prediction for inputs X

    Inputs: 
        X: numpy.ndarray (n,m) with n features and m samples
        parameters: dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Outputs:
        YPred: numpy.ndarray (1,m) of predictions
    '''
    # Forward propagate input 'X' using multi_layer_forward(.) and obtain the final activation 'A'
    # Using 'softmax_cross_entropy loss(.)', obtain softmax activation 'AL' with input 'A' from step 1
    # Predict class label 'YPred' as the 'argmax' of softmax activation from step-2. 
    # Note: the shape of 'YPred' is (1,m), where m is the number of samples

    # your code here
    A, cache = multi_layer_forward(X, parameters)
    AL, cache, loss = softmax_cross_entropy_loss(A)

    YPred = np.argmax(AL, axis=0, keepdims = True)
  

    return YPred
*/

std::vector<double> classify(const Eigen::MatrixXd& X, const parameters& params){
    auto forwardResult = multi_layer_forward(X, params);
    Eigen::MatrixXd A = forwardResult.first;

    SoftCrossEntropyLoss softmaxResult = softmax_cross_entropy_loss(A);
    Eigen::MatrixXd AL = softmaxResult.A;

    std::vector<double> YPred(AL.cols());
    for(int i = 0; i < AL.cols(); i++){
        AL.col(i).maxCoeff(&YPred[i]);
    }

    return YPred;
}


// Parameter Update using Batch-Gradient:
/*
'''
    Updates the network parameters with gradient descent

    Inputs:
        parameters: dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients: dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch: epoch number
        alpha: step size or learning rate
        
    Outputs:
        parameters: updated dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    '''
    
    L = len(parameters)//2
    for i in range(L):
        
        parameters["W"+str(i+1)] -= alpha * gradients["dW" + str(i+1)]
        parameters["b"+str(i+1)] -= alpha * gradients["db" + str(i+1)]

    return parameters
*/

std::map<std::string, Eigen::MatrixXd> update_parameters(const std::map<std::string, Eigen::MatrixXd>& parameters, const std::map<std::string, Eigen::MatrixXd>& gradients, double epoch, double alpha){
    int L = parameters.size() / 2;
    std::map<std::string, Eigen::MatrixXd> updated_params = parameters;

    for(int i = 0; i < L; i++){
        updated_params["W" + std::to_string(i + 1)] -= alpha * gradients.at("dW" + std::to_string(i + 1));
        updated_params["b" + std::to_string(i + 1)] -= alpha * gradients.at("db" + std::to_string(i + 1));
    }

    return updated_params;
}

// Neural Network:
/*
    '''
    Creates the multilayer network and trains the network

    Inputs:
        X: numpy.ndarray (n,m) of training data
        Y: numpy.ndarray (1,m) of training data labels
        net_dims: tuple of layer dimensions
        num_iterations: num of epochs to train
        learning_rate: step size for gradient descent
        log: boolean to print training progression 
    
    Outputs:
        costs: list of costs (or loss) over training
        parameters: dictionary of trained network parameters
    '''

    parameters = initialize_network(net_dims)
    A0 = X
    costs = []
    num_classes = 10
    alpha = learning_rate
    prev_parameter = None
    for ii in range(num_iterations):
        
        ## Forward Propagation
        # Step 1: Input 'A0' and 'parameters' into the network using multi_layer_forward()
        #         and calculate output of last layer 'A' (before softmax) and obtain cached activations as 'caches'
        # Step 2: Input 'A' and groundtruth labels 'Y' to softmax_cros_entropy_loss(.) and estimate
        #         activations 'AL', 'softmax_cache' and 'loss'
        
        ## Back Propagation
        # Step 3: Estimate gradient 'dAL' with softmax_cros_entropy_loss_der(.) using groundtruth 
        #         labels 'Y' and 'softmax_cache' 
        # Step 4: Estimate 'gradients' with multi_layer_backward(.) using 'dAL' and 'parameters' 
        # Step 5: Estimate updated 'parameters' and updated learning rate 'alpha' with update_parameters(.) 
        #         using 'parameters', 'gradients', loop variable 'ii' (epoch number) and 'learning_rate'
        #         Note: Use the same variable 'parameters' as input and output to the update_parameters(.) function
        
        # your code here

        #Step 1:
        A, caches = multi_layer_forward(A0, parameters)
        #Step 2:
        AL, softmax_cache, loss = softmax_cross_entropy_loss(A, Y)
        #Step 3:
        dAL = softmax_cross_entropy_loss_der(Y, softmax_cache)
        #Step 4:
        gradients = multi_layer_backward(dAL, caches, parameters)
        #Step 5: 
        parameters = update_parameters(parameters, gradients, ii, learning_rate)
        

        if ii % 20 == 0:
            costs.append(loss)
            if log:
                print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(ii+1, cost, learning_rate))
    
    return costs, parameters
}*/
struct NeuralNetworkResult {
    std::vector<double> costs;
    parameters params;
};

NeuralNetworkResult multi_layer_network(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const std::vector<int>& net_dims, int num_iterations, double learning_rate, bool log = false){
    parameters params = initialize_network(net_dims);
    Eigen::MatrixXd A0 = X;
    std::vector<double> costs;

    for(int ii = 0; ii < num_iterations; ii++){
        // Step 1:
        auto forwardResult = multi_layer_forward(A0, params);
        Eigen::MatrixXd A = forwardResult.first;
        std::vector<std::map<std::string, Eigen::MatrixXd>> caches = forwardResult.second;

        // Step 2:
        SoftCrossEntropyLoss softmaxResult = softmax_cross_entropy_loss(A, Y);
        Eigen::MatrixXd AL = softmaxResult.A;
        std::map<std::string, Eigen::MatrixXd> softmax_cache = softmaxResult.cache;
        double loss = softmaxResult.loss;

        // Step 3:
        Eigen::MatrixXd dAL = softmax_cross_entropy_loss_der(Y, softmax_cache);

        // Step 4:
        std::map<std::string, Eigen::MatrixXd> gradients = multi_layer_backward(dAL, caches, params);

        // Step 5:
        std::map<std::string, Eigen::MatrixXd> params_map;
        params_map = update_parameters(params_map, gradients, ii, learning_rate);
        if(ii % 20 == 0){
            costs.push_back(loss);
            if(log){
                std::cout << "Cost at iteration " << ii + 1 << " is: " << loss << ", learning rate: " << learning_rate << std::endl;
            }
        }
    }

    NeuralNetworkResult result;
    result.costs = costs;
    result.params = params;

    return result;
}





int main(int argc, char *argv[]) {
    // __________TODO___________ //
    // Convert all caches to maps


    // --- Setup hyperparameters and data dimensions ---
    std::vector<int> test_init;
    for(int i = 1; i < argc; i ++){
        test_init.push_back(atoi(argv[i]));
    }

    std::vector<int> net_dims = {784, 1568, 392, 10};

    //init learning rate and iterations 
    double learning_rate = 0.15;
    int num_iterations = 250;

    // Load mnist data into trainX, trainY, testX, testY:



    return 0;
}