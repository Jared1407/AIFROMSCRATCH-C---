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

//Parameters:
using ParameterMap = std::map<std::string, Eigen::MatrixXd>;


// --------------- Functions ----------------- //

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
ParameterMap initialize_network(const std::vector<int>& net_dims) {
    std::mt19937 gen(42);
    

    int numLayers = net_dims.size();
    ParameterMap params;
    //std::cout << numLayers << std::endl;

    for(int i  = 0; i < numLayers - 1; i++){
        std::normal_distribution<double> dist(net_dims[i+1],net_dims[i]);
        params["W" + std::to_string(i + 1)] = Eigen::MatrixXd::Random(net_dims[i+1], net_dims[i]) * 0.01;
        std::cout << "Params W" << i + 1 << ": " << "\n" << params["W" + std::to_string(i + 1)].rows() << "," << params["W" + std::to_string(i + 1)].cols() << std::endl;
        params["b" + std::to_string(i + 1)] = Eigen::MatrixXd::Zero(net_dims[i + 1],1);
        std::cout << "Params b" << i + 1 << ": " << "\n" << params["b" + std::to_string(i + 1)].rows() << "," << params["b" + std::to_string(i + 1)].cols() << std::endl;
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
std::pair<Eigen::MatrixXd, std::map<std::string, Eigen::MatrixXd>> linear_forward(const Eigen::MatrixXd &A_prev, const Eigen::MatrixXd &W, const Eigen::MatrixXd &b){
    std::map<std::string, Eigen::MatrixXd> cache;
    cache["A"] = A_prev;

    Eigen::MatrixXd Z = (W * A_prev).colwise() + b.col(0);
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

std::pair<Eigen::MatrixXd, std::map<std::string, Eigen::MatrixXd>> layer_forward(const Eigen::MatrixXd &A_prev, const Eigen::MatrixXd &W, const Eigen::MatrixXd &b, const std::string &activation){
    auto linearForward = linear_forward(A_prev, W, b);
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
std::pair<Eigen::MatrixXd, std::vector<std::map<std::string, Eigen::MatrixXd>>> multi_layer_forward(const Eigen::MatrixXd &A0, const ParameterMap &params){
    int L = params.size() / 2; // Number of layers
    Eigen::MatrixXd A = A0;
    std::vector<std::map<std::string, Eigen::MatrixXd>> caches;

    for(int l = 1; l < L; l++){
        auto& Wl = params.at("W" + std::to_string(l));
        auto& bl = params.at("b" + std::to_string(l));
        auto layerForward = layer_forward(A, Wl, bl, "relu");
        A = layerForward.first;
        caches.push_back(layerForward.second);
    }
    auto& WL = params.at("W" + std::to_string(L));
    auto& bL = params.at("b" + std::to_string(L));
    auto layerForward = layer_forward(A, WL, bL, "linear");
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
BackwardResult linear_backward(const Eigen::MatrixXd& dZ, const Eigen::MatrixXd& cache,const Eigen::MatrixXd &W, const Eigen::MatrixXd &b){
    Eigen::MatrixXd A = cache; // linear_backward still expects the linear cache as a matrix (A)

    // Compute dA_prev
    Eigen::MatrixXd dA_prev =W.transpose() * dZ;

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

BackwardResult layer_backward(const Eigen::MatrixXd& dA, const std::map<std::string, Eigen::MatrixXd>& cache, const Eigen::MatrixXd &W, const Eigen::MatrixXd &b, const std::string& activation){
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

    BackwardResult backResult = linear_backward(dZ, lin_cache, W, b);

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

std::map<std::string, Eigen::MatrixXd> multi_layer_backward(const Eigen::MatrixXd& dAL, const std::vector<std::map<std::string, Eigen::MatrixXd>>& caches, const ParameterMap &params){
    int L = caches.size();
    std::map<std::string, Eigen::MatrixXd> gradients;
    Eigen::MatrixXd dA = dAL;
    std::string activation = "linear";

    for(int l = L; l >= 1; l--){
        const auto& Wl = params.at("W" + std::to_string(l));
        const auto& bl = params.at("b" + std::to_string(l));
        BackwardResult backResult = layer_backward(dA, caches[l - 1], Wl, bl, activation);
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

std::vector<double> classify(const Eigen::MatrixXd& X, const ParameterMap &params){
    auto forwardResult = multi_layer_forward(X, params);
    Eigen::MatrixXd A = forwardResult.first;

    SoftCrossEntropyLoss softmaxResult = softmax_cross_entropy_loss(A);
    Eigen::MatrixXd AL = softmaxResult.A;

    std::vector<double> YPred(AL.cols());
    for(int i = 0; i < AL.cols(); i++){
        Eigen::Index index;
        AL.col(i).maxCoeff(&index);
        YPred[i] = static_cast<double>(index);
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
    ParameterMap params;
};

NeuralNetworkResult multi_layer_network(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const std::vector<int>& net_dims, int num_iterations, double learning_rate, bool log = false){
    ParameterMap params = initialize_network(net_dims);
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
        params = update_parameters(params, gradients, ii, learning_rate);
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

Eigen::MatrixXd one_hot_encode(const Eigen::MatrixXd& Y, int num_classes) {
    long m = Y.cols(); // Number of samples
    Eigen::MatrixXd Y_onehot = Eigen::MatrixXd::Zero(num_classes, m);
    for (long i = 0; i < m; ++i) {
        int class_index = static_cast<int>(Y(0, i));
        if (class_index >= 0 && class_index < num_classes) {
            Y_onehot(class_index, i) = 1.0;
        }
    }
    return Y_onehot;
}




int main(int argc, char *argv[]) {
    // Define network dimensions
    std::vector<int> net_dims = {784, 10};

    //init learning rate and iterations 
    double learning_rate = 0.15;
    int num_iterations = 250;

    // Load mnist data into trainX, trainY, testX, testY:

    // small helper to read big-endian ints from IDX files
    auto read_int32_be = [](std::ifstream &ifs)->int {
        unsigned char bytes[4];
        ifs.read(reinterpret_cast<char*>(bytes), 4);
        return (int)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
    };

    // load images: returns matrix of shape (rows*cols, num_images) normalized to [0,1]
    auto load_mnist_images = [&](const std::string &path)->Eigen::MatrixXd {
        std::ifstream ifs(path, std::ios::binary);
        if(!ifs.is_open()){
            std::cerr << "Unable to open images file: " << path << std::endl;
            return Eigen::MatrixXd();
        }
        int magic = read_int32_be(ifs);
        int num_images = read_int32_be(ifs);
        int rows = read_int32_be(ifs);
        int cols = read_int32_be(ifs);
        int img_size = rows * cols;
        Eigen::MatrixXd images(img_size, num_images);
        for(int i = 0; i < num_images; ++i){
            std::vector<unsigned char> buffer(img_size);
            ifs.read(reinterpret_cast<char*>(buffer.data()), img_size);
            for(int j = 0; j < img_size; ++j){
                images(j, i) = static_cast<double>(buffer[j]) / 255.0;
            }
        }
        return images;
    };

    // load labels: returns (1, num_items) matrix of label values
    auto load_mnist_labels = [&](const std::string &path)->Eigen::MatrixXd {
        std::ifstream ifs(path, std::ios::binary);
        if(!ifs.is_open()){
            std::cerr << "Unable to open labels file: " << path << std::endl;
            return Eigen::MatrixXd();
        }
        int magic = read_int32_be(ifs);
        int num_items = read_int32_be(ifs);
        Eigen::MatrixXd labels(1, num_items);
        for(int i = 0; i < num_items; ++i){
            unsigned char val = 0;
            ifs.read(reinterpret_cast<char*>(&val), 1);
            labels(0, i) = static_cast<double>(val);
        }
        return labels;
    };

    // filter X,Y to only include two digits (a and b). labels mapped to 0 (a) and 1 (b)
    auto filter_digits = [&](const Eigen::MatrixXd &X, const Eigen::MatrixXd &labels, int a, int b, Eigen::MatrixXd &X_out, Eigen::MatrixXd &Y_out){
        std::vector<int> keep;
        for(int i = 0; i < labels.cols(); ++i){
            int lab = static_cast<int>(labels(0,i));
            if(lab == a || lab == b) keep.push_back(i);
        }
        X_out.resize(X.rows(), keep.size());
        Y_out.resize(1, keep.size());
        for(size_t i = 0; i < keep.size(); ++i){
            X_out.col(i) = X.col(keep[i]);
            int lab = static_cast<int>(labels(0,keep[i]));
            Y_out(0,i) = (lab == a) ? 0.0 : 1.0;
        }
    };

    // Determine file paths (argv precedence)
    std::string train_images = (argc > 1) ? argv[1] : "train-images.idx3-ubyte";
    std::string train_labels = (argc > 2) ? argv[2] : "train-labels.idx1-ubyte";
    std::string test_images = (argc > 3) ? argv[3] : "t10k-images.idx3-ubyte";
    std::string test_labels = (argc > 4) ? argv[4] : "t10k-labels.idx1-ubyte";

    std::cout << "Loading training images from: " << train_images << std::endl;
    Eigen::MatrixXd trainX = load_mnist_images(train_images);
    Eigen::MatrixXd trainY = load_mnist_labels(train_labels);
    std::cout << "Loading test images from: " << test_images << std::endl;
    Eigen::MatrixXd testX = load_mnist_images(test_images);
    Eigen::MatrixXd testY = load_mnist_labels(test_labels);

    if(trainX.size() == 0 || trainY.size() == 0){
        std::cerr << "Failed to load training data. Exiting." << std::endl;
        return 1;
    }




    // optionally subsample for speed if dataset large
    int max_train = 2500;
    if(trainX.cols() > max_train){
        Eigen::MatrixXd Xsub = trainX.leftCols(max_train);
        Eigen::MatrixXd Ysub = trainY.leftCols(max_train);
        trainX = Xsub; trainY = Ysub;
    }

    // Train multi-layer network
    std::cout << "Training multi-layer neural network..." << std::endl;
    
    // Add multi-layer network training code here
    Eigen::MatrixXd trainY_onehot = one_hot_encode(trainY, 10);
    NeuralNetworkResult nnResult = multi_layer_network(trainX, trainY_onehot, net_dims, num_iterations, learning_rate, true);
    ParameterMap trainedParams = nnResult.params;
    std::cout << "Training complete." << std::endl;
    // Evaluate on test set
    std::cout << "Evaluating on test set..." << std::endl;
    std::vector<double> testPreds = classify(testX, trainedParams);
    int correct = 0;
    for(int i = 0; i < testY.cols(); ++i){
        if(static_cast<int>(testY(0,i)) == static_cast<int>(testPreds[i])){
            correct++;
        }
    }
    double testAccuracy = static_cast<double>(correct) / testY.cols();
    std::cout << "Test set accuracy: " << testAccuracy * 100.0 << "%" << std::endl;

    
    return 0;
}


