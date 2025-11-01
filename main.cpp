#include <iostream>
#include <cmath>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <algorithm>
#include <vector>
#include <random>
#include <numeric>
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

std::pair<Eigen::MatrixXd, double> initialize(int d, int seed = 1){
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



int main() {
    // --- Setup hyperparameters and data dimensions ---
    const double alpha = 0.04;
    const int n_epochs = 4000;

    // Dimensions from the Pima dataset example in the notebook
    const int d_features = 8;
    const int m_train = 5000;
    const int m_test = 26800;

    // --- Create dummy data to simulate the notebook's data ---
    // In a real application, you would load this from a file.
    Eigen::MatrixXd X_train = Eigen::MatrixXd::Random(d_features, m_train);
    Eigen::MatrixXd Y_train = (Eigen::MatrixXd::Random(1, m_train).array() > 0.5).cast<double>();

    Eigen::MatrixXd X_test = Eigen::MatrixXd::Random(d_features, m_test);
    Eigen::MatrixXd Y_test = (Eigen::MatrixXd::Random(1, m_test).array() > 0.5).cast<double>();

    // --- Step 1: Initialize parameters ---
    // train_X.shape[0] in Python corresponds to train_X.rows() in Eigen
    auto init_params = initialize(X_train.rows());
    Eigen::MatrixXd w_init = init_params.first;
    double b_init = init_params.second;

    // --- Step 2: Fit the model ---
    // We pass 'true' for the logging parameter to see the loss decrease during training.
    TrainingResult training_output = model_fit(w_init, b_init, X_train, Y_train, alpha, n_epochs, true);
    ModelParameters trained_params = training_output.params;
    std::vector<double> losses = training_output.losses; // This vector holds the loss history for plotting

    // --- Step 3: Predict and Evaluate ---
    PredictionResult train_prediction = model_predict(trained_params, X_train, Y_train);
    PredictionResult test_prediction = model_predict(trained_params, X_test, Y_test);

    // --- Step 4: Print the final accuracies ---
    std::cout << "\n--- Final Evaluation ---" << std::endl;
    std::cout << "Train Accuracy of the model: " << train_prediction.accuracy * 100 << "%" << std::endl;
    std::cout << "Test Accuracy of the model: " << test_prediction.accuracy * 100 << "%" << std::endl;

    return 0;
}