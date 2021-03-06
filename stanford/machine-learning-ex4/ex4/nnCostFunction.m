function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTION(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


y_hot = zeros(m, num_labels);
for i = 1:m
    y_hot(i, y(i)) = 1;
end

a_1s = [ones(m, 1) X];
z_2s = a_1s * Theta1';
a_2s = [ones(m, 1) sigmoid(z_2s)];
z_3s = a_2s * Theta2';
h = sigmoid(z_3s);
J_summand = (y_hot .* log(h)) + ((1-y_hot) .* log(1-h));
J = -(sum(J_summand(:)) / m);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


for t = 1:m
    delta_3 = (h(t,:) - y_hot(t,:))';
    delta_2 = (Theta2(:,2:end)' * delta_3) .* sigmoidGradient(z_2s(t,:))';
    
    Theta2_grad = Theta2_grad + delta_3 * a_2s(t,:);
    Theta1_grad = Theta1_grad + delta_2 * a_1s(t,:);
end

Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Regularization of J
Theta1_summand = Theta1(:, 2:end) .^ 2;
Theta2_summand = Theta2(:, 2:end) .^ 2;
J_reg_sum = sum(Theta1_summand(:)) + sum(Theta2_summand(:));
J = J + (lambda/(2*m)) * J_reg_sum;

% Regularization of gradient
Theta2_reg = [zeros(size(Theta2,1),1) (lambda / m) * Theta2(:,2:end)];
Theta1_reg = [zeros(size(Theta1,1),1) (lambda / m) * Theta1(:,2:end)];
Theta2_grad = Theta2_grad + Theta2_reg;
Theta1_grad = Theta1_grad + Theta1_reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
