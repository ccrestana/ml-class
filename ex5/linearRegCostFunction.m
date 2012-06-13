function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Cost (vectorized)
h_theta = X * theta;
j = (h_theta - y) .^ 2;
J = sum(j) / (2*m);
% Adding lambda part (regularization)
J = J + ((lambda/(2*m)) * sum(theta(2:size(theta)).^2));


% Now calculating the gradient (vectorized)
grad = (lambda/m).* theta;
grad(1) = 0;
grad = grad + (1/m) * X' * ((X * theta) - y);










% =========================================================================

grad = grad(:);

end
