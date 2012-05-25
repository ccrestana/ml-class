function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Expand input X: add a 1 to each example (the bias) --> a_1
a_1 = [ones(m, 1) X];

% Calculate the first layer response: a_2 = sigmoid(theta*a_1); 
a_2 = sigmoid(Theta1 * a_1')';

% Expand a_2: add a 1...
a_2 = [ones(m, 1) a_2];

% Calculate the final output layer response
out = sigmoid(Theta2 * a_2')';

% Pick the max predicted class, like in the predictOneVsAll...
[a, p] = max(out, [], 2);


% =========================================================================


end
