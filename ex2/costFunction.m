function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
 % number of training examples

% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
J = 0;
grad = zeros(size(theta));
m = length(y);
h=sigmoid(X*theta);

for i=1:m,
  J+=(y(i)*log(h(i)))+((1-y(i))*(log(1-h(i))));
endfor;
J=J*-1/m;


%gradient descent...


for i=1:1500,
   grad=(1/m)*((h-y)'*X)';
endfor;










% =============================================================

end
