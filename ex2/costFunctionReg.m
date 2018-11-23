function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%  theta=initial_theta;             derivatives of the cost w.r.t. each parameter in theta 

J = 0;
pen=0;
grad = zeros(size(theta));
z=X*theta;
h=sigmoid(z);
for i=1:m,
  J+=(-y(i)*(log(h(i))))-((1-y(i))*(log(1-h(i))));
endfor;
J/=m;
  
 for j=2:length(theta),
    pen+=lambda*theta(j)*theta(j);
endfor;
pen/=2*m;
J+=pen;




theta_1=theta;
theta_1(1)=0;

grad=((1/m)*(h-y)'*X)+(lambda/m)*theta_1';





% =============================================================

end
