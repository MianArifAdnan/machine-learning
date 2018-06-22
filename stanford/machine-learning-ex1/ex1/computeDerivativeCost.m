function dJ = computeDerivativeCost(X, y, theta)
%COMPUTEDERIVATIVECOST Compute derivative cost for linear regression
%   dJ = COMPUTEDERIVATIVECOST(X, y, theta, idx)

% Initialize some useful values
m = length(y); % number of training examples

h = X * theta; % This is the equation because of variable dims

dJ = (1 / m) * (X' * (h - y));

end
