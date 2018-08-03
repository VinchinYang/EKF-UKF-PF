function [mu, Sigma, predMu, predSigma, zhat, G, R, H, K ] = ekfUpdate( ...
    mu, Sigma, u, deltaT, M, z, Q, markerId)

% NOTE: The header is not set in stone.  You may change it if you like.
global FIELDINFO;
landmark_x = FIELDINFO.MARKER_X_POS(markerId);
landmark_y = FIELDINFO.MARKER_Y_POS(markerId);

stateDim=3;
motionDim=3;
observationDim=2;

% --------------------------------------------
% Prediction step
% --------------------------------------------

% EKF prediction of mean and covariance

% M is the model covariance matrix
%[alfas(1)*drot1^2+alfas(2)*dtrans^2  0  0;
% 0  alfas(3)*dtrans^2+alfas(4)*drot1^2+alfsa(4)*drot2^2  0;
% 0  0  alfas(1)*drot2^2+alfas(2)*dtrans^2]

% mu = [x; y; theta]  the previous state estimation
% Sigma               the previous state covarianve
% u  = [drot1; dtrans; drot2]
% motion model:
% mu = [x + dtrans*cos(theta + drot1);
%       y + dtrans*sin(theta + drot1);
%       theta + drot1 + drot2]
muPrev = mu;
SigmaPre = Sigma;
mu = prediction(muPrev, u);

% state Jacobian dmu/dx dmu/dy dmu/dtheta
G = [1 0 -u(2)*sin(muPrev(3)+u(1));
     0 1 u(2)*cos(muPrev(3)+u(1));
     0 0 1];
% inpute Jacobian dmu/drot1 dmu/ddtrans dmu/ddrot2
JV = [-u(2)*sin(muPrev(3)+u(1)) cos(muPrev(3)+u(1)) 0;
      u(2)*cos(muPrev(3)+u(1)) sin(muPrev(3)+u(1)) 0;
      1                       0                  1];
R = JV;
Sigma = G*SigmaPre*G' + JV*M*JV';
predMu = mu;
predSigma = Sigma;
%--------------------------------------------------------------
% Correction step
%--------------------------------------------------------------

% Compute expected observation and Jacobian
zhat  = observation(mu, markerId);%[bearing angle; markerId]
dx = landmark_x - mu(1);
dy = landmark_y - mu(2);
distSquar = dx^2 + dy^2;
% observationDim x stateDim = 2x3
H = [dy/distSquar -dx/distSquar -1;
      0 0 0];
% Innovation / residual covariance

S = H*Sigma*H' + Q;

% Kalman gain
K = Sigma*H'/S;
% Correction
mu = mu + K*(z-zhat);
Sigma = Sigma - K*H*Sigma;
