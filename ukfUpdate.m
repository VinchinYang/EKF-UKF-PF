function [mu, Sigma, predMu, predSigma, zHat, K ] = ukfUpdate( ...
    mu, Sigma, u, deltaT, M, z, Q, markerId)

% NOTE: The header is not set in stone.  You may change it if you like.
global FIELDINFO;
landmark_x = FIELDINFO.MARKER_X_POS(markerId);
landmark_y = FIELDINFO.MARKER_Y_POS(markerId);

stateDim=3;
motionDim=3;
observationDim=2;

% --------------------------------------------
% Setup UKF
% --------------------------------------------
% mu: [x;y;theta] Sigma: 3x3 matrix  u:[drot1;dtrans;drot2] M:3x3 matrix
% M is control noise covariance
% Q:2x2 matrix
% UKF params
augStateDim = stateDim + motionDim + observationDim;
alfa = 0.001; % alfa and k are the scaling parameters that determine how far
k = 0;    %the sigma points are spread from the mean.
lamda = alfa*alfa*(augStateDim+k)-augStateDim;
beda = 2;%optimal value for gausian noise 
% Augmented state
mu_a = [mu' 0 0 0 0 0]';
Sigma_a = blkdiag(Sigma, M, Q);%3+3+2 x 3+3+2

% Sigma points
numPoints = 2*augStateDim+1;
SigmaPoints = zeros(augStateDim, numPoints);
SigmaPoints(:,1) = mu_a;
L = chol((lamda + augStateDim)*Sigma_a, 'lower');
for ii = 1:augStateDim
    SigmaPoints(:,ii+1) = mu_a + L(:,ii);
    %comment this protection will not affect the results in this case,
    %but I think that it's good to use it.
    %SigmaPoints(3,ii+1) = minimizedAngle(SigmaPoints(3,ii+1));
    %SigmaPoints(4,ii+1) = minimizedAngle(SigmaPoints(4,ii+1));
    %SigmaPoints(6,ii+1) = minimizedAngle(SigmaPoints(6,ii+1));
    %SigmaPoints(7,ii+1) = minimizedAngle(SigmaPoints(7,ii+1));
end
for ii = augStateDim+1:2*augStateDim
    SigmaPoints(:,ii+1) = mu_a - L(:,ii-augStateDim);
    %comment this protection will not affect the results in this case,
    %but I think that it's good to use it.
    %SigmaPoints(3,ii+1) = minimizedAngle(SigmaPoints(3,ii+1));
    %SigmaPoints(4,ii+1) = minimizedAngle(SigmaPoints(4,ii+1));
    %SigmaPoints(6,ii+1) = minimizedAngle(SigmaPoints(6,ii+1));
    %SigmaPoints(7,ii+1) = minimizedAngle(SigmaPoints(7,ii+1));
end

% Weights
meanWeight = zeros(numPoints,1);
covarianceWeight = zeros(numPoints,1);
meanWeight(1) = lamda/(lamda+augStateDim);
covarianceWeight(1) = meanWeight(1) + (1-alfa^2+beda);
for ii = 1:2*augStateDim
    meanWeight(ii+1) = 1/(2*(lamda+augStateDim));
    covarianceWeight(ii+1) = meanWeight(ii+1);
end


% --------------------------------------------
% Prediction step
% --------------------------------------------

% UKF prediction of mean and covariance

% muHat = [x + dtrans*cos(theta + drot1);
%          y + dtrans*sin(theta + drot1);
%         theta + drot1 + drot2;]
muPred = zeros(stateDim,numPoints);
muHat = zeros(stateDim,1);
SigmaHat = zeros(stateDim, stateDim);
for ii = 1:numPoints
    uTemp = u+SigmaPoints(4:6,ii);
    %muPred(:,ii) = SigmaPoints(1:3,ii) + [uTemp(2)*cos(SigmaPoints(3,ii)+uTemp(1));
        %uTemp(2)*sin(SigmaPoints(3,ii)+uTemp(1));
        %uTemp(1) + uTemp(3)];
    %muPred(3,ii) = minimizedAngle(muPred(3,ii));
    muPred(:,ii) = prediction(SigmaPoints(1:3,ii),uTemp);
    muHat = muHat + meanWeight(ii)*muPred(:,ii);
    muHat(3) = minimizedAngle(muHat(3));
end

for ii = 1:numPoints
    SigmaHat = SigmaHat + covarianceWeight(ii)*(muPred(1:3,ii)-muHat)*(muPred(1:3,ii)-muHat)';
end
%--------------------------------------------------------------
% Correction step
%--------------------------------------------------------------

% UKF correction of mean and covariance
zBar =zeros(observationDim,numPoints); 
zHat = 0;
for ii = 1:numPoints
    zBar(:,ii)  = observation(muPred(:,ii), markerId)+SigmaPoints(7:8,ii);%[bearing angle; markerId]
    zBar(1,ii) = minimizedAngle(zBar(1,ii));
    zHat = zHat + zBar(:,ii)*meanWeight(ii);
    zHat(1) = minimizedAngle(zHat(1));
end
%zHat(1) = minimizedAngle(zHat(1));
S = zeros(observationDim,observationDim);
crossCovariance = zeros(stateDim,observationDim);
for ii = 1:numPoints
    S = S + covarianceWeight(ii)*(zBar(:,ii)-zHat)*(zBar(:,ii)-zHat)';
    crossCovariance = crossCovariance + covarianceWeight(ii)*(muPred(1:stateDim,ii)-muHat)*(zBar(:,ii)-zHat)';
end

K = crossCovariance/S;
mu = muHat + K*(z-zHat);
Sigma = SigmaHat - K*S*K';
predMu = muHat;
predSigma = SigmaHat;