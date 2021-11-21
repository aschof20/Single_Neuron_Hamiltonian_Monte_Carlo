% Hamiltonian Monte Carlo Single Neuron Model.

% Load dataset.
load A3.dat
% Predictor variables.
X = [ones(8,1) A3(:,1:2)];
% Response variable.
t = A3(:,3);
% Set seed.
rng(10);

%%%%%%%%%%%%%%%%%% Algorithm Parameters %%%%%%%%%%%%%%%%%%
% Weight Decay Rate.
alpha = 0.1;
% Learning Rate.
eta = 0.01;
% Hamiltonian Step Size.
epsilon = sqrt(2* eta);
% 'Leapfrogs'
Tau = 9;
% Lag to generate independent samples.
lag = 5;
% Initial steps.
burn_in = 10000;
% Total number of steps.
T = burn_in + 30*lag;
% Variable to calculate the acceptance rate of the Monte Carlo.
accepted = 0;
% Initialize the weight vector.
W = [0 0 0];
% Matrix to store the weight values.
W_stored = zeros(T, 3);
% Initialize the matrix.
W_stored(1,:) = W;

%%%%%%%%%%%%%%%%%% Algorithm Functions %%%%%%%%%%%%%%%%%%
% Function to calculate predicted values base on weights.
y = @(W) sigmf(W*X',[1 0]);

% Function to calculate the gradient of the objective function.
grad = @(W)  -(t' - y(W))*X + alpha*W;  

% Function to calculate the Objective Function with weight decay
% regularization.
M = @(W) -(t'*log(y(W)') + (1-t')*log(1-y(W))') + alpha*sum(W.^2, 2)'/2;
 
% Initialize the gradient and objective function. 
gradW = grad(W); 
M_w = M(W);   

% Loop T times.
for i = 1:T
    % Intialize momentum to random N~(0,1).
    p = randn(size(W));
    % Evaluate the H(w,p) probability.
    H = p*p'/2 +M_w;
 
    % Initialise the new weight with the current weight.
    Wprime = W;
    % Initialize the new gradient with the current gradient.
    gradW_new = gradW;
   
    % Make Tau 'leapfrogs' steps.
    for j = 1: Tau
        
        % Half step in p.
        p = p - (epsilon*gradW_new/2);
        
        % Half step in w.
        Wprime = Wprime + epsilon*p;
 
        % Calculate the new gradient.
        gradW_new =grad(Wprime);
        
        % Half step in p.
        p = p-epsilon*gradW_new/2;
 
    end
   
    % Calculate the new objective function.
    Mnew = M(Wprime);
    
    % Calculate the new value of H.
    Hnew = p'*p/2 + Mnew;
   
    % Calculate the difference between new and old H.
    dH = Hnew-H;
 
    % Evaluate if the step is accepted.
    if dH < 0
        accept = 1;
    elseif rand() < exp(-dH)
        accept = 1;
    else
        accept = 0;
    end
    
    % If accepted update current values to new values.
    if accept
        W = Wprime;
        M_w = Mnew;
        gradW = gradW_new;
    end
    
    % Increment the accepted variable.
    accepted = accepted + accept;
    % Add new weight values to a matrix.
    W_stored(i+1,:) = W;
end

% Calculate the acceptance rate.
acceptance_rate = accepted/(T-1)

% Generate matrix of 30 independent samples of neuronal ouput.
W_indep = W_stored(burn_in+lag:lag:T,:);

% Sum sampled output functions to find average neuron output.
learned_y = @(x) zeros(1, length(x));
for i = 1:length(W_indep)
    W = W_indep(i,:);
    
    learned_y = @(x) [learned_y(x); sigmf(W*x',[1 0])];
end
learned_y = @(x) sum(learned_y(x))/length(W_indep);

% Plots the decision boundary and autocorrelation.
% Sample autocorrelation.
figure(1);  
acf(W_stored(:,2), lag); hold off

% Decision Boundary.
figure(2);
% Plot the A3 dataset data points.
plot(X(1:4,2),X(1:4,3),'bo', 'MarkerFaceColor', 'b'); hold on
plot(X(5:8,2),X(5:8,3),'ro', 'MarkerFaceColor', 'r')
xlim([0 10]); ylim([0 10]); axis square
xlabel('x1'); ylabel('x2'); hold on

% Generate contours based on average neuron output of independent samples.
x1 = linspace(0,10);
x2 = x1;
[x1 x2] = meshgrid(x1, x2);
learned_y_cont = reshape(learned_y([ones(10000,1) x1(:) x2(:)]), 100, 100);
contour(x1, x2, learned_y_cont ,[0.12 0.27 0.73 0.88],  "--k"); hold on
contour(x1, x2, learned_y_cont, [0.5 0.5], "k" ); hold off