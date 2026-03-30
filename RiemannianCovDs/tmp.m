%load('./data_RGB_1/mat_ETH/1/1/Local_reSize256M256_bSize32M32M1_stSize28M28M1_aG0.9_bG0.6_mC1e-09_mG1e-09_mR1e-09_A_DE_s.mat')
%sprintf("ok")
% Define a 3x3 matrix
A = [1, 2, 3;
     4, 5, 6;
     7, 8, 9];

% Calculate the trace of the matrix
trace_A = trace(A);

% Display the matrix and its trace
disp('Matrix A:');
disp(A);
fprintf('Trace of A: %d\n', trace_A);

load('/Users/baizhonghai/TP/RiemannianCovDs/data_RGB/com_ETH/1/1/Local_reSize256M256_bSize32M32M1_stSize28M28M1.mat')
% Define the matrices
r_log_RCovDs = [1 2; 3 4];
c_log_RCovDs = [5 6; 7 9];

% Calculate the Frobenius norm
result = norm((r_log_RCovDs - c_log_RCovDs), 'fro');

% Display the matrices and the result
disp('Matrix 1:');
disp(r_log_RCovDs);

disp('Matrix 2:');
disp(c_log_RCovDs);

disp(['Frobenius Norm of the Matrix Difference: ' num2str(result)]);
