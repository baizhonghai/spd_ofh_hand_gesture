%% Riemannian CovDs (Region overlappping)
% Author : Kai-Xuan Chen
% Date1: 2018.08.24
% Date2: 2018.12.11

clear;
clc;
%% set Hy-Parameters
option = set_Option('NG','Local','RGB');  
%{
            option.resized_Row = 256;                                           % row size of image
            option.resized_Col = 256;                                           % column size of image
            option.block_Row = 32;                                              % row size of block
            option.block_Col = 32;                                              % column size of block
            option.block_Depth = 1;
            option.step_Row = 28;                                               % row step of block
            option.step_Col = 28;  
   
%}
%{
%尝试新的块大小5*5, 51,52
% Block parameters
option.block_Row = 52;        % Row size of each block
option.block_Col = 52;        % Column size of each block
option.block_Depth = 1;       % Depth size of each block (not explicitly used in the given line of code)

% Step parameters
option.step_Row = 51;         % Row step of the blocks
option.step_Col = 51;         % Column step of the blocks
%}

%{
%尝试新的块大小15*15, 17,18
% Block parameters
option.block_Row = 18;        % Row size of each block
option.block_Col = 18;        % Column size of each block
option.block_Depth = 1;       % Depth size of each block (not explicitly used in the given line of code)

% Step parameters
option.step_Row = 17;         % Row step of the blocks
option.step_Col = 17;         % Column step of the blocks
%}

%{
%尝试新的块大小9*9, 
% Block parameters
option.block_Row = 80;        % Row size of each block
option.block_Col = 80;        % Column size of each block
option.block_Depth = 1;       % Depth size of each block (not explicitly used in the given line of code)

% Step parameters
option.step_Row = 20;         % Row step of the blocks
option.step_Col = 30;         % Column step of the blocks
%}

%{
%尝试新的块大小10*10, 
% Block parameters
option.block_Row = 60;        % Row size of each block
option.block_Col = 50;        % Column size of each block
option.block_Depth = 1;       % Depth size of each block (not explicitly used in the given line of code)

% Step parameters
option.step_Row = 20;         % Row step of the blocks
option.step_Col = 30;         % Column step of the blocks
%}

% input : 
%       param 1: the name of dataset
%       param 2: the kind of feature extracting:{Local,Sift,VGG}
%       param 3: color types of images: {Gray,RGB}
% output:
%       option : the struct of parameters 
%% extract Gauss Component(mean vectors & covariance matrices)
[option,num_Each_Matrix_GaussCom,time_Matrix_GaussCom] = extract_GaussComponent(option);
% input :
%       option : the struct of parameters
% output:
%       option : resetted struct of parameters which contains the paths of Gauss component
%       num_Each_Matrix_GaussCom : the matrix of the number of samples in each class
%       time_Matrix_GaussCom : the time consuming matrix
%% generating RieCovDs via 'Riemannian metric' & 'gaussian embedding'
%type_Metric_Gauss = {{'DE','A'},{'DE','S'},{'DE','J'},{'DE','L'},...
 %                   {'IE','A'},{'IE','S'},{'IE','J'},{'IE','L'}};
type_Metric_Gauss ={{'IE','L'}};
% type_Metric_Gauss = {{'DE','L'},{'IE','L'}};
%       DE/IE : compute Riemannian Local Difference Vector(RieLDV) directly/indirectly on the manifold of Gaussians
%       A/S/J/L : use AIRM/S-divergence/J-divergence/Log-Euclidean Metric to commpute RieLDV
%pool = parpool(4);
%parfor i = 1:length(type_Metric_Gauss)
for i = 1:length(type_Metric_Gauss)
    rie_Metric = type_Metric_Gauss{i}{1,2}; 
    type_Gauss = type_Metric_Gauss{i}{1,1};
    new_Opt = set_OptionPath(option,rie_Metric,type_Gauss); % reset option via 'Riemannian metric' & 'gaussian embedding'
    [num_Each_Matrix,time_Matrix] = gen_RCovDs(new_Opt);    % generate Riemannian CovDs  
    if ~exist(new_Opt.dis_Matrix_Path,'file')
        dis_Matrix = compute_Dis(new_Opt);                  % generate distance matrix, if it does not exist.
    end
end
%delete(pool);