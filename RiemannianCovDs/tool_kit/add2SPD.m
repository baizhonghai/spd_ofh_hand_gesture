function spd_Matrix = add2SPD(sym_Matrix,option)
    [~,d,num] = size(sym_Matrix);
    for sam_th = 1:num
        currenr_Sam = sym_Matrix(:,:,sam_th);
        [~,S] = eig(currenr_Sam);
        [temp_min,~] = min(diag(S));
        while temp_min <= option.min_EigCov
            currenr_Sam = currenr_Sam + (option.min_EigCov)*trace(currenr_Sam)*eye(d);
            [~,S] = eig(currenr_Sam);
            [temp_min,~] = min(diag(S));
        end 
        spd_Matrix(:,:,sam_th) = currenr_Sam;
    end

end

%{
This MATLAB function, add2SPD, appears to take a set of symmetric matrices 
and ensures that each matrix becomes symmetric positive definite (SPD) by adjusting its eigenvalues.
Let's break down the key components of the code:

Input Parameters:

sym_Matrix: A 3D array representing a set of symmetric matrices. Each slice along the third dimension (sam_th) is a symmetric matrix.
option: A structure containing various options and parameters, including min_EigCov, which is the threshold for the minimum eigenvalue.
Matrix Dimensions:

The function extracts the size of the input symmetric matrices using [~,d,num] = size(sym_Matrix).
d: The size of each matrix (assuming they are square matrices).
num: The number of matrices in the set.
Loop Over Matrices:

The function then iterates over each symmetric matrix in the set (sam_th).
Ensure Positive Definiteness:

For each matrix, it computes the eigenvalues using eig(currenr_Sam) and extracts the minimum eigenvalue.
It checks if the minimum eigenvalue is less than or equal to the specified threshold (option.min_EigCov).
If the minimum eigenvalue is below the threshold, it adjusts the matrix to ensure positive definiteness.
This is done by adding a scaled identity matrix to the matrix until the minimum eigenvalue becomes greater than option.min_EigCov.
The adjusted matrix is stored in spd_Matrix.
Output:

The resulting set of SPD matrices is stored in spd_Matrix, and the function returns this set.
This function is useful when dealing with covariance matrices, which should be symmetric positive definite.
The adjustment is performed to avoid numerical issues associated with near-zero or negative eigenvalues, 
ensuring that the matrices remain valid covariance or SPD matrices.
%}