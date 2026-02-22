function [T06, positions] = fPUMA(theta1, theta2, theta3, theta4, theta5, theta6, varargin)
% fPUMA - Forward Kinematics for PUMA 560 Robot Manipulator (v2.2 FIXED)
%
% DESCRIPTION:
%   Computes the forward kinematics for the PUMA 560 6-DOF robot
%   manipulator using Denavit-Hartenberg convention.
%
% INPUT PARAMETERS:
%   theta1-theta6 - Joint angles in degrees
%   varargin      - Optional: 'verbose', true/false (default: false)
%
% OUTPUT PARAMETERS:
%   T06       - 4x4 homogeneous transformation matrix
%   positions - 7x3 matrix of frame positions
%
% FIXES IN v2.2:
%   - Added verbose flag for silent operation
%   - Unified variable naming (alpha, not A)
%   - Precomputed trig for efficiency
%
% Author: Robotics Team
% Date: February 2026
% Version: 2.2 (FIXED)

%% Parse input arguments
p = inputParser;
addParameter(p, 'verbose', false, @islogical);
parse(p, varargin{:});
verbose = p.Results.verbose;

%% Setup
if ~verbose
    format compact
    format short
end

%% DH Parameters for PUMA 560 (unified naming)
theta = [theta1, theta2, theta3, theta4, theta5, theta6];
alpha = [-90, 0, 90, -90, 90, 0];  % Link twist (degrees) - UNIFIED NAME
r = [0, 431.80, -20.32, 0, 0, 0];  % Link lengths (mm)
d = [671.83, 139.70, 0, 431.80, 0, 56.50];  % Link offsets (mm)

if verbose
    fprintf('=== FORWARD KINEMATICS ===\n\n');
    fprintf('Input Joint Angles (degrees):\n');
    for i = 1:6
        fprintf('  θ₁ = %8.4f°\n', theta(i));
    end
    fprintf('\n');
end

%% Construct Transformation Matrices (with precomputed trig)
T = cell(1, 6);

for i = 1:6
    ct = cosd(theta(i));
    st = sind(theta(i));
    ca = cosd(alpha(i));
    sa = sind(alpha(i));
    
    T{i} = [ct, -st*ca,  st*sa, r(i)*ct;
            st,  ct*ca, -ct*sa, r(i)*st;
            0,   sa,     ca,    d(i);
            0,   0,      0,     1];
    
    if verbose
        fprintf('T%d%d matrix computed.\n', i-1, i);
    end
end

if verbose
    fprintf('\n');
end

%% Compute Cumulative Transformations
T_cumulative = cell(1, 6);
T_cumulative{1} = T{1};

for i = 2:6
    T_cumulative{i} = T_cumulative{i-1} * T{i};
    if verbose
        fprintf('T0%d matrix computed.\n', i);
    end
end

if verbose
    fprintf('\n');
end

%% Extract Final Transformation T06
T06 = T_cumulative{6};

%% Display Results
if verbose
    fprintf('=== TRANSFORMATION MATRIX T06 ===\n\n');
    fprintf('T06 = \n');
    for i = 1:4
        fprintf('  [%8.4f  %8.4f  %8.4f  %10.4f]\n', T06(i,:));
    end
    fprintf('\n');
    
    % Extract vectors
    n_vector = T06(1:3, 1);
    o_vector = T06(1:3, 2);
    a_vector = T06(1:3, 3);
    p_vector = T06(1:3, 4);
    
    fprintf('Orientation and Position Vectors:\n');
    fprintf('  n (normal):      [%8.4f, %8.4f, %8.4f]\n', n_vector);
    fprintf('  o (orientation): [%8.4f, %8.4f, %8.4f]\n', o_vector);
    fprintf('  a (approach):    [%8.4f, %8.4f, %8.4f]\n', a_vector);
    fprintf('  P (position):    [%8.4f, %8.4f, %8.4f] mm\n\n', p_vector);
end

%% Extract Frame Positions
positions = zeros(7, 3);
positions(1,:) = [0, 0, 0];

for i = 1:6
    positions(i+1, :) = T_cumulative{i}(1:3, 4)';
end

if verbose
    fprintf('Frame Positions (mm):\n');
    fprintf('  Base:    [%8.2f, %8.2f, %8.2f]\n', positions(1,:));
    for i = 1:6
        fprintf('  Frame %d: [%8.2f, %8.2f, %8.2f]\n', i, positions(i+1,:));
    end
    fprintf('\n');
    
    % Verify rotation matrix
    R = T06(1:3, 1:3);
    det_R = det(R);
    ortho_error = norm(R' * R - eye(3), 'fro');
    
    fprintf('Rotation Matrix Verification:\n');
    fprintf('  Determinant: %.6f (should be 1.0)\n', det_R);
    fprintf('  Orthogonality error: %.6e (should be ~0)\n\n', ortho_error);
end

end