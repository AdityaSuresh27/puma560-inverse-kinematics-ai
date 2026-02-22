function [T06, positions] = fPUMA(theta1, theta2, theta3, theta4, theta5, theta6, varargin)
% fPUMA - Forward Kinematics for PUMA 560 Robot Manipulator (v2.3)
%
% DESCRIPTION:
%   Computes the forward kinematics for the PUMA 560 6-DOF robot
%   manipulator using Denavit-Hartenberg convention.
%
% SYNTAX:
%   [T06, positions] = fPUMA(theta1, theta2, theta3, theta4, theta5, theta6)
%   [T06, positions] = fPUMA(theta1, theta2, theta3, theta4, theta5, theta6, verbose)
%
% INPUT PARAMETERS:
%   theta1-theta6 - Joint angles in degrees
%   verbose       - (optional) true/false, default false
%
% OUTPUT PARAMETERS:
%   T06       - 4x4 homogeneous transformation matrix
%   positions - 7x3 matrix of frame origin positions [base; frame1..6]
%
% FIXES IN v2.3:
%   - varargin now handles positional bool (not inputParser name-value)
%   - Fixed fprintf loop that always printed theta1 label for all joints
%
% Author: Robotics Team
% Date: February 2026
% Version: 2.3

%% Parse varargin: first optional arg is verbose flag
verbose = false;
if nargin >= 7 && ~isempty(varargin{1})
    verbose = logical(varargin{1});
end

%% DH Parameters for PUMA 560
theta_vec = [theta1, theta2, theta3, theta4, theta5, theta6];
alpha = [-90,  0,  90, -90,  90,  0];   % Link twist angles (degrees)
r     = [0, 431.80, -20.32, 0, 0, 0];   % Link lengths (mm)
d     = [671.83, 139.70, 0, 431.80, 0, 56.50];  % Link offsets (mm)

if verbose
    fprintf('=== FORWARD KINEMATICS (fPUMA v2.3) ===\n\n');
    fprintf('Input Joint Angles:\n');
    for i = 1:6
        fprintf('  theta%d = %8.4f deg\n', i, theta_vec(i));  % FIXED: was always theta1
    end
    fprintf('\n');
end

%% Compute individual DH transformation matrices
T_cell = cell(1, 6);
for i = 1:6
    ct = cosd(theta_vec(i));
    st = sind(theta_vec(i));
    ca = cosd(alpha(i));
    sa = sind(alpha(i));

    T_cell{i} = [ct, -st*ca,  st*sa, r(i)*ct;
                 st,  ct*ca, -ct*sa, r(i)*st;
                  0,     sa,    ca,    d(i);
                  0,      0,     0,       1];
end

%% Cumulative transformations T01, T02, ..., T06
T_cum = cell(1, 6);
T_cum{1} = T_cell{1};
for i = 2:6
    T_cum{i} = T_cum{i-1} * T_cell{i};
end

%% Final result
T06 = T_cum{6};

%% Extract frame origin positions
positions = zeros(7, 3);
positions(1,:) = [0, 0, 0];  % Base frame
for i = 1:6
    positions(i+1,:) = T_cum{i}(1:3, 4)';
end

%% Verbose output
if verbose
    fprintf('Transformation Matrix T06:\n');
    for row = 1:4
        fprintf('  [%10.6f  %10.6f  %10.6f  %12.6f]\n', T06(row,:));
    end
    fprintf('\n');

    n = T06(1:3,1);  o = T06(1:3,2);
    a = T06(1:3,3);  p = T06(1:3,4);
    fprintf('Vectors:\n');
    fprintf('  n (normal):      [%8.4f, %8.4f, %8.4f]\n', n);
    fprintf('  o (orientation): [%8.4f, %8.4f, %8.4f]\n', o);
    fprintf('  a (approach):    [%8.4f, %8.4f, %8.4f]\n', a);
    fprintf('  P (position):    [%8.4f, %8.4f, %8.4f] mm\n\n', p);

    fprintf('Frame Positions (mm):\n');
    fprintf('  Base:    [%8.2f, %8.2f, %8.2f]\n', positions(1,:));
    for i = 1:6
        fprintf('  Frame %d: [%8.2f, %8.2f, %8.2f]\n', i, positions(i+1,:));
    end
    fprintf('\n');

    % Rotation matrix sanity check
    R = T06(1:3, 1:3);
    det_R = det(R);
    orth_err = norm(R' * R - eye(3), 'fro');
    fprintf('Rotation Matrix Check:\n');
    fprintf('  det(R)        = %.8f  (should be 1.0)\n', det_R);
    fprintf('  Orthogonality = %.2e  (should be ~0)\n\n', orth_err);
end

end