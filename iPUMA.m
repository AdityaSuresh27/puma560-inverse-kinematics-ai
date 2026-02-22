function [J, P_plot, valid, config] = iPUMA(nx, ny, nz, ox, oy, oz, ax, ay, az, Px, Py, Pz, varargin)
% iPUMA - Inverse Kinematics for PUMA 560 Robot Manipulator (v2.2 FIXED)
%
% DESCRIPTION:
%   Solves the inverse kinematics problem for the PUMA 560 6-DOF robot
%   manipulator using analytical decoupling method with ALL 8 solution branches.
%
% INPUT PARAMETERS:
%   nx, ny, nz - Components of the normal vector (n-vector)
%   ox, oy, oz - Components of the orientation vector (o-vector)
%   ax, ay, az - Components of the approach vector (a-vector)
%   Px, Py, Pz - Position coordinates of the end-effector (mm)
%   varargin   - Optional: 'verbose', true/false (default: false)
%
% OUTPUT PARAMETERS:
%   J      - 1x6 vector of joint angles [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆] in degrees
%   P_plot - 8x3 matrix of frame positions for plotting
%   valid  - Boolean flag indicating if solution is within joint limits
%   config - Structure with configuration info (shoulder, elbow, wrist)
%
% FIXES IN v2.2:
%   - Removed MATLAB ternary operators (don't exist in MATLAB)
%   - Fixed θ₁ left-shoulder calculation (was: phi1 + alpha1 + 180)
%   - Removed excessive rounding that hurt ANN training
%   - Unified variable naming (alpha everywhere)
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
% Link twist angles (degrees)
alpha = [-90, 0, 90, -90, 90, 0];

% Link lengths (mm)
r = [0, 431.80, -20.32, 0, 0, 0];

% Link offsets along z-axis (mm)
d = [671.83, 139.70, 0, 431.80, 0, 56.50];

%% Joint Limits
joint_limits = [-160, 160;   % θ₁
                -225, 45;     % θ₂
                -45, 225;     % θ₃
                -110, 170;    % θ₄
                -100, 100;    % θ₅
                -266, 266];   % θ₆

%% Construct Homogeneous Transformation Matrix T06
T0_6 = [nx, ox, ax, Px;
        ny, oy, ay, Py;
        nz, oz, az, Pz;
        0,  0,  0,  1];

%% STEP 1: Calculate Wrist Center Position (Frame 5)
P5 = [Px - d(6) * ax;
      Py - d(6) * ay;
      Pz - d(6) * az];

if verbose
    fprintf('=== INVERSE KINEMATICS SOLUTION ===\n\n');
    fprintf('Wrist Center Position (Frame 5):\n');
    fprintf('  P5x = %.4f mm\n', P5(1));
    fprintf('  P5y = %.4f mm\n', P5(2));
    fprintf('  P5z = %.4f mm\n\n', P5(3));
end

%% STEP 2: Generate ALL 8 IK Solutions

solutions = [];
configs = [];

% Compute shared geometric quantities
C1 = sqrt(P5(1)^2 + P5(2)^2);
D1 = d(2) / C1;
alpha1 = atan2d(D1, sqrt(abs(1 - D1^2)));
phi1 = atan2d(P5(2), P5(1));

C2 = P5(3) - d(1);
C3 = sqrt(C1^2 + C2^2);
C4 = sqrt(r(3)^2 + d(4)^2);
phi2 = atan2d(C2, C1);

if verbose
    fprintf('Position Kinematics Setup:\n');
    fprintf('  C1 = %.4f mm (radial distance)\n', C1);
    fprintf('  α₁ = %.4f° (shoulder offset angle)\n', alpha1);
    fprintf('  φ₁ = %.4f° (wrist center angle)\n', phi1);
    fprintf('  C2 = %.4f mm, C3 = %.4f mm, C4 = %.4f mm\n\n', C2, C3, C4);
end

% Loop over 8 configurations
for shoulder_idx = 1:2  % Right (1) or Left (2)
    for elbow_idx = 1:2  % Down (1) or Up (2)
        for wrist_idx = 1:2  % No-flip (1) or Flip (2)
            
            % --- θ₁ (Shoulder: Right/Left) ---
            if shoulder_idx == 1  % Right shoulder
                theta1 = phi1 - alpha1;
                shoulder_sign = 1;
            else  % Left shoulder - FIXED: removed +180
                theta1 = phi1 + alpha1;
                shoulder_sign = -1;
            end
            
            % Normalize θ₁ to [-180, 180]
            theta1 = mod(theta1 + 180, 360) - 180;
            
            % --- θ₂ and θ₃ (Elbow: Down/Up) ---
            D2 = (C3^2 + r(2)^2 - C4^2) / (2 * r(2) * C3);
            D2 = max(-1, min(1, D2));  % Clamp for numerical stability
            
            D3 = (r(2)^2 + C4^2 - C3^2) / (2 * r(2) * C4);
            D3 = max(-1, min(1, D3));
            
            if elbow_idx == 1  % Elbow down
                alpha2 = atan2d(sqrt(abs(1 - D2^2)), D2);
                beta = atan2d(sqrt(abs(1 - D3^2)), D3);
                elbow_sign = 1;
            else  % Elbow up
                alpha2 = atan2d(-sqrt(abs(1 - D2^2)), D2);
                beta = atan2d(-sqrt(abs(1 - D3^2)), D3);
                elbow_sign = -1;
            end
            
            theta2 = alpha2 - phi2;  % REMOVED ROUNDING
            theta3 = beta - 90;
            
            % --- Compute T03 ---
            T0_3 = compute_T03(theta1, theta2, theta3, alpha, r, d);
            
            % --- Compute T36 ---
            T3_6 = T0_3 \ T0_6;  % More stable than inv(T0_3) * T0_6
            
            % --- θ₄, θ₅, θ₆ (Wrist orientation) ---
            % Check for wrist singularity
            if abs(abs(T3_6(3,3)) - 1) < 1e-6
                % Singularity: θ₅ ≈ 0° or ±180°
                theta5 = 0;
                theta4 = 0;
                theta6 = atan2d(T3_6(2,1), T3_6(1,1));
                wrist_sign = 0;  % Singular
            else
                if wrist_idx == 1  % No-flip
                    theta5 = atan2d(sqrt(abs(1 - T3_6(3,3)^2)), T3_6(3,3));
                    wrist_sign = 1;
                else  % Flip
                    theta5 = atan2d(-sqrt(abs(1 - T3_6(3,3)^2)), T3_6(3,3));
                    wrist_sign = -1;
                end
                
                theta4 = atan2d(T3_6(2,3), T3_6(1,3));  % REMOVED ROUNDING
                theta6 = atan2d(T3_6(3,2), -T3_6(3,1));  % REMOVED ROUNDING
            end
            
            % --- Store solution ---
            J_candidate = [theta1, theta2, theta3, theta4, theta5, theta6];
            
            % Check joint limits
            is_valid = all(J_candidate >= joint_limits(:,1)' & ...
                          J_candidate <= joint_limits(:,2)');
            
            if is_valid
                solutions = [solutions; J_candidate];
                configs = [configs; shoulder_sign, elbow_sign, wrist_sign];
            end
        end
    end
end

%% STEP 3: Select Best Solution
if isempty(solutions)
    % No valid solutions
    J = [NaN, NaN, NaN, NaN, NaN, NaN];
    P_plot = NaN(8, 3);
    valid = false;
    config = struct('shoulder', 0, 'elbow', 0, 'wrist', 0, ...
                    'description', 'No valid solution');
    
    if verbose
        fprintf('✗ No valid IK solutions within joint limits.\n\n');
    end
    return;
end

% Select first valid solution (could add preference logic here)
J = solutions(1, :);
selected_config = configs(1, :);

% Create config structure
config = struct();
config.shoulder = selected_config(1);
config.elbow = selected_config(2);
config.wrist = selected_config(3);

% FIXED: Replace ternary operators with if/else
if selected_config(1) > 0
    shoulder_str = 'Right';
else
    shoulder_str = 'Left';
end

if selected_config(2) > 0
    elbow_str = 'Down';
else
    elbow_str = 'Up';
end

if selected_config(3) > 0
    wrist_str = 'No-flip';
elseif selected_config(3) < 0
    wrist_str = 'Flip';
else
    wrist_str = 'Singular';
end

config.description = sprintf('%s / %s / %s', shoulder_str, elbow_str, wrist_str);

valid = true;

if verbose
    fprintf('=== SOLUTION SUMMARY ===\n');
    fprintf('Found %d valid solution(s)\n', size(solutions, 1));
    fprintf('Selected configuration: %s\n\n', config.description);
    
    fprintf('Joint Angles (degrees):\n');
    fprintf('  θ₁ = %10.4f°\n', J(1));
    fprintf('  θ₂ = %10.4f°\n', J(2));
    fprintf('  θ₃ = %10.4f°\n', J(3));
    fprintf('  θ₄ = %10.4f°\n', J(4));
    fprintf('  θ₅ = %10.4f°\n', J(5));
    fprintf('  θ₆ = %10.4f°\n\n', J(6));
end

%% STEP 4: Compute Frame Positions for Plotting
P_plot = compute_frame_positions(J, alpha, r, d, Px, Py, Pz);

if verbose
    fprintf('End-effector verification:\n');
    fprintf('  Desired:  [%.2f, %.2f, %.2f] mm\n', Px, Py, Pz);
    fprintf('  Achieved: [%.2f, %.2f, %.2f] mm\n', P_plot(7,1), P_plot(7,2), P_plot(7,3));
    fprintf('  Error: %.4f mm\n\n', norm([Px, Py, Pz] - P_plot(7,:)));
end

end

%% Helper Functions

function T0_3 = compute_T03(theta1, theta2, theta3, alpha, r, d)
    % Compute T03 transformation matrix
    theta = [theta1, theta2, theta3];
    T = cell(1, 3);
    
    for i = 1:3
        ct = cosd(theta(i));
        st = sind(theta(i));
        ca = cosd(alpha(i));
        sa = sind(alpha(i));
        
        T{i} = [ct, -st*ca,  st*sa, r(i)*ct;
                st,  ct*ca, -ct*sa, r(i)*st;
                0,   sa,     ca,    d(i);
                0,   0,      0,     1];
    end
    
    T0_3 = T{1} * T{2} * T{3};
end

function P_plot = compute_frame_positions(J, alpha, r, d, Px, Py, Pz)
    % Compute all frame positions for plotting
    P_plot = zeros(8, 3);
    P_plot(1,:) = [0, 0, 0];  % Base
    
    T = eye(4);
    for i = 1:6
        ct = cosd(J(i));
        st = sind(J(i));
        ca = cosd(alpha(i));
        sa = sind(alpha(i));
        
        Ti = [ct, -st*ca,  st*sa, r(i)*ct;
              st,  ct*ca, -ct*sa, r(i)*st;
              0,   sa,     ca,    d(i);
              0,   0,      0,     1];
        
        T = T * Ti;
        P_plot(i+1, :) = T(1:3, 4)';
    end
    
    P_plot(8,:) = [Px, Py, Pz];  % End-effector
end