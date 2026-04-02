function [J, P_plot, valid, config] = iPUMA(nx, ny, nz, ox, oy, oz, ax, ay, az, Px, Py, Pz, varargin)
% iPUMA - Inverse Kinematics for PUMA 560 Robot Manipulator (v2.4)
%
% DESCRIPTION:
%   Solves the inverse kinematics for the PUMA 560 6-DOF robot using
%   analytical decoupling with all 8 solution branches.
%
% SYNTAX:
%   [J, P_plot, valid, config] = iPUMA(nx,ny,nz, ox,oy,oz, ax,ay,az, Px,Py,Pz)
%   [J, P_plot, valid, config] = iPUMA(..., verbose)
%   [J, P_plot, valid, config] = iPUMA(..., verbose, config_select)
%
% INPUTS:
%   nx,ny,nz       - Normal vector components of end-effector frame
%   ox,oy,oz       - Orientation vector components
%   ax,ay,az       - Approach vector components
%   Px,Py,Pz       - End-effector position (mm)
%   verbose        - (optional) true/false, default false
%   config_select  - (optional) 1-8 to select a specific branch;
%                    0 or omitted = return first valid branch
%
% OUTPUTS:
%   J      - 1x6 joint angles [theta1..theta6] in degrees
%   P_plot - 8x3 frame positions for plotting
%   valid  - true if a solution was found within joint limits
%   config - struct with branch/shoulder/elbow/wrist info
%
% BRANCH NUMBERING (shoulder x elbow x wrist):
%   1: Right / Down  / No-flip
%   2: Right / Down  / Flip
%   3: Right / Up    / No-flip
%   4: Right / Up    / Flip
%   5: Left  / Down  / No-flip
%   6: Left  / Down  / Flip
%   7: Left  / Up    / No-flip
%   8: Left  / Up    / Flip
%
% Author: Robotics Team
% Date: February 2026
% Version: 2.4

%% Parse optional arguments
verbose       = false;
config_select = 0;

if nargin >= 13 && ~isempty(varargin{1})
    verbose = logical(varargin{1});
end
if nargin >= 14 && ~isempty(varargin{2})
    config_select = round(double(varargin{2}));
end

%% PUMA 560 DH Parameters
alpha_dh = [-90,  0,  90, -90,  90,  0];   % Link twist angles (deg)
r_dh     = [0, 431.80, -20.32, 0, 0, 0];   % Link lengths (mm)
d_dh     = [671.83, 139.70, 0, 431.80, 0, 56.50]; % Link offsets (mm)

%% Joint Limits (degrees)
jlim = [-160,  160;   % theta1
        -225,   45;   % theta2
         -45,  225;   % theta3
        -110,  170;   % theta4
        -100,  100;   % theta5
        -266,  266];  % theta6

%% Target transformation matrix
T0_6 = [nx, ox, ax, Px;
        ny, oy, ay, Py;
        nz, oz, az, Pz;
         0,  0,  0,  1];

%% Wrist center (Frame 5 origin)
P5 = [Px - d_dh(6)*ax;
      Py - d_dh(6)*ay;
      Pz - d_dh(6)*az];

if verbose
    fprintf('=== INVERSE KINEMATICS (iPUMA v2.4) ===\n\n');
    fprintf('Wrist Center P5 = [%.4f, %.4f, %.4f] mm\n\n', P5(1), P5(2), P5(3));
end

%% theta1 geometry (radial plane)
C1_rad = sqrt(P5(1)^2 + P5(2)^2);
D1     = d_dh(2) / C1_rad;

if abs(D1) > 1
    if verbose, fprintf('Out of workspace: D1=%.4f > 1\n', D1); end
    J = nan(1,6); P_plot = nan(8,3); valid = false;
    config = struct('shoulder',0,'elbow',0,'wrist',0,'branch',0,'description','Out of workspace');
    return;
end

alpha1 = atan2d(D1, sqrt(1 - D1^2));   % Shoulder offset (positive value)
phi1   = atan2d(P5(2), P5(1));         % Wrist center azimuth

% phi3: geometric offset from link-3 twist (r3/d4 coupling)
phi3 = atan2d(r_dh(3), d_dh(4));

if verbose
    fprintf('theta1 geometry: phi1=%.4f, alpha1=%.4f\n', phi1, alpha1);
    fprintf('Link-3 offset: phi3=%.4f\n\n', phi3);
end

%% Generate all 8 solutions
all_J     = nan(8, 6);
all_valid = false(8, 1);
all_signs = zeros(8, 3);  % [shoulder, elbow, wrist]

branch = 0;
for shoulder_idx = 1:2
    for elbow_idx = 1:2
        for wrist_idx = 1:2
            branch = branch + 1;

            %--- theta1 ---
            if shoulder_idx == 1    % Right shoulder
                theta1 = phi1 - alpha1;
                s_sign = 1;
            else                    % Left shoulder (negative sqrt branch)
                theta1 = phi1 + alpha1 - 180;
                s_sign = -1;
            end
            theta1 = wrap180(theta1);

            %--- theta2, theta3 ---
            % C1 must be the distance projected along theta1 direction
            % (signed, in the shoulder plane) -- NOT the radial distance
            t1_rad = deg2rad(theta1);
            C1 = P5(1)*cos(t1_rad) + P5(2)*sin(t1_rad);
            C2 = P5(3) - d_dh(1);
            C3 = sqrt(C1^2 + C2^2);
            C4 = sqrt(r_dh(3)^2 + d_dh(4)^2);
            phi2 = atan2d(C2, C1);

            D2 = (C3^2 + r_dh(2)^2 - C4^2) / (2 * r_dh(2) * C3);
            D3 = (r_dh(2)^2 + C4^2 - C3^2) / (2 * r_dh(2) * C4);

            % Clamp for numerical noise at workspace boundary
            D2 = max(-1, min(1, D2));
            D3 = max(-1, min(1, D3));

            if elbow_idx == 1   % Elbow down
                alpha2 = atan2d( sqrt(1 - D2^2), D2);
                beta   = atan2d( sqrt(1 - D3^2), D3);
                e_sign = 1;
            else                % Elbow up
                alpha2 = atan2d(-sqrt(1 - D2^2), D2);
                beta   = atan2d(-sqrt(1 - D3^2), D3);
                e_sign = -1;
            end

            theta2 = alpha2 - phi2;       % Correct PUMA 560 convention
            theta3 = beta   - 90 - phi3;  % Include link-3 geometry offset

            %--- T03 and T36 ---
            T0_3 = compute_T03(theta1, theta2, theta3, alpha_dh, r_dh, d_dh);
            T3_6 = T0_3 \ T0_6;

            %--- theta4, theta5, theta6 (ZYZ Euler decomposition) ---
            sin5_sq = 1 - T3_6(3,3)^2;

            if sin5_sq < 1e-8
                % Wrist singularity: theta5 ~ 0
                theta5 = 0;
                theta4 = 0;
                theta6 = atan2d(T3_6(2,1), T3_6(1,1));
                w_sign = 0;
            else
                if wrist_idx == 1   % No-flip
                    theta5 = atan2d( sqrt(sin5_sq), T3_6(3,3));
                    w_sign = 1;
                else                % Flip
                    theta5 = atan2d(-sqrt(sin5_sq), T3_6(3,3));
                    w_sign = -1;
                end
                theta4 = atan2d( T3_6(2,3),  T3_6(1,3));
                theta6 = atan2d( T3_6(3,2), -T3_6(3,1));
            end

            J_cand = [theta1, theta2, theta3, theta4, theta5, theta6];

            % Check joint limits
            in_limits = all(J_cand >= jlim(:,1)' & J_cand <= jlim(:,2)');

            all_J(branch, :)     = J_cand;
            all_valid(branch)    = in_limits;
            all_signs(branch, :) = [s_sign, e_sign, w_sign];
        end
    end
end

%% Select solution
if config_select >= 1 && config_select <= 8
    if all_valid(config_select)
        chosen = config_select;
    else
        if verbose
            fprintf('Config %d is outside joint limits.\n', config_select);
        end
        J = nan(1,6); P_plot = nan(8,3); valid = false;
        config = struct('shoulder',0,'elbow',0,'wrist',0,'branch',config_select, ...
                        'description',sprintf('Config %d out of limits', config_select));
        return;
    end
else
    valid_list = find(all_valid);
    if isempty(valid_list)
        if verbose, fprintf('No valid IK solutions within joint limits.\n'); end
        J = nan(1,6); P_plot = nan(8,3); valid = false;
        config = struct('shoulder',0,'elbow',0,'wrist',0,'branch',0,'description','No valid solution');
        return;
    end
    chosen = valid_list(1);
end

J     = all_J(chosen, :);
valid = true;

%% Config structure
s = all_signs(chosen, :);
if s(1) > 0,   s_str = 'Right';
else,           s_str = 'Left';    end
if s(2) > 0,   e_str = 'Down';
else,           e_str = 'Up';      end
if s(3) > 0,   w_str = 'No-flip';
elseif s(3)<0, w_str = 'Flip';
else,           w_str = 'Singular'; end

config = struct();
config.shoulder    = s(1);
config.elbow       = s(2);
config.wrist       = s(3);
config.branch      = chosen;
config.description = sprintf('Config %d: %s / %s / %s', chosen, s_str, e_str, w_str);

%% Frame positions for plotting
P_plot = compute_frame_positions(J, alpha_dh, r_dh, d_dh, Px, Py, Pz);

%% Verbose output
if verbose
    fprintf('Valid solutions: %d / 8\n', sum(all_valid));
    fprintf('Selected: %s\n\n', config.description);
    fprintf('Joint Angles:\n');
    for i = 1:6
        fprintf('  theta%d = %10.4f deg\n', i, J(i));
    end
    fprintf('\nEnd-effector verification:\n');
    fprintf('  Desired:  [%.4f, %.4f, %.4f] mm\n', Px, Py, Pz);
    fprintf('  Achieved: [%.4f, %.4f, %.4f] mm\n', P_plot(7,1), P_plot(7,2), P_plot(7,3));
    fprintf('  Error:    %.6f mm\n\n', norm([Px,Py,Pz] - P_plot(7,:)));
    fprintf('All branches:\n');
    fprintf('  Br | Valid | theta1    theta2    theta3    theta4    theta5    theta6\n');
    fprintf('  ---|-------|----------------------------------------------------------\n');
    for b = 1:8
        if all_valid(b), v='YES'; else, v='no '; end
        fprintf('  %d  |  %s  |', b, v);
        fprintf(' %8.3f', all_J(b,:));
        fprintf('\n');
    end
    fprintf('\n');
end

end

%% -------------------------------------------------------------------------
function angle = wrap180(angle)
    angle = mod(angle + 180, 360) - 180;
end

%% -------------------------------------------------------------------------
function T0_3 = compute_T03(theta1, theta2, theta3, alpha, r, d)
    theta = [theta1, theta2, theta3];
    T0_3 = eye(4);
    for i = 1:3
        ct = cosd(theta(i));  st = sind(theta(i));
        ca = cosd(alpha(i));  sa = sind(alpha(i));
        Ti = [ct, -st*ca,  st*sa, r(i)*ct;
              st,  ct*ca, -ct*sa, r(i)*st;
               0,     sa,    ca,    d(i);
               0,      0,     0,       1];
        T0_3 = T0_3 * Ti;
    end
end

%% -------------------------------------------------------------------------
function P_plot = compute_frame_positions(J, alpha, r, d, Px, Py, Pz)
    P_plot = zeros(8, 3);
    P_plot(1,:) = [0, 0, 0];
    T = eye(4);
    for i = 1:6
        ct = cosd(J(i));  st = sind(J(i));
        ca = cosd(alpha(i));  sa = sind(alpha(i));
        Ti = [ct, -st*ca,  st*sa, r(i)*ct;
              st,  ct*ca, -ct*sa, r(i)*st;
               0,     sa,    ca,    d(i);
               0,      0,     0,       1];
        T = T * Ti;
        P_plot(i+1,:) = T(1:3,4)';
    end
    P_plot(8,:) = [Px, Py, Pz];
end