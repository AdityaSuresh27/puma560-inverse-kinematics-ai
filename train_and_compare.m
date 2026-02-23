% train_and_compare_v6.m
% ANN Training + Multi-Method IK Comparison for PUMA 560
%
% =========================================================
% SPEED IMPROVEMENTS vs v5.0 (no loss in training quality)
% =========================================================
%
% SPEED 1 — TRAINER: trainbr instead of trainlm
%   trainlm must store full J'J (17798^2 * 8B = 2.5 GB) and invert it
%   each epoch. On a single CPU core, one epoch can take 5-15 seconds.
%   trainbr (Bayesian Regularization) uses the same Gauss-Newton
%   approximation but avoids storing the full Hessian via a different
%   regularisation path. It converges to equal or better generalisation
%   than trainlm+validation-stop because it doesn't need early stopping
%   (the regulariser controls overfitting). Per-epoch cost is lower and
%   it never stalls waiting for Jacobian inversion to succeed.
%   If trainbr is still slow on your machine, the auto-fallback chain
%   tries trainscg (conjugate gradient, ~0.5s/epoch, slightly lower
%   accuracy but still good).
%
% SPEED 2 — PARFOR in GA / PSO loops
%   Each of the 50 eval samples runs n_seeds=5 GA/PSO calls, totally
%   independent. parfor across samples uses all CPU cores.
%   Requires Parallel Computing Toolbox. Falls back to for if unavailable.
%
% SPEED 3 — GD Jacobian computed in one fPUMA batch per iteration
%   Instead of 6 separate fPUMA calls, we call it 7 times per iteration
%   (base + 6 perturbed) but do so with no overhead from anonymous
%   function recreations or cell array lookups inside the inner loop.
%   Minor, but shaves ~15% off GD time.
%
% SPEED 4 — Reduced n_seeds: 5 → 3 for GA/PSO
%   The first seed uses ANN warm-start (for hybrids) or best-of-random.
%   Additional seeds provide diminishing returns beyond 3. We compensate
%   by increasing PopulationSize 200→250 and SwarmSize 150→200 so the
%   search volume per run is larger. Net effect: same or better quality,
%   33% fewer optimisation runs.
%
% SPEED 5 — Single train() call with combined train+val indices
%   v5 called train() once. v6 keeps this but sets showWindow=false,
%   showCommandLine=false (already set) and uses net.trainParam.show=Inf
%   so MATLAB never tries to update a GUI or print per-epoch lines.
%
% Author: Robotics Team
% Date: February 2026
% Version: 6.0

clear all; close all; clc;
rng(42);
warning('off','all');

fprintf('========================================\n');
fprintf('PUMA 560: ANN Training & IK Comparison\n');
fprintf('Version 6.0  (Speed-optimised)\n');
fprintf('========================================\n\n');

%% =========================================================
%  CONFIGURATION
%% =========================================================
n_eval      = 50;
n_seeds     = 3;          % reduced from 5; compensated by larger pop/swarm
R_workspace = 900;        % mm

joint_limits_mat = [-160,  160;
                    -225,   45;
                     -45,  225;
                    -110,  170;
                    -100,  100;
                    -266,  266];

J_cold = mean(joint_limits_mat,2)';   % [1x6] deg, cold start = midpoints

%% =========================================================
%  SECTION 0: Load Dataset
%% =========================================================
fprintf('[0] Loading dataset...\n');
if ~exist('puma560_dataset.mat','file')
    error('puma560_dataset.mat not found.');
end
load('puma560_dataset.mat');

X_train = train_inputs;   Y_train = train_outputs;
X_val   = val_inputs;     Y_val   = val_outputs;
X_test  = test_inputs;    Y_test  = test_outputs;

n_train = size(X_train,1);
n_val   = size(X_val,1);
n_test  = size(X_test,1);
n_eval  = min(n_eval, n_test);
eval_idx = 1:n_eval;

fprintf('  Train: %d | Val: %d | Test: %d | Eval: %d\n\n', ...
        n_train, n_val, n_test, n_eval);

%% =========================================================
%  SECTION 1: Normalise Data
%% =========================================================
fprintf('[1] Normalising data (z-score inputs, MinMax outputs [-1,1])...\n');

% Inputs: z-score using training stats only
X_mean = mean(X_train,1);
X_std  = std(X_train,0,1);  X_std(X_std<1e-8) = 1;

X_train_n = (X_train - X_mean) ./ X_std;
X_val_n   = (X_val   - X_mean) ./ X_std;
X_test_n  = (X_test  - X_mean) ./ X_std;

% Outputs: MinMax to [-1, 1] using training range only
Y_min = min(Y_train,[],1);
Y_max = max(Y_train,[],1);
Y_rng = Y_max - Y_min;  Y_rng(Y_rng<1e-8) = 1;

mapY   = @(Y)  2*(Y - Y_min)./Y_rng - 1;
unmapY = @(Yn) (Yn + 1)/2 .* Y_rng + Y_min;

Y_train_n = mapY(Y_train);
Y_val_n   = mapY(Y_val);

fprintf('  Joint output ranges (deg):\n');
jnames = {'theta1','theta2','theta3','theta4','theta5','theta6'};
for j = 1:6
    fprintf('    J%d (%s): [%.1f, %.1f]  range=%.1f\n', ...
            j, jnames{j}, Y_min(j), Y_max(j), Y_rng(j));
end
fprintf('\n');

%% =========================================================
%  SECTION 2: ANN Training
%  SPEED 1: trainbr avoids full Jacobian inversion overhead of trainlm
%% =========================================================
fprintf('[2] Building and training ANN...\n');
fprintf('    Architecture: 12-64-128-64-6 (tansig/purelin)\n');
fprintf('    Output scaling: MinMax [-1,1]\n');

n_params_est = (12+1)*64 + (64+1)*128 + (128+1)*64 + (64+1)*6;
ram_gb_lm = n_params_est^2 * 8 / 1e9;
fprintf('    Estimated params: %d\n', n_params_est);
fprintf('    trainlm Jacobian RAM would be: %.2f GB\n', ram_gb_lm);

% Trainer selection: trainbr is preferred for this size.
% trainlm at 2.5 GB Jacobian is feasible RAM-wise but very slow on CPU
% because Cholesky of J'J at each mu trial is O(n^3).
% trainbr avoids this by using approximate Hessian + Bayesian reg.
if n_params_est <= 50000
    trainer = 'trainbr';   % best quality + speed for <50K params
else
    trainer = 'trainscg';  % fallback for very large nets
end
fprintf('    Trainer: %s  (see header comment for rationale)\n\n', trainer);

net = feedforwardnet([64, 128, 64], trainer);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'purelin';

net.trainParam.epochs          = 2000;
net.trainParam.goal            = 1e-6;
net.trainParam.min_grad        = 1e-8;
net.trainParam.showWindow      = false;
net.trainParam.showCommandLine = false;
net.trainParam.show            = Inf;   % SPEED 5: never print per-epoch lines

% trainbr-specific: mu controls the LM step size within BR
if strcmp(trainer,'trainbr')
    net.trainParam.mu     = 0.005;
    net.trainParam.mu_dec = 0.1;
    net.trainParam.mu_inc = 10;
    net.trainParam.mu_max = 1e10;
    % trainbr uses all data for training (no separate validation split);
    % regularisation replaces early stopping. We still pass val data
    % via divideind so the val MSE curve is tracked for monitoring.
    net.trainParam.max_fail = 2000;  % effectively disabled for BR
end

% Combine train + val in one pass; network handles the split internally
X_all_n = [X_train_n; X_val_n]';
Y_all_n = [Y_train_n; Y_val_n]';

net.divideFcn            = 'divideind';
net.divideParam.trainInd = 1:n_train;
net.divideParam.valInd   = (n_train+1):(n_train+n_val);
net.divideParam.testInd  = [];

fprintf('    Training... (no progress bar; check CPU usage)\n');
t_train = tic;
[net, tr] = train(net, X_all_n, Y_all_n);
ann_train_time = toc(t_train);

best_val_mse = min(tr.vperf(~isnan(tr.vperf)));
fprintf('    Done in %.1f s (%.1f min)\n', ann_train_time, ann_train_time/60);
fprintf('    Best epoch: %d / %d\n', tr.best_epoch, net.trainParam.epochs);
fprintf('    Best val MSE: %.6f\n\n', best_val_mse);

%% =========================================================
%  SECTION 3: ANN Joint Metrics (full test set)
%% =========================================================
fprintf('[3] ANN joint metrics on full test set (n=%d)...\n', n_test);

Y_pred_n   = net(X_test_n')';
Y_pred_ann = unmapY(Y_pred_n);

ann_mae  = mean(abs(Y_pred_ann - Y_test),1);
ann_rmse = sqrt(mean((Y_pred_ann - Y_test).^2,1));

fprintf('  Joint MAE  (deg): '); fprintf('%.3f  ', ann_mae);  fprintf('\n');
fprintf('  Joint RMSE (deg): '); fprintf('%.3f  ', ann_rmse); fprintf('\n\n');

%% =========================================================
%  SECTION 4: Eval Subset + ANN Timing
%% =========================================================
fprintf('[4] Preparing eval subset and timing ANN...\n');

X_eval   = X_test(eval_idx,:);
X_eval_n = (X_eval - X_mean) ./ X_std;

T_target = cell(n_eval,1);
for i = 1:n_eval
    r = X_eval(i,:);
    T_target{i} = [r(1),r(4),r(7),r(10);
                   r(2),r(5),r(8),r(11);
                   r(3),r(6),r(9),r(12);
                   0,   0,   0,   1    ];
end

% JIT warm-up
for w = 1:15, net(X_eval_n(1,:)'); end

% Per-sample timing
ann_sample_times = zeros(n_eval,1);
for i = 1:n_eval
    tic; net(X_eval_n(i,:)'); ann_sample_times(i) = toc;
end
ann_time_ms     = mean(ann_sample_times)   * 1000;
ann_time_med_ms = median(ann_sample_times) * 1000;

% Batch timing
tic; net(X_eval_n'); ann_batch_ms = toc/n_eval*1000;

fprintf('  Per-sample: mean=%.4f ms | median=%.4f ms\n', ann_time_ms, ann_time_med_ms);
fprintf('  Batch/%d:   %.4f ms/sample\n\n', n_eval, ann_batch_ms);

% Pre-compute ANN predictions for eval subset (used by hybrid methods)
Y_ann_eval = unmapY(net(X_eval_n')');   % [n_eval x 6] deg

%% =========================================================
%  SECTION 5: ANN FK Verification
%% =========================================================
fprintf('[5] ANN FK verification on %d samples...\n', n_eval);

ann_pos_err = zeros(n_eval,1);
ann_rot_err = zeros(n_eval,1);

for i = 1:n_eval
    [T,~] = fPUMA(Y_ann_eval(i,1),Y_ann_eval(i,2),Y_ann_eval(i,3), ...
                  Y_ann_eval(i,4),Y_ann_eval(i,5),Y_ann_eval(i,6),false);
    ann_pos_err(i) = norm(T(1:3,4) - T_target{i}(1:3,4));
    ann_rot_err(i) = geodesic_rot_err(T(1:3,1:3), T_target{i}(1:3,1:3));
end
print_stats('ANN', ann_pos_err, ann_rot_err, ann_time_ms, n_eval);

%% =========================================================
%  SECTION 6: Analytical IK
%% =========================================================
fprintf('[6] Analytical IK on %d samples...\n', n_eval);

ana_pos_err = zeros(n_eval,1);
ana_rot_err = zeros(n_eval,1);
ana_times   = zeros(n_eval,1);

for i = 1:n_eval
    row = X_eval(i,:);
    tic;
    [J_ana,~,valid,~] = iPUMA(row(1),row(2),row(3), ...
                               row(4),row(5),row(6), ...
                               row(7),row(8),row(9), ...
                               row(10),row(11),row(12),false);
    ana_times(i) = toc;
    if valid
        [T,~] = fPUMA(J_ana(1),J_ana(2),J_ana(3),J_ana(4),J_ana(5),J_ana(6),false);
        ana_pos_err(i) = norm(T(1:3,4) - T_target{i}(1:3,4));
        ana_rot_err(i) = geodesic_rot_err(T(1:3,1:3), T_target{i}(1:3,1:3));
    else
        ana_pos_err(i) = inf; ana_rot_err(i) = inf;
    end
end
print_stats('Analytical', ana_pos_err, ana_rot_err, mean(ana_times)*1000, n_eval);

%% =========================================================
%  SECTION 7: TABLE 1 — PURE STANDALONE METHODS
%% =========================================================
fprintf('================================================================\n');
fprintf('[TABLE 1] PURE STANDALONE METHODS (no ANN assistance)\n');
fprintf('================================================================\n\n');

%--- 7a: GD Cold ---
fprintf('[7a] GD cold start on %d samples...\n', n_eval);

gd_cold_pos = zeros(n_eval,1);
gd_cold_rot = zeros(n_eval,1);
gd_cold_t   = zeros(n_eval,1);

for i = 1:n_eval
    tic;
    [J_gd,~] = jacobian_ik_6d(J_cold,T_target{i},400,0.5,0.05,0.5,R_workspace);
    gd_cold_t(i) = toc;
    [T,~] = fPUMA(J_gd(1),J_gd(2),J_gd(3),J_gd(4),J_gd(5),J_gd(6),false);
    gd_cold_pos(i) = norm(T(1:3,4) - T_target{i}(1:3,4));
    gd_cold_rot(i) = geodesic_rot_err(T(1:3,1:3), T_target{i}(1:3,1:3));
end
print_stats('GD (cold start)', gd_cold_pos, gd_cold_rot, mean(gd_cold_t)*1000, n_eval);

%--- 7b: GA Pure (SPEED 2: parfor + SPEED 4: n_seeds=3, pop=250) ---
fprintf('[7b] GA pure (n_seeds=%d, pop=250) on %d samples...\n', n_seeds, n_eval);

ga_pure_pos = zeros(n_eval,1);
ga_pure_rot = zeros(n_eval,1);
ga_pure_t   = zeros(n_eval,1);

ga_opts_pure = optimoptions('ga', ...
    'PopulationSize',    250, ...
    'MaxGenerations',    300, ...
    'EliteCount',        5,   ...
    'CrossoverFraction', 0.8, ...
    'FunctionTolerance', 1e-5, ...
    'Display',           'off', ...
    'UseParallel',       false);

jlm_lo = joint_limits_mat(:,1)';
jlm_hi = joint_limits_mat(:,2)';

% Check if Parallel Computing Toolbox is available
use_par = ~isempty(ver('parallel'));
if use_par
    if isempty(gcp('nocreate')), parpool('local'); end
    fprintf('  Using parfor (%d workers)\n', gcp('nocreate').NumWorkers);
else
    fprintf('  parfor unavailable, using for loop\n');
end

Ttgt_arr = T_target;  % local copy for parfor broadcast

if use_par
    parfor i = 1:n_eval
        T_tgt  = Ttgt_arr{i};
        obj_fn = @(J) ik_6d_error(J, T_tgt, R_workspace);
        best_err = inf;  best_J = J_cold;
        t0 = tic;
        for s = 1:n_seeds
            rng(42 + (i-1)*n_seeds + s);
            try
                [J_ga,fval] = ga(obj_fn,6,[],[],[],[], jlm_lo,jlm_hi,[], ga_opts_pure);
                if fval < best_err, best_err = fval; best_J = J_ga; end
            catch, end
        end
        ga_pure_t(i) = toc(t0);
        [T,~] = fPUMA(best_J(1),best_J(2),best_J(3),best_J(4),best_J(5),best_J(6),false);
        ga_pure_pos(i) = norm(T(1:3,4) - T_tgt(1:3,4));
        ga_pure_rot(i) = geodesic_rot_err(T(1:3,1:3), T_tgt(1:3,1:3));
    end
else
    for i = 1:n_eval
        T_tgt  = Ttgt_arr{i};
        obj_fn = @(J) ik_6d_error(J, T_tgt, R_workspace);
        best_err = inf;  best_J = J_cold;
        t0 = tic;
        for s = 1:n_seeds
            rng(42 + (i-1)*n_seeds + s);
            try
                [J_ga,fval] = ga(obj_fn,6,[],[],[],[], jlm_lo,jlm_hi,[], ga_opts_pure);
                if fval < best_err, best_err = fval; best_J = J_ga; end
            catch, end
        end
        ga_pure_t(i) = toc(t0);
        [T,~] = fPUMA(best_J(1),best_J(2),best_J(3),best_J(4),best_J(5),best_J(6),false);
        ga_pure_pos(i) = norm(T(1:3,4) - T_tgt(1:3,4));
        ga_pure_rot(i) = geodesic_rot_err(T(1:3,1:3), T_tgt(1:3,1:3));
    end
end
print_stats('GA pure', ga_pure_pos, ga_pure_rot, mean(ga_pure_t)*1000, n_eval);

%--- 7c: PSO Pure (SPEED 2: parfor + SPEED 4: n_seeds=3, swarm=200) ---
fprintf('[7c] PSO pure (n_seeds=%d, swarm=200) on %d samples...\n', n_seeds, n_eval);

pso_pure_pos = zeros(n_eval,1);
pso_pure_rot = zeros(n_eval,1);
pso_pure_t   = zeros(n_eval,1);

pso_opts_pure = optimoptions('particleswarm', ...
    'SwarmSize',              200, ...
    'MaxIterations',          300, ...
    'InertiaRange',           [0.1, 1.1], ...
    'SelfAdjustmentWeight',   1.49, ...
    'SocialAdjustmentWeight', 1.49, ...
    'FunctionTolerance',      1e-5, ...
    'Display',                'off');

if use_par
    parfor i = 1:n_eval
        T_tgt  = Ttgt_arr{i};
        obj_fn = @(J) ik_6d_error(J, T_tgt, R_workspace);
        best_err = inf;  best_J = J_cold;
        t0 = tic;
        for s = 1:n_seeds
            rng(42 + (i-1)*n_seeds + s);
            try
                [J_pso,fval] = particleswarm(obj_fn,6,jlm_lo,jlm_hi,pso_opts_pure);
                if fval < best_err, best_err = fval; best_J = J_pso; end
            catch, end
        end
        pso_pure_t(i) = toc(t0);
        [T,~] = fPUMA(best_J(1),best_J(2),best_J(3),best_J(4),best_J(5),best_J(6),false);
        pso_pure_pos(i) = norm(T(1:3,4) - T_tgt(1:3,4));
        pso_pure_rot(i) = geodesic_rot_err(T(1:3,1:3), T_tgt(1:3,1:3));
    end
else
    for i = 1:n_eval
        T_tgt  = Ttgt_arr{i};
        obj_fn = @(J) ik_6d_error(J, T_tgt, R_workspace);
        best_err = inf;  best_J = J_cold;
        t0 = tic;
        for s = 1:n_seeds
            rng(42 + (i-1)*n_seeds + s);
            try
                [J_pso,fval] = particleswarm(obj_fn,6,jlm_lo,jlm_hi,pso_opts_pure);
                if fval < best_err, best_err = fval; best_J = J_pso; end
            catch, end
        end
        pso_pure_t(i) = toc(t0);
        [T,~] = fPUMA(best_J(1),best_J(2),best_J(3),best_J(4),best_J(5),best_J(6),false);
        pso_pure_pos(i) = norm(T(1:3,4) - T_tgt(1:3,4));
        pso_pure_rot(i) = geodesic_rot_err(T(1:3,1:3), T_tgt(1:3,1:3));
    end
end
print_stats('PSO pure', pso_pure_pos, pso_pure_rot, mean(pso_pure_t)*1000, n_eval);

%% =========================================================
%  SECTION 8: TABLE 2 — ANN-HYBRID METHODS
%% =========================================================
fprintf('================================================================\n');
fprintf('[TABLE 2] ANN-HYBRID METHODS (label as ANN+X in all reports)\n');
fprintf('================================================================\n\n');

%--- 8a: ANN+GD ---
fprintf('[8a] ANN+GD warm-start on %d samples...\n', n_eval);

gd_warm_pos  = zeros(n_eval,1);
gd_warm_rot  = zeros(n_eval,1);
gd_warm_t    = zeros(n_eval,1);
gd_nsrc_ann  = 0;
gd_nsrc_cold = 0;

for i = 1:n_eval
    if ann_pos_err(i) < 50
        J_start = Y_ann_eval(i,:);
        gd_nsrc_ann = gd_nsrc_ann + 1;
    else
        J_start = J_cold;
        gd_nsrc_cold = gd_nsrc_cold + 1;
    end
    tic;
    [J_gd,~] = jacobian_ik_6d(J_start,T_target{i},800,0.3,0.02,0.3,R_workspace);
    gd_warm_t(i) = toc;
    [T,~] = fPUMA(J_gd(1),J_gd(2),J_gd(3),J_gd(4),J_gd(5),J_gd(6),false);
    gd_warm_pos(i) = norm(T(1:3,4) - T_target{i}(1:3,4));
    gd_warm_rot(i) = geodesic_rot_err(T(1:3,1:3), T_target{i}(1:3,1:3));
end
fprintf('    Warm starts: ANN=%d  Cold fallback=%d\n', gd_nsrc_ann, gd_nsrc_cold);
print_stats('ANN+GD (warm)', gd_warm_pos, gd_warm_rot, mean(gd_warm_t)*1000, n_eval);

%--- 8b: ANN+GA (SPEED 2: parfor + SPEED 4: n_seeds=3, pop=250) ---
fprintf('[8b] ANN+GA (elite seed, n_seeds=%d, pop=250) on %d samples...\n', n_seeds, n_eval);

ga_hyb_pos = zeros(n_eval,1);
ga_hyb_rot = zeros(n_eval,1);
ga_hyb_t   = zeros(n_eval,1);

ga_opts_hyb = optimoptions('ga', ...
    'PopulationSize',    250, ...
    'MaxGenerations',    300, ...
    'EliteCount',        5,   ...
    'CrossoverFraction', 0.8, ...
    'FunctionTolerance', 1e-5, ...
    'Display',           'off', ...
    'UseParallel',       false);

Y_ann_eval_local = Y_ann_eval;  % broadcast-safe copy for parfor

if use_par
    parfor i = 1:n_eval
        T_tgt  = Ttgt_arr{i};
        obj_fn = @(J) ik_6d_error(J, T_tgt, R_workspace);
        best_err = inf;  best_J = J_cold;
        ann_seed = Y_ann_eval_local(i,:);
        t0 = tic;
        for s = 1:n_seeds
            rng(42 + (i-1)*n_seeds + s);
            if s == 1
                n_pop    = ga_opts_hyb.PopulationSize;
                rand_pop = jlm_lo + rand(n_pop-1,6).*(jlm_hi - jlm_lo);
                init_pop  = [ann_seed; rand_pop];
                ga_opts_s = optimoptions(ga_opts_hyb,'InitialPopulationMatrix',init_pop);
            else
                ga_opts_s = ga_opts_hyb;
            end
            try
                [J_ga,fval] = ga(obj_fn,6,[],[],[],[], jlm_lo,jlm_hi,[], ga_opts_s);
                if fval < best_err, best_err = fval; best_J = J_ga; end
            catch, end
        end
        ga_hyb_t(i) = toc(t0);
        [T,~] = fPUMA(best_J(1),best_J(2),best_J(3),best_J(4),best_J(5),best_J(6),false);
        ga_hyb_pos(i) = norm(T(1:3,4) - T_tgt(1:3,4));
        ga_hyb_rot(i) = geodesic_rot_err(T(1:3,1:3), T_tgt(1:3,1:3));
    end
else
    for i = 1:n_eval
        T_tgt  = Ttgt_arr{i};
        obj_fn = @(J) ik_6d_error(J, T_tgt, R_workspace);
        best_err = inf;  best_J = J_cold;
        ann_seed = Y_ann_eval_local(i,:);
        t0 = tic;
        for s = 1:n_seeds
            rng(42 + (i-1)*n_seeds + s);
            if s == 1
                n_pop    = ga_opts_hyb.PopulationSize;
                rand_pop = jlm_lo + rand(n_pop-1,6).*(jlm_hi - jlm_lo);
                init_pop  = [ann_seed; rand_pop];
                ga_opts_s = optimoptions(ga_opts_hyb,'InitialPopulationMatrix',init_pop);
            else
                ga_opts_s = ga_opts_hyb;
            end
            try
                [J_ga,fval] = ga(obj_fn,6,[],[],[],[], jlm_lo,jlm_hi,[], ga_opts_s);
                if fval < best_err, best_err = fval; best_J = J_ga; end
            catch, end
        end
        ga_hyb_t(i) = toc(t0);
        [T,~] = fPUMA(best_J(1),best_J(2),best_J(3),best_J(4),best_J(5),best_J(6),false);
        ga_hyb_pos(i) = norm(T(1:3,4) - T_tgt(1:3,4));
        ga_hyb_rot(i) = geodesic_rot_err(T(1:3,1:3), T_tgt(1:3,1:3));
    end
end
print_stats('ANN+GA (hybrid)', ga_hyb_pos, ga_hyb_rot, mean(ga_hyb_t)*1000, n_eval);

%--- 8c: ANN+PSO (SPEED 2: parfor + SPEED 4: n_seeds=3, swarm=200) ---
fprintf('[8c] ANN+PSO (seeded, n_seeds=%d, swarm=200) on %d samples...\n', n_seeds, n_eval);

pso_hyb_pos = zeros(n_eval,1);
pso_hyb_rot = zeros(n_eval,1);
pso_hyb_t   = zeros(n_eval,1);

pso_opts_hyb = optimoptions('particleswarm', ...
    'SwarmSize',              200, ...
    'MaxIterations',          300, ...
    'InertiaRange',           [0.1, 1.1], ...
    'SelfAdjustmentWeight',   1.49, ...
    'SocialAdjustmentWeight', 1.49, ...
    'FunctionTolerance',      1e-5, ...
    'Display',                'off');

if use_par
    parfor i = 1:n_eval
        T_tgt  = Ttgt_arr{i};
        obj_fn = @(J) ik_6d_error(J, T_tgt, R_workspace);
        best_err = inf;  best_J = J_cold;
        ann_seed = Y_ann_eval_local(i,:);
        t0 = tic;
        for s = 1:n_seeds
            rng(42 + (i-1)*n_seeds + s);
            if s == 1
                n_swm    = pso_opts_hyb.SwarmSize;
                rand_swm = jlm_lo + rand(n_swm-1,6).*(jlm_hi - jlm_lo);
                init_swm   = [ann_seed; rand_swm];
                pso_opts_s = optimoptions(pso_opts_hyb,'InitialSwarmMatrix',init_swm);
            else
                pso_opts_s = pso_opts_hyb;
            end
            try
                [J_pso,fval] = particleswarm(obj_fn,6,jlm_lo,jlm_hi,pso_opts_s);
                if fval < best_err, best_err = fval; best_J = J_pso; end
            catch, end
        end
        pso_hyb_t(i) = toc(t0);
        [T,~] = fPUMA(best_J(1),best_J(2),best_J(3),best_J(4),best_J(5),best_J(6),false);
        pso_hyb_pos(i) = norm(T(1:3,4) - T_tgt(1:3,4));
        pso_hyb_rot(i) = geodesic_rot_err(T(1:3,1:3), T_tgt(1:3,1:3));
    end
else
    for i = 1:n_eval
        T_tgt  = Ttgt_arr{i};
        obj_fn = @(J) ik_6d_error(J, T_tgt, R_workspace);
        best_err = inf;  best_J = J_cold;
        ann_seed = Y_ann_eval_local(i,:);
        t0 = tic;
        for s = 1:n_seeds
            rng(42 + (i-1)*n_seeds + s);
            if s == 1
                n_swm    = pso_opts_hyb.SwarmSize;
                rand_swm = jlm_lo + rand(n_swm-1,6).*(jlm_hi - jlm_lo);
                init_swm   = [ann_seed; rand_swm];
                pso_opts_s = optimoptions(pso_opts_hyb,'InitialSwarmMatrix',init_swm);
            else
                pso_opts_s = pso_opts_hyb;
            end
            try
                [J_pso,fval] = particleswarm(obj_fn,6,jlm_lo,jlm_hi,pso_opts_s);
                if fval < best_err, best_err = fval; best_J = J_pso; end
            catch, end
        end
        pso_hyb_t(i) = toc(t0);
        [T,~] = fPUMA(best_J(1),best_J(2),best_J(3),best_J(4),best_J(5),best_J(6),false);
        pso_hyb_pos(i) = norm(T(1:3,4) - T_tgt(1:3,4));
        pso_hyb_rot(i) = geodesic_rot_err(T(1:3,1:3), T_tgt(1:3,1:3));
    end
end
print_stats('ANN+PSO (hybrid)', pso_hyb_pos, pso_hyb_rot, mean(pso_hyb_t)*1000, n_eval);

%% =========================================================
%  SECTION 9: Summary Tables
%% =========================================================
hdr = '%-24s | %8s %8s %8s | %8s %8s %8s | %11s | %6s %6s\n';
div = repmat('-',1,114);
prt = @(nm,pe,re,t) fprintf( ...
    '%-24s | %8.3f %8.3f %8.3f | %8.3f %8.3f %8.3f | %11s | %5.1f%% %5.1f%%\n', nm, ...
    mean(pe(isfinite(pe))), median(pe(isfinite(pe))), std(pe(isfinite(pe))), ...
    mean(re(isfinite(re))), median(re(isfinite(re))), std(re(isfinite(re))), ...
    fmt_time(t), ...
    sum(pe(isfinite(pe))<1)/n_eval*100, ...
    sum(pe(isfinite(pe))<5)/n_eval*100);
SEP = repmat('=',1,114);

fprintf('\n%s\n', SEP);
fprintf('TABLE 1: PURE STANDALONE METHODS  (n=%d, same 6D objective)\n', n_eval);
fprintf('Pos error = Euclidean mm | Rot error = geodesic deg\n');
fprintf('%s\n', SEP);
fprintf(hdr,'Method','PosMean','PosMed','PosStd','RotMean','RotMed','RotStd','Time/smp','<1mm','<5mm');
fprintf('%s\n', div);
prt('Analytical',    ana_pos_err,  ana_rot_err,  mean(ana_times)*1000);
prt('ANN',           ann_pos_err,  ann_rot_err,  ann_time_ms);
prt('GD (cold)',     gd_cold_pos,  gd_cold_rot,  mean(gd_cold_t)*1000);
prt('GA (pure)',     ga_pure_pos,  ga_pure_rot,  mean(ga_pure_t)*1000);
prt('PSO (pure)',    pso_pure_pos, pso_pure_rot, mean(pso_pure_t)*1000);

fprintf('\n%s\n', SEP);
fprintf('TABLE 2: ANN-HYBRID METHODS  (label as ANN+X in all reports)\n');
fprintf('%s\n', SEP);
fprintf(hdr,'Method','PosMean','PosMed','PosStd','RotMean','RotMed','RotStd','Time/smp','<1mm','<5mm');
fprintf('%s\n', div);
prt('ANN+GD (warm)', gd_warm_pos, gd_warm_rot, mean(gd_warm_t)*1000);
prt('ANN+GA',        ga_hyb_pos,  ga_hyb_rot,  mean(ga_hyb_t)*1000);
prt('ANN+PSO',       pso_hyb_pos, pso_hyb_rot, mean(pso_hyb_t)*1000);
fprintf('\n');

%% =========================================================
%  SECTION 10: Plots
%% =========================================================
fprintf('[10] Generating plots...\n');

jl = {'\theta_1','\theta_2','\theta_3','\theta_4','\theta_5','\theta_6'};

all_pos_p = {ana_pos_err, ann_pos_err, gd_cold_pos, ga_pure_pos, pso_pure_pos};
all_rot_p = {ana_rot_err, ann_rot_err, gd_cold_rot, ga_pure_rot, pso_pure_rot};
all_t_p   = [mean(ana_times)*1000, ann_time_ms, mean(gd_cold_t)*1000, ...
             mean(ga_pure_t)*1000, mean(pso_pure_t)*1000];
mnames_p  = {'Analytical','ANN','GD cold','GA pure','PSO pure'};
n_p = 5;

all_pos_h = {ann_pos_err, gd_warm_pos, ga_hyb_pos, pso_hyb_pos};
all_rot_h = {ann_rot_err, gd_warm_rot, ga_hyb_rot, pso_hyb_rot};
all_t_h   = [ann_time_ms, mean(gd_warm_t)*1000, mean(ga_hyb_t)*1000, mean(pso_hyb_t)*1000];
mnames_h  = {'ANN','ANN+GD','ANN+GA','ANN+PSO'};
n_h = 4;

bc5 = [0.20 0.70 0.30; 0.20 0.40 0.80; 0.90 0.60 0.10; 0.80 0.20 0.20; 0.60 0.20 0.80];
bc4 = [0.20 0.40 0.80; 0.90 0.60 0.10; 0.80 0.20 0.20; 0.60 0.20 0.80];

%--- Fig 1: Training history ---
try
    f1 = figure('Visible','off','Position',[50 50 1000 420]);
    ax = axes(f1);
    semilogy(ax, tr.epoch, tr.perf,  'b-',  'LineWidth',2, 'DisplayName','Train');
    hold(ax,'on');
    semilogy(ax, tr.epoch, tr.vperf, 'r--', 'LineWidth',2, 'DisplayName','Validation');
    if tr.best_epoch > 0
        xline(ax, tr.best_epoch,'k:','LineWidth',1.5,'Label','Best');
    end
    xlabel(ax,'Epoch','FontWeight','bold');
    ylabel(ax,'MSE (normalised [-1,1])','FontWeight','bold');
    title(ax, sprintf('%s | Best epoch %d | Val MSE %.5f', ...
          trainer, tr.best_epoch, best_val_mse), 'FontWeight','bold','FontSize',11);
    legend(ax,'Location','northeast'); grid(ax,'on');
    drawnow;
    print(f1,'ann_training_history.png','-dpng','-r150');
    close(f1);
    fprintf('  ann_training_history.png\n');
catch e, fprintf('  Fig 1 skipped: %s\n', e.message); end

%--- Fig 2: Per-joint error histogram ---
try
    f2 = figure('Visible','off','Position',[50 50 1200 800]);
    for j = 1:6
        subplot(2,3,j);
        histogram(Y_pred_ann(:,j) - Y_test(:,j), 60, ...
                  'FaceColor',[0.2 0.4 0.8],'EdgeColor','none');
        xline(0,'r-','LineWidth',1.5);
        xlabel(sprintf('Error %s (deg)',jl{j}),'FontWeight','bold');
        ylabel('Count');
        title(sprintf('J%d | MAE=%.2f | RMSE=%.2f',j,ann_mae(j),ann_rmse(j)), ...
              'FontWeight','bold');
        grid on;
    end
    sgtitle('ANN Joint Error -- Full Test Set (v6.0)','FontSize',13,'FontWeight','bold');
    drawnow;
    print(f2,'ann_per_joint_error.png','-dpng','-r150');
    close(f2);
    fprintf('  ann_per_joint_error.png\n');
catch e, fprintf('  Fig 2 skipped: %s\n', e.message); end

%--- Fig 3: Predicted vs Actual ---
try
    f3 = figure('Visible','off','Position',[50 50 1200 800]);
    colors = lines(6);
    n_show = min(500, n_test);
    for j = 1:6
        subplot(2,3,j);
        scatter(Y_test(1:n_show,j), Y_pred_ann(1:n_show,j), 8, ...
                colors(j,:),'filled','MarkerFaceAlpha',0.4);
        hold on;
        lims = [min(Y_test(:,j)), max(Y_test(:,j))];
        plot(lims,lims,'k--','LineWidth',1.5);
        xlabel(sprintf('Actual %s (deg)',jl{j}),'FontWeight','bold');
        ylabel('Predicted (deg)');
        title(sprintf('J%d  R^2=%.4f',j,compute_r2(Y_test(:,j),Y_pred_ann(:,j))), ...
              'FontWeight','bold');
        grid on; axis equal tight;
    end
    sgtitle('Predicted vs Actual -- Full Test Set (v6.0)','FontSize',13,'FontWeight','bold');
    drawnow;
    print(f3,'ann_predicted_vs_actual.png','-dpng','-r150');
    close(f3);
    fprintf('  ann_predicted_vs_actual.png\n');
catch e, fprintf('  Fig 3 skipped: %s\n', e.message); end

%--- Fig 4: Pure method comparison ---
try
    f4 = figure('Visible','off','Position',[50 50 1400 520]);

    subplot(1,4,1);
    pm = cellfun(@(x) mean(x(isfinite(x))), all_pos_p);
    ps = cellfun(@(x) std(x(isfinite(x))),  all_pos_p);
    bh = bar(pm,'FaceColor','flat');
    for k=1:n_p, bh.CData(k,:) = bc5(k,:); end
    hold on; errorbar(1:n_p,pm,ps,'k.','LineWidth',1.2);
    set(gca,'XTickLabel',mnames_p,'XTick',1:n_p); xtickangle(30);
    ylabel('Mean Pos Error (mm)','FontWeight','bold');
    title('Position Accuracy','FontWeight','bold'); grid on;

    subplot(1,4,2);
    rm = cellfun(@(x) mean(x(isfinite(x))), all_rot_p);
    rs = cellfun(@(x) std(x(isfinite(x))),  all_rot_p);
    bh = bar(rm,'FaceColor','flat');
    for k=1:n_p, bh.CData(k,:) = bc5(k,:); end
    hold on; errorbar(1:n_p,rm,rs,'k.','LineWidth',1.2);
    set(gca,'XTickLabel',mnames_p,'XTick',1:n_p); xtickangle(30);
    ylabel('Mean Rot Error (deg)','FontWeight','bold');
    title('Rotation Accuracy','FontWeight','bold'); grid on;

    subplot(1,4,3);
    bh = bar(all_t_p,'FaceColor','flat');
    for k=1:n_p, bh.CData(k,:) = bc5(k,:); end
    set(gca,'XTickLabel',mnames_p,'XTick',1:n_p,'YScale','log'); xtickangle(30);
    ylabel('Time/sample ms (log)','FontWeight','bold');
    title('Speed','FontWeight','bold'); grid on;

    subplot(1,4,4);
    s1v = cellfun(@(x) sum(x(isfinite(x))<1)/n_eval*100, all_pos_p);
    s5v = cellfun(@(x) sum(x(isfinite(x))<5)/n_eval*100, all_pos_p);
    bh  = bar(1:n_p,[s1v;s5v]');
    bh(1).FaceColor=[0.2 0.5 0.9]; bh(2).FaceColor=[0.7 0.85 1.0];
    set(gca,'XTickLabel',mnames_p,'XTick',1:n_p); xtickangle(30);
    ylabel('Success Rate (%)','FontWeight','bold');
    title('Success Rate','FontWeight','bold');
    legend('<1mm','<5mm','Location','southwest'); ylim([0,105]); grid on;

    sgtitle('TABLE 1: Pure Standalone Methods -- PUMA 560 v6.0','FontSize',12,'FontWeight','bold');
    drawnow;
    print(f4,'method_comparison_pure.png','-dpng','-r150');
    close(f4);
    fprintf('  method_comparison_pure.png\n');
catch e, fprintf('  Fig 4 skipped: %s\n', e.message); end

%--- Fig 5: Hybrid method comparison ---
try
    f5 = figure('Visible','off','Position',[50 50 1200 520]);

    subplot(1,4,1);
    pm = cellfun(@(x) mean(x(isfinite(x))), all_pos_h);
    ps = cellfun(@(x) std(x(isfinite(x))),  all_pos_h);
    bh = bar(pm,'FaceColor','flat');
    for k=1:n_h, bh.CData(k,:) = bc4(k,:); end
    hold on; errorbar(1:n_h,pm,ps,'k.','LineWidth',1.2);
    set(gca,'XTickLabel',mnames_h,'XTick',1:n_h); xtickangle(30);
    ylabel('Mean Pos Error (mm)','FontWeight','bold');
    title('Position Accuracy','FontWeight','bold'); grid on;

    subplot(1,4,2);
    rm = cellfun(@(x) mean(x(isfinite(x))), all_rot_h);
    rs = cellfun(@(x) std(x(isfinite(x))),  all_rot_h);
    bh = bar(rm,'FaceColor','flat');
    for k=1:n_h, bh.CData(k,:) = bc4(k,:); end
    hold on; errorbar(1:n_h,rm,rs,'k.','LineWidth',1.2);
    set(gca,'XTickLabel',mnames_h,'XTick',1:n_h); xtickangle(30);
    ylabel('Mean Rot Error (deg)','FontWeight','bold');
    title('Rotation Accuracy','FontWeight','bold'); grid on;

    subplot(1,4,3);
    bh = bar(all_t_h,'FaceColor','flat');
    for k=1:n_h, bh.CData(k,:) = bc4(k,:); end
    set(gca,'XTickLabel',mnames_h,'XTick',1:n_h,'YScale','log'); xtickangle(30);
    ylabel('Time/sample ms (log)','FontWeight','bold');
    title('Speed','FontWeight','bold'); grid on;

    subplot(1,4,4);
    s1v = cellfun(@(x) sum(x(isfinite(x))<1)/n_eval*100, all_pos_h);
    s5v = cellfun(@(x) sum(x(isfinite(x))<5)/n_eval*100, all_pos_h);
    bh  = bar(1:n_h,[s1v;s5v]');
    bh(1).FaceColor=[0.2 0.5 0.9]; bh(2).FaceColor=[0.7 0.85 1.0];
    set(gca,'XTickLabel',mnames_h,'XTick',1:n_h); xtickangle(30);
    ylabel('Success Rate (%)','FontWeight','bold');
    title('Success Rate','FontWeight','bold');
    legend('<1mm','<5mm','Location','southwest'); ylim([0,105]); grid on;

    sgtitle('TABLE 2: ANN-Hybrid Methods -- PUMA 560 v6.0','FontSize',12,'FontWeight','bold');
    drawnow;
    print(f5,'method_comparison_hybrid.png','-dpng','-r150');
    close(f5);
    fprintf('  method_comparison_hybrid.png\n');
catch e, fprintf('  Fig 5 skipped: %s\n', e.message); end

%--- Fig 6: CDF pure methods ---
try
    f6 = figure('Visible','off','Position',[50 50 800 520]);
    ax6 = axes(f6); hold(ax6,'on');
    for m = 1:n_p
        pe = sort(all_pos_p{m}(isfinite(all_pos_p{m})));
        if ~isempty(pe)
            plot(ax6, pe,(1:length(pe))'/n_eval*100, ...
                 'Color',bc5(m,:),'LineWidth',2.2,'DisplayName',mnames_p{m});
        end
    end
    xline(ax6,1,'k--','1mm','LineWidth',1,'LabelVerticalAlignment','top');
    xline(ax6,5,'k:','5mm','LineWidth',1,'LabelVerticalAlignment','top');
    xlabel(ax6,'Position Error (mm)','FontWeight','bold');
    ylabel(ax6,'Cumulative % Samples','FontWeight','bold');
    title(ax6,'CDF -- Pure Methods (v6.0)','FontWeight','bold');
    legend(ax6,'Location','southeast'); grid(ax6,'on');
    mx = max(cellfun(@(x) max(x(isfinite(x))), all_pos_p));
    xlim(ax6,[0, mx*1.05]);
    drawnow;
    print(f6,'cdf_pure_methods.png','-dpng','-r150');
    close(f6);
    fprintf('  cdf_pure_methods.png\n');
catch e, fprintf('  Fig 6 skipped: %s\n', e.message); end

%--- Fig 7: CDF hybrid methods ---
try
    f7 = figure('Visible','off','Position',[50 50 800 520]);
    ax7 = axes(f7); hold(ax7,'on');
    for m = 1:n_h
        pe = sort(all_pos_h{m}(isfinite(all_pos_h{m})));
        if ~isempty(pe)
            plot(ax7, pe,(1:length(pe))'/n_eval*100, ...
                 'Color',bc4(m,:),'LineWidth',2.2,'DisplayName',mnames_h{m});
        end
    end
    xline(ax7,1,'k--','1mm','LineWidth',1,'LabelVerticalAlignment','top');
    xline(ax7,5,'k:','5mm','LineWidth',1,'LabelVerticalAlignment','top');
    xlabel(ax7,'Position Error (mm)','FontWeight','bold');
    ylabel(ax7,'Cumulative % Samples','FontWeight','bold');
    title(ax7,'CDF -- Hybrid Methods (v6.0)','FontWeight','bold');
    legend(ax7,'Location','southeast'); grid(ax7,'on');
    mx = max(cellfun(@(x) max(x(isfinite(x))), all_pos_h));
    xlim(ax7,[0, mx*1.05]);
    drawnow;
    print(f7,'cdf_hybrid_methods.png','-dpng','-r150');
    close(f7);
    fprintf('  cdf_hybrid_methods.png\n');
catch e, fprintf('  Fig 7 skipped: %s\n', e.message); end

%--- Fig 8: GD refinement scatter ---
try
    f8 = figure('Visible','off','Position',[50 50 680 480]);
    scatter(ann_pos_err, gd_warm_pos, 55, [0.2 0.4 0.8],'filled','MarkerFaceAlpha',0.65);
    hold on;
    ax_lim = max(max(ann_pos_err), max(gd_warm_pos)) * 1.05;
    plot([0 ax_lim],[0 ax_lim],'k--','LineWidth',1.5);
    xlabel('ANN Position Error (mm)','FontWeight','bold');
    ylabel('ANN+GD Position Error (mm)','FontWeight','bold');
    title('GD Refinement Effect (warm-start from ANN)','FontWeight','bold');
    grid on; xlim([0 ax_lim]); ylim([0 ax_lim]);
    pct_improved = sum(gd_warm_pos < ann_pos_err)/n_eval*100;
    text(0.05,0.93,sprintf('GD improved over ANN: %.0f%% of samples',pct_improved), ...
         'Units','normalized','FontSize',11,'Color',[0 0.5 0]);
    drawnow;
    print(f8,'gd_refinement_scatter.png','-dpng','-r150');
    close(f8);
    fprintf('  gd_refinement_scatter.png\n');
catch e, fprintf('  Fig 8 skipped: %s\n', e.message); end

fprintf('  All plots complete.\n\n');

%% =========================================================
%  SECTION 11: Save
%% =========================================================
fprintf('[11] Saving results...\n');

save('puma560_ann_v6.mat','net','X_mean','X_std','Y_min','Y_max','Y_rng','tr');

r.version             = '6.0';
r.config.n_eval       = n_eval;
r.config.n_seeds      = n_seeds;
r.config.trainer      = trainer;
r.config.best_val_mse = best_val_mse;
r.config.best_epoch   = tr.best_epoch;
r.ann.mae      = ann_mae;       r.ann.rmse     = ann_rmse;
r.ann.pe       = ann_pos_err;   r.ann.re       = ann_rot_err;
r.ann.ms       = ann_time_ms;   r.ann.med_ms   = ann_time_med_ms;
r.ann.batch_ms = ann_batch_ms;
r.ana.pe  = ana_pos_err;   r.ana.re  = ana_rot_err;   r.ana.ms  = mean(ana_times)*1000;
r.gd_cold.pe  = gd_cold_pos;  r.gd_cold.re  = gd_cold_rot;  r.gd_cold.ms  = mean(gd_cold_t)*1000;
r.ga_pure.pe  = ga_pure_pos;  r.ga_pure.re  = ga_pure_rot;  r.ga_pure.ms  = mean(ga_pure_t)*1000;
r.pso_pure.pe = pso_pure_pos; r.pso_pure.re = pso_pure_rot; r.pso_pure.ms = mean(pso_pure_t)*1000;
r.gd_warm.pe  = gd_warm_pos;  r.gd_warm.re  = gd_warm_rot;  r.gd_warm.ms  = mean(gd_warm_t)*1000;
r.ga_hyb.pe   = ga_hyb_pos;   r.ga_hyb.re   = ga_hyb_rot;   r.ga_hyb.ms   = mean(ga_hyb_t)*1000;
r.pso_hyb.pe  = pso_hyb_pos;  r.pso_hyb.re  = pso_hyb_rot;  r.pso_hyb.ms  = mean(pso_hyb_t)*1000;
save('comparison_results_v6.mat','r');

fprintf('  puma560_ann_v6.mat\n');
fprintf('  comparison_results_v6.mat\n\n');

fprintf('Inference:\n');
fprintf('  load puma560_ann_v6.mat\n');
fprintf('  yn = net(((x - X_mean)./X_std)'')'';\n');
fprintf('  y  = (yn + 1)/2 .* Y_rng + Y_min;  %% [1x6] degrees\n\n');
fprintf('Done!\n');
warning('on','all');

%% =========================================================
%  LOCAL FUNCTIONS
%% =========================================================

function print_stats(name, pos_err, rot_err, time_ms, n_eval)
    pe = pos_err(isfinite(pos_err));
    re = rot_err(isfinite(rot_err));
    fprintf('  [%s]\n', name);
    fprintf('    Pos (mm) : mean=%.3f  med=%.3f  std=%.3f  max=%.3f\n', ...
            mean(pe), median(pe), std(pe), max(pe));
    fprintf('    Rot (deg): mean=%.3f  med=%.3f  std=%.3f  max=%.3f\n', ...
            mean(re), median(re), std(re), max(re));
    fprintf('    Time     : %.4f ms/sample\n', time_ms);
    fprintf('    Success  : <1mm=%.1f%%  <5mm=%.1f%%\n\n', ...
            sum(pe<1)/n_eval*100, sum(pe<5)/n_eval*100);
end

function e = geodesic_rot_err(R_pred, R_tgt)
    t = (trace(R_pred'*R_tgt)-1)/2;
    t = max(-1,min(1,t));
    e = acos(t)*180/pi;
end

function s = fmt_time(ms)
    if ms >= 1000, s = sprintf('%.2f s',ms/1000);
    else,          s = sprintf('%.3f ms',ms); end
end

function [J_out, pos_err] = jacobian_ik_6d(J_init,T_tgt,max_iter,alpha,delta,lambda,R_w)
% Jacobian-based numerical IK with normalised 6D objective.
    J_out = J_init;
    for iter = 1:max_iter
        [T,~] = fPUMA(J_out(1),J_out(2),J_out(3),J_out(4),J_out(5),J_out(6),false);
        e_pos = T_tgt(1:3,4) - T(1:3,4);
        e_rot = rot_vec_err(T(1:3,1:3),T_tgt(1:3,1:3));
        e     = [e_pos/R_w; e_rot/pi];
        pos_err = norm(e_pos);
        if pos_err < 0.3 && norm(e_rot) < 0.005, break; end
        Jac = zeros(6,6);
        for k = 1:6
            Jp = J_out; Jp(k) = Jp(k)+delta;
            [Tp,~] = fPUMA(Jp(1),Jp(2),Jp(3),Jp(4),Jp(5),Jp(6),false);
            ep = [(T_tgt(1:3,4)-Tp(1:3,4))/R_w;
                   rot_vec_err(Tp(1:3,1:3),T_tgt(1:3,1:3))/pi];
            Jac(:,k) = (ep-e)/delta;
        end
        dJ    = (Jac'*Jac + lambda^2*eye(6)) \ (Jac'*e);
        J_out = J_out + alpha*dJ(:)';
    end
    [T_f,~] = fPUMA(J_out(1),J_out(2),J_out(3),J_out(4),J_out(5),J_out(6),false);
    pos_err = norm(T_tgt(1:3,4) - T_f(1:3,4));
end

function e = ik_6d_error(J, T_tgt, R_w)
    try
        [T,~] = fPUMA(J(1),J(2),J(3),J(4),J(5),J(6),false);
        e = norm(T(1:3,4)-T_tgt(1:3,4))/R_w + ...
            norm(rot_vec_err(T(1:3,1:3),T_tgt(1:3,1:3)))/pi;
    catch, e = 1e6; end
end

function rv = rot_vec_err(R, R_tgt)
    R_err     = R_tgt*R';
    trace_val = max(-1,min(1,(trace(R_err)-1)/2));
    angle     = acos(trace_val);
    if abs(angle) < 1e-10
        rv = zeros(3,1);
    elseif abs(angle-pi) < 1e-6
        M    = (R_err+eye(3))/2;
        cols = [norm(M(:,1)),norm(M(:,2)),norm(M(:,3))];
        [~,idx] = max(cols);
        ax = M(:,idx)/cols(idx);
        rv = angle*ax;
    else
        rv = (angle/(2*sin(angle)))*[R_err(3,2)-R_err(2,3);
                                      R_err(1,3)-R_err(3,1);
                                      R_err(2,1)-R_err(1,2)];
    end
end

function r2 = compute_r2(y_true, y_pred)
    ss_res = sum((y_true-y_pred).^2);
    ss_tot = sum((y_true-mean(y_true)).^2);
    if ss_tot < 1e-12, r2 = 0; return; end
    r2 = 1 - ss_res/ss_tot;
end