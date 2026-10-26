%% generate_reference.m
% Runs the original MATLAB implementation (https://github.com/WildSolar/General_PV_Irradiation_Model,
% folder 'source', cloned next to this repository) on a
% set of reference cases and stores the intermediate and final results in
% tests/data/. These files are used by the Python test suite to check
% that legacy mode reproduces the MATLAB code.
%
% Run from any directory with:  matlab -batch "run('<path>/generate_reference.m')"

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '..', '..', 'General_PV_Irradiation_Model', 'source'));
addpath(fullfile(here, 'stubs'));   % headless waitbar stub (must shadow the builtin)
outdir = fullfile(here, '..', 'data');
if ~exist(outdir, 'dir'); mkdir(outdir); end
global waitbarhdl
waitbarhdl = [];

%% Irradiation cases
% 1-5: example_script.m, 6: high sun (> 1 rad -> complex airmass path),
% 7: overcast (Perez bin 1), 8: sun from the west, 9: DHI < 0 (non-physical)
irradiation.DNI = [linspace(550,600,5), 800, 2, 300, 900];
irradiation.GHI = [linspace(200,400,5), 950, 150, 200, 300];
irradiation.sunposition.azimuth_deg = [linspace(-40,-20,5), 10, 0, 100, -5];
irradiation.sunposition.elevation_deg = [linspace(11,16,5), 62, 30, 25, 20];
irradiation.omega = [0.79*ones(1,5), 0.5, 0.79, 0.3, 0.6];
irradiation.omega_fg = [0.3*ones(1,5), 0.2, 0.5, 0.2, 0.3];

%% Reduced-precision simulation options
simopts.name = 'ref';
simopts.savePlots = 0;
simopts.n_chi = 20;
simopts.n_sigma = 60;
simopts.n_phi = 30;
simopts.n_sig = 30;
simopts.n_d = 20;
simopts.n_zeta = 20;
simopts.hemispherePrecision = 'low';
simopts.maxRowsConsidered = 20;

%% Geometries
geoms = {};
g.mu_deg = 60; g.gamma_deg = 20; g.alpha_g_deg = 0; g.rho_bar = 1.125; g.h_bar = 1.84;
g.gamma_fg_deg = 20; g.alpha_fg_deg = 160; geoms{end+1} = g;          % example_script
g.mu_deg = 90; g.gamma_deg = 30; g.alpha_g_deg = -25; g.rho_bar = 0.9; g.h_bar = 1.2;
g.gamma_fg_deg = 10; g.alpha_fg_deg = 200; geoms{end+1} = g;          % vertical, steep, rotated
g.mu_deg = 30; g.gamma_deg = 0; g.alpha_g_deg = 15; g.rho_bar = 2.5; g.h_bar = 1.0;
g.gamma_fg_deg = 5; g.alpha_fg_deg = 180; geoms{end+1} = g;           % horizontal ground

%% Sky (geometry independent)
[sky_elements, areasinelev, zenithvec] = hemispherePartition(simopts.hemispherePrecision);
N = length(irradiation.DNI);
radiationMaps = zeros(N, length(sky_elements));
for i = 1:N
    sunposition.azimuth = irradiation.sunposition.azimuth_deg(i)/180*pi;
    sunposition.zenith = pi/2 - irradiation.sunposition.elevation_deg(i)/180*pi;
    radiationMaps(i,:) = totalSkyLuminance(sky_elements, irradiation.DNI(i), irradiation.GHI(i), sunposition, areasinelev, zenithvec);
end
sky_area = cellfun(@(s) s.area, sky_elements);
sky_elev = cellfun(@(s) s.elevation_deg, sky_elements);
sky_azim = cellfun(@(s) s.azimuth_deg, sky_elements);
save(fullfile(outdir, 'ref_sky.mat'), 'radiationMaps', 'sky_area', 'sky_elev', 'sky_azim', 'irradiation', 'simopts');

%% Per geometry: multiplication table and results
for k = 1:length(geoms)
    conf = geoms{k};
    mu = conf.mu_deg/180*pi; gamma = conf.gamma_deg/180*pi;
    chi_vec = linspace(-0.5, 0.5, simopts.n_chi + 1);
    sigma_vec = linspace(0, 180, simopts.n_sigma + 1)/180*pi;
    MT = buildMultiplicationTable(mu, conf.rho_bar, gamma, conf.h_bar, chi_vec, sigma_vec, simopts);
    res = runModel(conf, irradiation, simopts);
    [front, back, fd, bd, fng, bng, ffg] = deal(zeros(N, simopts.n_chi + 1));
    globalCheck = zeros(N,1);
    for i = 1:N
        front(i,:) = res{i}.irradFront; back(i,:) = res{i}.irradBack;
        fd(i,:) = res{i}.irrad_fract_front_direct; bd(i,:) = res{i}.irrad_fract_back_direct;
        fng(i,:) = res{i}.irrad_fract_front_nearground_refl; bng(i,:) = res{i}.irrad_fract_back_nearground_refl;
        ffg(i,:) = res{i}.irrad_fract_front_farground_refl;
        globalCheck(i) = res{i}.globalCheck;
    end
    save(fullfile(outdir, sprintf('ref_geom%d.mat', k)), 'conf', 'MT', 'front', 'back', 'fd', 'bd', 'fng', 'bng', 'ffg', 'globalCheck');
end

%% Full-precision run of example_script.m settings (first irradiation case only)
full = simopts;
full.n_chi = 100; full.n_sigma = 180; full.n_phi = 100; full.n_sig = 100;
full.n_d = 50; full.n_zeta = 50; full.hemispherePrecision = 'high';
irr1.DNI = 550; irr1.GHI = 200; irr1.sunposition.azimuth_deg = -40;
irr1.sunposition.elevation_deg = 11; irr1.omega = 0.79; irr1.omega_fg = 0.3;
tic; res = runModel(geoms{1}, irr1, full); t_full = toc;
front = res{1}.irradFront; back = res{1}.irradBack;
fd = res{1}.irrad_fract_front_direct; bd = res{1}.irrad_fract_back_direct;
fng = res{1}.irrad_fract_front_nearground_refl; bng = res{1}.irrad_fract_back_nearground_refl;
ffg = res{1}.irrad_fract_front_farground_refl;
save(fullfile(outdir, 'ref_full.mat'), 'front', 'back', 'fd', 'bd', 'fng', 'bng', 'ffg', 't_full', 'full');
fprintf('Reference generation done (full run %.1f s)\n', t_full);
