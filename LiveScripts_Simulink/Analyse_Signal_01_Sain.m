%% ANALYSE DU SIGNAL 01 : ÉTAT SAIN (RÉFÉRENCE)
%%
%% Analyse vibratoire multi-domaine d'un système palier hydrodynamique
%% Signal produit par le modèle Simulink PFD_Signal_Generator (v3.1)
%%
%% Ce script analyse le signal de l'état SAIN, qui constitue la référence
%% de tout le travail. Les états défectueux (défauts simples puis défauts
%% mixtes) seront ensuite comparés à cette référence.
%%
%% CONTENU (4 familles d'analyse uniquement) :
%%   0. Production du signal sain à partir du modèle Simulink
%%   1. Analyse temporelle
%%   2. Analyse statistique
%%   3. Analyse fréquentielle (FFT + DSP de Welch)
%%   4. Analyse temps-fréquence (spectrogramme STFT + ondelettes CWT)
%%   5. Synthèse et interprétation
%%
%% Version : 1.1 (Phase 1 - État sain) - 2026-08
%% Compatible : MATLAB R2024b, Signal Processing Toolbox, Statistics and
%%              Machine Learning Toolbox, Wavelet Toolbox
%% ========================================================================

clear; clc; close all;

% Se placer à la racine du projet (dossier parent de LiveScripts_Simulink)
scriptPath = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptPath);
cd(projectRoot);
fprintf('Répertoire de travail : %s\n\n', pwd);

%% Configuration
% =========================================================================
% Fichier signal à analyser (produit par le modèle Simulink)
SIGNAL_FILE = 'data_signaux_simulink/sain_001.mat';

% Répertoire de sortie des figures (300 DPI)
OUTPUT_DIR = 'Figures_Simulink/Sain';
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

EXPORT_DPI = 300;

fprintf('========================================================================\n');
fprintf('   ANALYSE DE L''ÉTAT SAIN - SIGNAL ISSU DU MODÈLE SIMULINK\n');
fprintf('========================================================================\n');
fprintf('Signal  : %s\n', SIGNAL_FILE);
fprintf('Figures : %s\n\n', OUTPUT_DIR);

%% ========================================================================
%% SECTION 0 : PRODUCTION DU SIGNAL SAIN À PARTIR DU MODÈLE SIMULINK
%% ========================================================================
%
% Le signal analysé ici est produit par le modèle Simulink
% PFD_Signal_Generator. Pour l'état sain, la configuration est :
%
%   - Fault_Type          = 1  (Sain : aucun défaut injecté)
%   - Severity_Level      = 1.0 (nominal ; sans effet car le défaut est nul)
%   - Enable_Evolution    = 0  (pas d'évolution temporelle)
%   - Transient_Type      = 1  (aucun transitoire)
%   - Speed_RPM           = 3600 tr/min  (fréquence de rotation 1X = 60 Hz)
%   - Load_Percent        = 70 %
%   - Temperature_C       = 60 °C
%
% CHAÎNE DE PRODUCTION DU SIGNAL (voir le document Production_Signal_Sain) :
%
%   Operating_Conditions : calcule les facteurs de fonctionnement et le
%       nombre de Sommerfeld S = 0.15*exp(-0.03*(T-60))*(Omega/60)/lf,
%       avec lf = 0.3 + 0.7*(charge/100) = 0.79 à 70 % de charge,
%       soit S = 0.190 au point de fonctionnement nominal.
%   Base_Signal    : bruit blanc de base du palier en fonctionnement.
%   Fault_Injection -> Severity_Control -> Transient_Behavior :
%       pour Fault_Type = 1, la sortie de Fault_Injection est NULLE.
%       Le signal sain ne contient donc AUCUNE signature de défaut.
%   Noise_Model    : 7 sources de bruit de mesure réalistes (bruit capteur,
%       interférence secteur 50 Hz, bruit rose, dérive lente sinusoïdale
%       (~0.67 Hz), dérive linéaire du capteur, impulsions parasites de
%       faible amplitude, raie de repliement haute fréquence ~10 kHz).
%   Signal_Sum -> Quantizer : somme des contributions puis quantification
%       (pas de 0.001, effet convertisseur analogique-numérique), appliquée
%       en aval de la somme.
%
% Le signal sain est donc : bruit de base + bruits de mesure, quantifié.
% Dans le cadre de ce modèle de simulation, c'est ce qui caractérise la
% machine saine : un plancher de bruit large bande, sans composante de
% rotation dominante ni signature de défaut. Les seules structures
% présentes sont les artefacts de mesure simulés volontairement.
%
% POUR REPRODUIRE CE SIGNAL :
%   Option A (interface) : ouvrir PFD_Signal_Generator.slx, mettre la
%       constante Fault_Type à 1, lancer Run (5 s), signal dans x_sim.
%   Option B (script)    : >> generate_simulink_signals
%       (régénère automatiquement les 11 signaux, dont sain_001.mat)
%
% =========================================================================

fprintf('SECTION 0 : Production du signal (modèle Simulink)\n');
fprintf('--------------------------------------------------\n');

% Chargement du signal
data = load(SIGNAL_FILE);
x = data.x(:);          % Signal vibratoire (colonne)
fs = double(data.fs);   % Fréquence d'échantillonnage (Hz)
fault = char(data.fault);

N = length(x);          % Nombre d'échantillons
T = (N-1) / fs;         % Durée du support temporel (0 à 5 s inclus)
t = (0:N-1)' / fs;      % Vecteur temps

fprintf('État                : %s (sain)\n', fault);
fprintf('Fréquence d''échantillonnage : %d Hz\n', fs);
fprintf('Durée               : %.2f s\n', T);
fprintf('Nombre d''échantillons : %d\n\n', N);

% Métadonnées (paramètres exacts du modèle lors de la génération)
if isfield(data, 'metadata')
    meta = data.metadata;
    fprintf('Paramètres du modèle Simulink (métadonnées) :\n');
    fprintf('  Vitesse     : %.0f tr/min (%.1f Hz)\n', meta.speed_rpm, meta.speed_rpm/60);
    fprintf('  Charge      : %.0f %%\n', meta.load_percent);
    fprintf('  Température : %.0f °C\n', meta.temperature_C);
    fprintf('  Sommerfeld  : %.4f\n', meta.sommerfeld_number);
    fprintf('  Sévérité    : %s\n', char(meta.severity));
    fprintf('  Générateur  : %s\n\n', char(meta.generator_version));
    Omega = meta.speed_rpm / 60;   % Fréquence de rotation 1X (Hz)
else
    fprintf('Pas de métadonnées : vitesse par défaut 3600 tr/min.\n\n');
    Omega = 60;
end

fprintf('Fréquence de rotation (1X) : %.2f Hz\n', Omega);
fprintf('Harmoniques attendus (défauts) : 2X = %.0f Hz, 3X = %.0f Hz\n', 2*Omega, 3*Omega);
fprintf('Zone sous-synchrone (tourbillonnement d''huile) : %.0f-%.0f Hz\n', ...
    0.42*Omega, 0.48*Omega);
fprintf('--------------------------------------------------\n\n');

%% ========================================================================
%% SECTION 1 : ANALYSE TEMPORELLE
%% ========================================================================
%
% L'analyse temporelle donne une première vue du comportement vibratoire :
% amplitude générale, présence éventuelle de chocs, de modulations ou de
% transitoires. Pour un palier sain, on attend un signal de faible
% amplitude, d'aspect aléatoire, sans chocs répétitifs. Une lente
% ondulation reste visible sur la vue complète : c'est la composante
% ADDITIVE très basse fréquence de la dérive environnementale simulée
% (< 1 Hz), un artefact de mesure et non un phénomène de rotation.
%
% =========================================================================

fprintf('SECTION 1 : Analyse temporelle\n');
fprintf('--------------------------------------------------\n');

%% Figure 1 : signal temporel complet (5 s)
fig1 = figure('Name', 'Signal sain - Vue complète', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

plot(t, x, 'b-', 'LineWidth', 0.4);
xlabel('Temps (s)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 1 : Signal temporel complet - État sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(sprintf('Durée : %.0f s, fs = %d Hz, N = %d échantillons', T, fs, N), ...
    'FontSize', 11);
grid on;
xlim([0, T]);
set(gca, 'FontSize', 11);

exportgraphics(fig1, fullfile(OUTPUT_DIR, 'Fig1_Sain_Temporel_Complet.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 1 exportée : Fig1_Sain_Temporel_Complet.png\n');

%% Figure 2 : zoom temporel (100 premières millisecondes)
% Le zoom permet de vérifier l'absence de motif périodique dominant.
% Les traits verticaux marquent la période de rotation (1/Omega = 16.7 ms) :
% pour un signal sain, aucun motif ne se répète à cette période.
zoom_dur = 0.100;                 % 100 ms
idx_zoom = t <= zoom_dur;
T_rot = 1 / Omega;                % Période de rotation (s)

fig2 = figure('Name', 'Signal sain - Zoom', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

plot(t(idx_zoom)*1000, x(idx_zoom), 'b-', 'LineWidth', 0.8);
hold on;
for kk = 0:floor(zoom_dur / T_rot)
    xline(kk * T_rot * 1000, 'r--', 'LineWidth', 1.0);
end
xlabel('Temps (ms)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 2 : Zoom temporel (100 ms) - État sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(sprintf(['Traits rouges : période de rotation T = %.2f ms ' ...
    '(1X = %.0f Hz). Aucun motif périodique visible.'], T_rot*1000, Omega), ...
    'FontSize', 11);
grid on;
xlim([0, zoom_dur*1000]);
set(gca, 'FontSize', 11);

exportgraphics(fig2, fullfile(OUTPUT_DIR, 'Fig2_Sain_Temporel_Zoom.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 2 exportée : Fig2_Sain_Temporel_Zoom.png\n\n');

%% ========================================================================
%% SECTION 2 : ANALYSE STATISTIQUE
%% ========================================================================
%
% Indicateurs statistiques du signal temporel.
%
% DÉFINITIONS (estimateurs utilisés par MATLAB) :
%   Moyenne          : mu = (1/N) * somme(x_i)
%   Valeur efficace  : RMS = sqrt((1/N) * somme(x_i^2))
%   Écart-type       : sigma = sqrt((1/(N-1)) * somme((x_i - mu)^2))
%                      (estimateur d'échantillon, fonction std)
%   Valeur crête     : max|x|
%   Crête-à-crête    : max(x) - min(x)
%   Asymétrie        : E[(x-mu)^3] / sigma_b^3   (0 = symétrique ;
%                      sigma_b = écart-type biaisé, fonction skewness)
%   Kurtosis         : E[(x-mu)^4] / sigma_b^4   (3 = gaussien ;
%                      kurtosis non normalisé, fonction kurtosis)
%   Facteur de crête : FC = max|x| / RMS
%
% INTERPRÉTATION PHYSIQUE (seuils indicatifs, propres à ce cadre d'étude,
% utilisés de façon identique pour tous les états afin de permettre la
% comparaison sain / défauts) :
%   - Kurtosis proche de 3 : distribution gaussienne, pas de chocs
%     détectables (un kurtosis nettement > 3 signale des événements
%     impulsifs, typiques de certains défauts comme le manque de
%     lubrification ou la cavitation)
%   - Facteur de crête < 5 : pas de pics transitoires sévères
%   - Moyenne proche de 0 : pas de décalage capteur significatif
%
% =========================================================================

fprintf('SECTION 2 : Analyse statistique\n');
fprintf('--------------------------------------------------\n');

stat_moyenne  = mean(x);
stat_rms      = rms(x);
stat_ecart    = std(x);
stat_crete    = max(abs(x));
stat_c2c      = max(x) - min(x);
stat_asym     = skewness(x);
stat_kurt     = kurtosis(x);
stat_fc       = stat_crete / stat_rms;

% Tableau des 8 indicateurs principaux
noms = {'Moyenne'; 'Valeur efficace (RMS)'; 'Écart-type'; 'Valeur crête'; ...
    'Crête-à-crête'; 'Asymétrie (skewness)'; 'Kurtosis'; 'Facteur de crête'};
valeurs = [stat_moyenne; stat_rms; stat_ecart; stat_crete; ...
    stat_c2c; stat_asym; stat_kurt; stat_fc];
references = {'~ 0 (pas de biais)'; '-'; '-'; '-'; '-'; ...
    '0 = symétrique'; '3 = gaussien'; '< 5 = normal'};

tableau_stats = table(noms, valeurs, references, ...
    'VariableNames', {'Indicateur', 'Valeur', 'Reference_sain'});
disp(tableau_stats);

% Export du tableau statistique (livrable) en CSV UTF-8
writetable(tableau_stats, fullfile(OUTPUT_DIR, 'Tableau_Statistiques_Sain.csv'), ...
    'Encoding', 'UTF-8');
fprintf('  Tableau exporté : Tableau_Statistiques_Sain.csv\n');

fprintf('\nINTERPRÉTATION (référence saine ; seuils indicatifs du cadre d''étude) :\n');
if stat_kurt < 4
    fprintf('  OK : kurtosis = %.2f, proche de 3 (gaussien) -> pas de chocs detectables\n', stat_kurt);
else
    fprintf('  ATTENTION : kurtosis = %.2f, élevé pour un état sain\n', stat_kurt);
end
if stat_fc < 5
    fprintf('  OK : facteur de crête = %.2f, dans la plage normale\n', stat_fc);
else
    fprintf('  ATTENTION : facteur de crête = %.2f, élevé\n', stat_fc);
end
if abs(stat_moyenne) < 0.01
    fprintf('  OK : moyenne = %.4f, pas de biais significatif\n', stat_moyenne);
else
    fprintf('  NOTE : moyenne = %.4f (légère dérive du capteur simulée)\n', stat_moyenne);
end
fprintf(['  NOTE : le modèle injecte des impulsions parasites de faible\n' ...
         '  amplitude ; elles restent noyées dans le plancher de bruit\n' ...
         '  (kurtosis ~ 3) et ne créent aucune impulsivité mesurable.\n\n']);

%% Figure 3 : distribution d'amplitude (histogramme + ajustement gaussien + Q-Q)
fig3 = figure('Name', 'Signal sain - Distribution', ...
    'Position', [100, 100, 1200, 550], 'Color', 'white');

subplot(1, 2, 1);
histogram(x, 100, 'Normalization', 'pdf', 'FaceColor', [0.3, 0.5, 0.8], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.7);
hold on;
x_range = linspace(min(x), max(x), 200);
plot(x_range, normpdf(x_range, stat_moyenne, stat_ecart), 'r-', 'LineWidth', 2.2);
xlabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Densité de probabilité', 'FontSize', 13, 'FontWeight', 'bold');
title('Distribution d''amplitude', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Signal mesuré', 'Ajustement gaussien'}, 'Location', 'northwest', ...
    'FontSize', 10);
grid on;
set(gca, 'FontSize', 11);

annotation('textbox', [0.33, 0.62, 0.14, 0.24], 'String', sprintf( ...
    ['Indicateurs :\n' ...
     'Moyenne = %.4f\n' ...
     'Écart-type = %.4f\n' ...
     'Asymétrie = %.3f\n' ...
     'Kurtosis = %.3f'], ...
    stat_moyenne, stat_ecart, stat_asym, stat_kurt), ...
    'FontSize', 9, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
    'FitBoxToText', 'on');

subplot(1, 2, 2);
qqplot(x);
title('Diagramme Q-Q (comparaison à la loi normale)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Quantiles normaux théoriques', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Quantiles de l''échantillon', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 11);

sgtitle('Figure 3 : Analyse statistique de la distribution - État sain', ...
    'FontSize', 15, 'FontWeight', 'bold');

exportgraphics(fig3, fullfile(OUTPUT_DIR, 'Fig3_Sain_Distribution_Amplitude.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 3 exportée : Fig3_Sain_Distribution_Amplitude.png\n\n');

%% ========================================================================
%% SECTION 3 : ANALYSE FRÉQUENTIELLE
%% ========================================================================
%
% Deux outils complémentaires :
%   1. Spectre d'amplitude par FFT (transformée de Fourier discrète) :
%      X[k] = somme_n x[n] * exp(-j*2*pi*k*n/N)
%   2. Densité spectrale de puissance (DSP) par la méthode de Welch :
%      moyenne des périodogrammes sur des segments fenêtrés (Hann 4096,
%      recouvrement 50 %), plus robuste au bruit que la FFT brute.
%
% La composante continue (moyenne) est retirée avant le calcul pour que le
% bin à 0 Hz ne masque pas le contenu utile.
%
% FRÉQUENCES CARACTÉRISTIQUES À 3600 tr/min :
%   1X = 60 Hz (balourd), 2X = 120 Hz et 3X = 180 Hz (désalignement),
%   0.42-0.48X = 25-29 Hz (tourbillonnement d'huile),
%   1500-2500 Hz (cavitation).
% Pour l'état SAIN, AUCUNE de ces composantes ne doit dominer :
% c'est précisément leur ABSENCE qui définit la référence.
%
% ARTEFACTS DE MESURE SIMULÉS (présents volontairement, à connaître) :
%   - contenu très basse fréquence (< 1 Hz) : dérive lente simulée ;
%     c'est le MAXIMUM GLOBAL du spectre, exclu de la recherche de
%     fréquence dominante (seuil à 10 % de la vitesse de rotation) ;
%   - une raie à 50 Hz : interférence électromagnétique du secteur (EMI),
%     à ne pas confondre avec le 1X à 60 Hz ;
%   - une raie de repliement de très faible amplitude vers 10.04 kHz
%     (visible seulement en bande complète, hors de la zone tracée ici).
%
% =========================================================================

fprintf('SECTION 3 : Analyse fréquentielle\n');
fprintf('--------------------------------------------------\n');

x_ac = x - mean(x);   % Suppression de la composante continue

% --- Spectre d'amplitude FFT ---
NFFT = 2^nextpow2(N);
X_fft = fft(x_ac, NFFT);
f_fft = (0:NFFT/2)' * fs / NFFT;
X_mag = abs(X_fft(1:NFFT/2+1)) / N;
X_mag(2:end-1) = 2 * X_mag(2:end-1);

% --- DSP de Welch ---
fen = 4096;
[Pxx, f_psd] = pwelch(x_ac, hann(fen), fen/2, fen, fs);

% --- Indicateurs spectraux ---
% Recherche de la fréquence dominante au-dessus de 10 % de la vitesse de
% rotation (seuil = 6 Hz ici), pour exclure la dérive très basse fréquence
% qui constitue le maximum global du spectre.
f_min = max(f_psd(2), 0.1 * Omega);
masque = f_psd >= f_min;
f_valides = f_psd(masque);
[~, i_max] = max(Pxx(masque));
f_dominante = f_valides(i_max);
% Maximum global : relevé sur la FFT (grille fine de fs/NFFT = 0.16 Hz),
% car la grille de Welch (pas de 5 Hz) ne résout pas la dérive < 1 Hz
% (son énergie tombe dans le bin 0-2.5 Hz).
[~, i_glob] = max(X_mag(2:end));
f_max_globale = f_fft(i_glob + 1);

centroide = sum(f_psd .* Pxx) / sum(Pxx);
Pxx_n = Pxx / sum(Pxx);
entropie = -sum(Pxx_n .* log2(Pxx_n + eps));
platitude = exp(mean(log(Pxx + eps))) / (mean(Pxx) + eps);

% Niveaux spectraux (racine de la DSP, en unité/sqrt(Hz)) aux fréquences
% caractéristiques. Ces niveaux servent de référence RELATIVE pour la
% comparaison avec les défauts (mêmes paramètres d'estimation partout) ;
% ce ne sont pas des amplitudes de raies au sens du spectre FFT.
% La grille de Welch a un pas de fs/fen = 5 Hz : on relève le bin le plus
% proche de chaque fréquence cible et on affiche sa fréquence réelle.
[~, i1X] = min(abs(f_psd - Omega));
[~, i2X] = min(abs(f_psd - 2*Omega));
[~, i3X] = min(abs(f_psd - 3*Omega));
[~, iSub] = min(abs(f_psd - 0.45*Omega));
niv_1X  = sqrt(Pxx(i1X));
niv_2X  = sqrt(Pxx(i2X));
niv_3X  = sqrt(Pxx(i3X));
niv_sub = sqrt(Pxx(iSub));

% Critère opérationnel d'absence de pic : un bin caractéristique est
% considéré non significatif si sa DSP ne dépasse pas de plus de 3 dB le
% plancher médian local (médiane de la DSP entre 10 et 300 Hz).
plancher_med = median(Pxx(f_psd >= 10 & f_psd <= 300));
exces_dB = 10*log10([Pxx(iSub), Pxx(i1X), Pxx(i2X), Pxx(i3X)] / plancher_med);

fprintf('\n--- INDICATEURS SPECTRAUX ---\n');
fprintf('Maximum global (FFT, grille %.2f Hz) : %.2f Hz (dérive simulée < 1 Hz, artefact)\n', ...
    fs/NFFT, f_max_globale);
fprintf('Fréquence dominante (recherche >= %.0f Hz) : %.2f Hz\n', f_min, f_dominante);
fprintf('Centroïde spectral  : %.1f Hz\n', centroide);
fprintf('Entropie spectrale  : %.2f bits\n', entropie);
fprintf('Platitude spectrale : %.4f (1 = bruit large bande, 0 = tonal)\n\n', platitude);
fprintf('Niveaux spectraux (racine de DSP) aux fréquences caractéristiques,\n');
fprintf('relevés au bin de Welch le plus proche, avec l''excès par rapport au\n');
fprintf('plancher médian local 10-300 Hz (critère : pic significatif si > 3 dB) :\n');
fprintf('  0.45X, cible %.0f Hz, bin %.0f Hz : %.3e  (%+.1f dB)  tourbillonnement d''huile\n', ...
    0.45*Omega, f_psd(iSub), niv_sub, exces_dB(1));
fprintf('  1X,    cible %.0f Hz, bin %.0f Hz : %.3e  (%+.1f dB)  balourd\n', ...
    Omega, f_psd(i1X), niv_1X, exces_dB(2));
fprintf('  2X,    cible %.0f Hz, bin %.0f Hz : %.3e  (%+.1f dB)  désalignement\n', ...
    2*Omega, f_psd(i2X), niv_2X, exces_dB(3));
fprintf('  3X,    cible %.0f Hz, bin %.0f Hz : %.3e  (%+.1f dB)  désalignement\n', ...
    3*Omega, f_psd(i3X), niv_3X, exces_dB(4));

fprintf('\nINTERPRÉTATION (référence saine) :\n');
if abs(f_dominante - 50) < 2
    fprintf(['  Hors dérive (< 1 Hz), la composante la plus énergétique (%.1f Hz)\n' ...
             '  correspond à l''interférence secteur 50 Hz simulée (artefact de\n' ...
             '  mesure), PAS à une composante de rotation : le 1X à %.0f Hz\n' ...
             '  ne domine pas le spectre.\n'], f_dominante, Omega);
elseif abs(f_dominante - Omega)/Omega < 0.10
    fprintf('  La fréquence dominante (%.1f Hz) est proche du 1X (%.0f Hz).\n', ...
        f_dominante, Omega);
else
    fprintf('  Fréquence dominante : %.1f Hz (à examiner sur le spectre).\n', f_dominante);
end
if platitude > 0.5
    fprintf(['  Platitude spectrale élevée (%.2f) : le spectre est un plancher\n' ...
             '  de bruit large bande, sans raie dominante -> conforme à un\n' ...
             '  palier sain dans le cadre du modèle.\n'], platitude);
end
if all(exces_dB < 3)
    fprintf(['  Aucun pic significatif à 1X, 2X, 3X ni en zone sous-synchrone\n' ...
             '  (tous les excès < 3 dB au-dessus du plancher médian local) :\n' ...
             '  absence de signature de défaut. Ces niveaux serviront de\n' ...
             '  référence de comparaison pour les défauts.\n\n']);
else
    fprintf(['  ATTENTION : au moins un bin caractéristique dépasse le plancher\n' ...
             '  médian local de plus de 3 dB ; à examiner sur la Figure 6.\n\n']);
end

%% Figure 4 : spectre d'amplitude FFT (0-500 Hz)
fig4 = figure('Name', 'Signal sain - Spectre FFT', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

plot(f_fft, X_mag, 'b-', 'LineWidth', 0.8);
hold on;
xline(Omega,   'r--', '1X',    'LineWidth', 1.3, 'FontSize', 11, 'LabelOrientation', 'horizontal');
xline(2*Omega, 'g--', '2X',    'LineWidth', 1.3, 'FontSize', 11, 'LabelOrientation', 'horizontal');
xline(3*Omega, 'm--', '3X',    'LineWidth', 1.3, 'FontSize', 11, 'LabelOrientation', 'horizontal');
xline(50, 'k:', 'EMI 50 Hz', 'LineWidth', 1.2, 'FontSize', 10, ...
    'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'middle');
xlabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 4 : Spectre d''amplitude (FFT) - État sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(['Zoom 0-500 Hz. Seuls émergent la raie EMI 50 Hz et la dérive ' ...
    'très basse fréquence (< 1 Hz) ; pas de 1X/2X/3X.'], 'FontSize', 11);
grid on;
xlim([0, 500]);
set(gca, 'FontSize', 11);

exportgraphics(fig4, fullfile(OUTPUT_DIR, 'Fig4_Sain_Spectre_FFT.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 4 exportée : Fig4_Sain_Spectre_FFT.png\n');

%% Figure 5 : DSP de Welch (bande complète 0 - fs/2, échelle log)
fig5 = figure('Name', 'Signal sain - DSP Welch', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

semilogy(f_psd, Pxx, 'b-', 'LineWidth', 0.9);
hold on;
% Repères 1X/2X/3X sans étiquettes (trop proches à cette échelle ;
% voir la Figure 6 pour la zone 0-500 Hz étiquetée)
xline(Omega,   'r--', 'LineWidth', 1.3);
xline(2*Omega, 'g--', 'LineWidth', 1.3);
xline(3*Omega, 'm--', 'LineWidth', 1.3);
% Raie de repliement simulée (artefact haute fréquence)
xline(10040, 'k:', 'repliement ~10.04 kHz', 'LineWidth', 1.0, 'FontSize', 9, ...
    'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'top');
xlabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('DSP (unité^2/Hz)', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 5 : Densité spectrale de puissance (méthode de Welch) - État sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(sprintf(['Bande complète 0 - %d Hz. Fenêtre de Hann %d points, ' ...
    'recouvrement 50 %%'], fs/2, fen), 'FontSize', 11);
grid on;
xlim([0, fs/2]);
set(gca, 'FontSize', 11);

annotation('textbox', [0.55, 0.68, 0.2, 0.2], 'String', sprintf( ...
    ['Indicateurs spectraux :\n' ...
     'max global (FFT) = %.2f Hz (dérive)\n' ...
     'f dominante (>= 6 Hz) = %.1f Hz\n' ...
     'Entropie = %.2f bits\n' ...
     'Platitude = %.3f'], ...
    f_max_globale, f_dominante, entropie, platitude), ...
    'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
    'FitBoxToText', 'on');

exportgraphics(fig5, fullfile(OUTPUT_DIR, 'Fig5_Sain_DSP_Welch.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 5 exportée : Fig5_Sain_DSP_Welch.png\n');

%% Figure 6 : DSP zoom basses fréquences (0-500 Hz, en dB)
fig6 = figure('Name', 'Signal sain - DSP zoom', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

plot(f_psd, 10*log10(Pxx + eps), 'b-', 'LineWidth', 1.0);
hold on;
xline(Omega,   'r--', '1X = 60 Hz',  'LineWidth', 1.3, 'FontSize', 10, 'LabelOrientation', 'horizontal');
xline(2*Omega, 'g--', '2X = 120 Hz', 'LineWidth', 1.3, 'FontSize', 10, 'LabelOrientation', 'horizontal');
xline(3*Omega, 'm--', '3X = 180 Hz', 'LineWidth', 1.3, 'FontSize', 10, 'LabelOrientation', 'horizontal');
xline(50, 'k:', 'EMI 50 Hz', 'LineWidth', 1.2, 'FontSize', 10, ...
    'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');

% Zone sous-synchrone (où apparaîtrait un tourbillonnement d'huile)
xr = [0.42*Omega, 0.48*Omega];
yl = ylim;
patch([xr(1) xr(2) xr(2) xr(1)], [yl(1) yl(1) yl(2) yl(2)], ...
    [1.0 0.9 0.7], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
text(mean(xr), yl(2)-4, 'zone 0.42-0.48X', 'FontSize', 9, ...
    'HorizontalAlignment', 'center');

xlabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('DSP (dB)', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 6 : DSP en zone basses fréquences (0-500 Hz) - État sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(['Absence de raies aux fréquences caractéristiques des défauts : ' ...
    'hormis la remontée de dérive sous ~5 Hz, seule la raie EMI 50 Hz émerge.'], ...
    'FontSize', 11);
grid on;
xlim([0, 500]);
set(gca, 'FontSize', 11);

exportgraphics(fig6, fullfile(OUTPUT_DIR, 'Fig6_Sain_DSP_Zoom.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 6 exportée : Fig6_Sain_DSP_Zoom.png\n\n');

%% ========================================================================
%% SECTION 4 : ANALYSE TEMPS-FRÉQUENCE
%% ========================================================================
%
% Deux méthodes complémentaires :
%
% 1. Spectrogramme (transformée de Fourier à court terme, STFT) :
%    S(t,f) = |intégrale de x(tau)*w(tau-t)*exp(-j*2*pi*f*tau) dtau|^2
%    Compromis temps-fréquence fixé par la longueur de la fenêtre.
%    Fenêtre choisie : 2048 points (100 ms), pas de grille fs/2048 = 10 Hz.
%    NOTE : la largeur effective de la fenêtre de Hann (~20 Hz) ne permet
%    pas de résoudre finement deux composantes distantes de 10 Hz ; la
%    distinction 50 Hz (EMI) / 60 Hz (1X) est établie par l'analyse
%    fréquentielle (Figures 4 à 6, résolution 0.16 à 5 Hz). Le rôle du
%    spectrogramme est de montrer l'ÉVOLUTION TEMPORELLE du contenu.
%
% 2. Transformée en ondelettes continue (CWT, ondelette de Morlet
%    analytique) :
%    W(a,b) = (1/sqrt(a)) * intégrale de x(t)*psi*((t-b)/a) dt
%    Résolution multi-échelle : fine en temps pour les hautes fréquences,
%    fine en fréquence pour les basses fréquences. Idéale pour révéler
%    des transitoires (chocs, bouffées) que la STFT étale.
%
% Les deux représentations sont affichées dans la bande 0-500 Hz, celle
% des signatures de défaut basse fréquence ; les conclusions tirées ici
% ne portent que sur cette bande. Pour l'état sain, on attend un contenu
% globalement stable dans le temps : pas de bande horizontale liée à un
% défaut, pas de transitoire détectable lié à un défaut. La seule
% structure attendue est la bande d'énergie autour de 50 Hz (EMI,
% constante dans le temps) et l'énergie de dérive tout en bas de la bande.
%
% =========================================================================

fprintf('SECTION 4 : Analyse temps-fréquence\n');
fprintf('--------------------------------------------------\n');

%% Figure 7 : spectrogramme STFT
fig7 = figure('Name', 'Signal sain - Spectrogramme', ...
    'Position', [100, 100, 1200, 600], 'Color', 'white');
% tiledlayout : réserve l'espace du titre au-dessus de l'image
% (un titre d'axes classique n'est pas rendu de façon fiable au-dessus
% d'imagesc en export sans affichage)
tl7 = tiledlayout(fig7, 1, 1, 'Padding', 'compact');
nexttile(tl7);

fen_spec = 2048;                    % 100 ms -> résolution 10 Hz
rec_spec = round(0.875 * fen_spec); % 87.5 % de recouvrement
nfft_spec = 4096;
[S, F, T_spec] = spectrogram(x, hann(fen_spec), rec_spec, nfft_spec, fs);
S_dB = 10 * log10(abs(S).^2 + eps);

imagesc(T_spec, F, S_dB);
axis xy;
colormap('jet');
cb = colorbar;
cb.Label.String = 'Puissance (dB)';
cb.Label.FontSize = 11;
ylim([0, 500]);
hold on;
yline(Omega,   'w--', 'LineWidth', 1.2);
yline(2*Omega, 'w--', 'LineWidth', 1.2);
yline(3*Omega, 'w--', 'LineWidth', 1.2);
text(T*0.02, Omega+12,   '1X', 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');
text(T*0.02, 2*Omega+12, '2X', 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');
text(T*0.02, 3*Omega+12, '3X', 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');

xlabel('Temps (s)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
title(tl7, 'Figure 7 : Spectrogramme (STFT) - État sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(tl7, sprintf(['Fenêtre de Hann %d points (%.0f ms), recouvrement ' ...
    '%.1f %%. Bande 0-500 Hz : contenu stable dans le temps, bande EMI ' ...
    '~50 Hz constante, aucun transitoire lié à un défaut détectable.'], ...
    fen_spec, 1000*fen_spec/fs, 100*rec_spec/fen_spec), 'FontSize', 11);
set(gca, 'FontSize', 11);

exportgraphics(fig7, fullfile(OUTPUT_DIR, 'Fig7_Sain_Spectrogramme_STFT.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 7 exportée : Fig7_Sain_Spectrogramme_STFT.png\n');

%% Figure 8 : transformée en ondelettes continue (CWT)
fig8 = figure('Name', 'Signal sain - CWT', ...
    'Position', [100, 100, 1200, 600], 'Color', 'white');

[cfs, frq] = cwt(x, 'amor', fs);

surface(t, frq, abs(cfs));
axis tight;
shading interp;
view(0, 90);
colormap('parula');
cb = colorbar;
cb.Label.String = 'Module';
cb.Label.FontSize = 11;
ylim([0, 500]);
hold on;
yline(Omega,   'w--', 'LineWidth', 1.2);
yline(2*Omega, 'w--', 'LineWidth', 1.2);
yline(3*Omega, 'w--', 'LineWidth', 1.2);

xlabel('Temps (s)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 8 : Transformée en ondelettes continue (Morlet analytique) - État sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(['Bande 0-500 Hz : aucun transitoire ni composante persistante ' ...
    'liée à un défaut détectable.'], 'FontSize', 11);
set(gca, 'FontSize', 11, 'YScale', 'linear');

exportgraphics(fig8, fullfile(OUTPUT_DIR, 'Fig8_Sain_CWT.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 8 exportée : Fig8_Sain_CWT.png\n\n');

%% ========================================================================
%% SECTION 5 : SYNTHÈSE ET INTERPRÉTATION
%% ========================================================================

fprintf('SECTION 5 : Synthèse\n');
fprintf('--------------------------------------------------\n');

tableau_txt = sprintf([ ...
'  %-24s %12.6f   (~ 0 : pas de biais)\n' ...
'  %-24s %12.6f\n' ...
'  %-24s %12.6f\n' ...
'  %-24s %12.6f\n' ...
'  %-24s %12.6f\n' ...
'  %-24s %12.4f   (0 = symétrique)\n' ...
'  %-24s %12.4f   (3 = gaussien)\n' ...
'  %-24s %12.4f   (< 5 = normal)\n'], ...
'Moyenne', stat_moyenne, 'Valeur efficace (RMS)', stat_rms, ...
'Écart-type', stat_ecart, 'Valeur crête', stat_crete, ...
'Crête-à-crête', stat_c2c, 'Asymétrie', stat_asym, ...
'Kurtosis', stat_kurt, 'Facteur de crête', stat_fc);

interp_txt = sprintf([ ...
'INTERPRÉTATION DES RÉSULTATS - ÉTAT SAIN (RÉFÉRENCE)\n' ...
'=====================================================\n\n' ...
'Signal analysé : %s\n' ...
'Produit par le modèle Simulink PFD_Signal_Generator\n' ...
'(3600 tr/min soit 1X = 60 Hz, charge 70 %%, température 60 °C,\n' ...
'fs = %d Hz, durée %.0f s). Les conclusions ci-dessous décrivent le\n' ...
'comportement sain tel qu''encodé par le modèle de simulation ; elles\n' ...
'servent de référence interne pour la comparaison avec les défauts\n' ...
'simulés dans les phases suivantes.\n\n' ...
'1. ANALYSE TEMPORELLE\n' ...
'Le signal présente un aspect aléatoire et de faible amplitude\n' ...
'(valeur efficace RMS = %.4f, valeur crête = %.3f). Une lente\n' ...
'ondulation est visible sur la vue complète : c''est la composante\n' ...
'additive très basse fréquence de la dérive environnementale simulée\n' ...
'(< 1 Hz), un artefact de mesure. Le zoom sur 100 ms ne révèle aucun\n' ...
'motif périodique à la période de rotation (%.2f ms) ni chocs\n' ...
'répétitifs : comportement attendu d''un palier hydrodynamique sain,\n' ...
'où le film d''huile amortit les vibrations de l''arbre.\n\n' ...
'2. ANALYSE STATISTIQUE\n' ...
'Tableau des indicateurs :\n%s\n' ...
'Au regard des seuils indicatifs du cadre d''étude (appliqués à\n' ...
'l''identique à tous les états), le kurtosis vaut %.2f, très proche\n' ...
'de 3 (distribution gaussienne), et le facteur de crête vaut %.2f\n' ...
'(< 5) : aucune impulsivité détectable. Les impulsions parasites\n' ...
'simulées par le modèle restent noyées dans le plancher de bruit.\n' ...
'L''asymétrie est quasi nulle (%.3f) et la moyenne négligeable\n' ...
'(%.4f). L''histogramme et le diagramme Q-Q sont compatibles avec\n' ...
'une distribution normale.\n\n' ...
'3. ANALYSE FRÉQUENTIELLE\n' ...
'Le spectre est un plancher de bruit large bande (platitude spectrale\n' ...
'%.3f, entropie %.2f bits), SANS raie dominante liée à la rotation :\n' ...
'ni 1X (60 Hz, signature de balourd), ni 2X/3X (120/180 Hz, signature\n' ...
'de désalignement), ni composante sous-synchrone entre 25 et 29 Hz\n' ...
'(signature de tourbillonnement d''huile) : aux bins correspondants,\n' ...
'la DSP ne dépasse pas le plancher médian local de plus de 3 dB\n' ...
'(critère opérationnel retenu). Trois artefacts de mesure\n' ...
'simulés sont identifiés et ne doivent pas être confondus avec des\n' ...
'signatures de défaut : (i) la dérive très basse fréquence (< 1 Hz),\n' ...
'qui constitue le maximum global du spectre ; (ii) l''interférence\n' ...
'secteur à 50 Hz, composante la plus énergétique hors dérive, à ne\n' ...
'pas confondre avec le 1X à 60 Hz ; (iii) une raie de repliement de\n' ...
'très faible amplitude vers 10.04 kHz. C''est précisément l''ABSENCE\n' ...
'des raies caractéristiques qui définit la signature de l''état sain\n' ...
'et en fait la référence de comparaison pour tous les défauts.\n\n' ...
'4. ANALYSE TEMPS-FRÉQUENCE (bande affichée : 0-500 Hz)\n' ...
'Le spectrogramme (STFT, fenêtre de Hann 2048 points) montre un\n' ...
'contenu spectral globalement stable dans le temps : la seule bande\n' ...
'horizontale persistante est la bande d''énergie autour de 50 Hz\n' ...
'(EMI, constante dans le temps, donc non liée à un phénomène\n' ...
'tournant ; la distinction fine 50/60 Hz est établie par l''analyse\n' ...
'fréquentielle, Figures 4 à 6, et non par la STFT dont la fenêtre a\n' ...
'une largeur effective d''environ 20 Hz). L''énergie de dérive\n' ...
'apparaît tout en bas de la bande. La transformée en ondelettes\n' ...
'(CWT, Morlet analytique), plus fine en temps aux hautes fréquences,\n' ...
'ne révèle aucun transitoire lié à un défaut détectable dans cette\n' ...
'bande ; les impulsions parasites simulées, de très faible amplitude,\n' ...
'restent indiscernables du fond. En résumé : aucun transitoire ni\n' ...
'composante 1X détectable dans la bande affichée, et un contenu\n' ...
'spectral globalement stable hormis les artefacts modélisés.\n\n' ...
'CONCLUSION\n' ...
'Le signal sain issu du modèle Simulink présente toutes les\n' ...
'caractéristiques attendues de l''état sain tel qu''encodé par le\n' ...
'modèle : distribution gaussienne (kurtosis proche de 3), spectre\n' ...
'plat sans signature de défaut, contenu temps-fréquence globalement\n' ...
'stable dans la bande analysée,\n' ...
'artefacts de mesure identifiés et expliqués. Il constitue la\n' ...
'RÉFÉRENCE à laquelle les 7 défauts simples et les 3 défauts mixtes\n' ...
'seront comparés dans les phases suivantes.\n'], ...
SIGNAL_FILE, fs, T, stat_rms, stat_crete, T_rot*1000, tableau_txt, ...
stat_kurt, stat_fc, stat_asym, stat_moyenne, platitude, entropie);

% Sauvegarde du texte d'interprétation
fid = fopen(fullfile(OUTPUT_DIR, 'Interpretation_Sain.txt'), 'w', 'n', 'UTF-8');
fprintf(fid, '%s', interp_txt);
fclose(fid);

fprintf('%s\n', interp_txt);
fprintf('  Texte d''interprétation sauvegardé : Interpretation_Sain.txt\n\n');

fprintf('========================================================================\n');
fprintf('   ANALYSE TERMINÉE - 8 figures + tableau CSV exportés dans %s\n', OUTPUT_DIR);
fprintf('========================================================================\n');
fprintf('Prochaine étape : analyse des 7 défauts simples (Phase 2).\n\n');
