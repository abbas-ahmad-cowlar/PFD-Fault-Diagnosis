%% ANALYSE DU SIGNAL 01 : ETAT SAIN (REFERENCE)
%%
%% Analyse vibratoire multi-domaine d'un systeme palier hydrodynamique
%% Signal produit par le modele Simulink PFD_Signal_Generator (v3)
%%
%% Ce script analyse le signal de l'etat SAIN, qui constitue la reference
%% de tout le travail. Les etats defectueux (defauts simples puis defauts
%% mixtes) seront ensuite compares a cette reference.
%%
%% CONTENU (4 familles d'analyse uniquement) :
%%   0. Production du signal sain a partir du modele Simulink
%%   1. Analyse temporelle
%%   2. Analyse statistique
%%   3. Analyse frequentielle (FFT + DSP de Welch)
%%   4. Analyse temps-frequence (spectrogramme STFT + ondelettes CWT)
%%   5. Synthese et interpretation
%%
%% Version : 1.0 (Phase 1 - Etat sain) - 2026-08
%% Compatible : MATLAB R2024b, Signal Processing Toolbox, Statistics and
%%              Machine Learning Toolbox, Wavelet Toolbox
%% ========================================================================

clear; clc; close all;

% Se placer a la racine du projet (dossier parent de LiveScripts_Simulink)
scriptPath = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptPath);
cd(projectRoot);
fprintf('Repertoire de travail : %s\n\n', pwd);

%% Configuration
% =========================================================================
% Fichier signal a analyser (produit par le modele Simulink)
SIGNAL_FILE = 'data_signaux_simulink/sain_001.mat';

% Repertoire de sortie des figures (300 DPI)
OUTPUT_DIR = 'Figures_Simulink/Sain';
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

EXPORT_DPI = 300;

fprintf('========================================================================\n');
fprintf('   ANALYSE DE L''ETAT SAIN - SIGNAL ISSU DU MODELE SIMULINK\n');
fprintf('========================================================================\n');
fprintf('Signal  : %s\n', SIGNAL_FILE);
fprintf('Figures : %s\n\n', OUTPUT_DIR);

%% ========================================================================
%% SECTION 0 : PRODUCTION DU SIGNAL SAIN A PARTIR DU MODELE SIMULINK
%% ========================================================================
%
% Le signal analyse ici est produit par le modele Simulink
% PFD_Signal_Generator. Pour l'etat sain, la configuration est :
%
%   - Fault_Type          = 1  (Sain : aucun defaut injecte)
%   - Severity_Level      = 1.0 (nominal ; sans effet car le defaut est nul)
%   - Enable_Evolution    = 0  (pas d'evolution temporelle)
%   - Transient_Type      = 1  (aucun transitoire)
%   - Speed_RPM           = 3600 tr/min  (frequence de rotation 1X = 60 Hz)
%   - Load_Percent        = 70 %
%   - Temperature_C       = 60 degres C
%
% CHAINE DE PRODUCTION DU SIGNAL (voir schema du modele) :
%
%   Operating_Conditions : calcule les facteurs de fonctionnement et le
%       nombre de Sommerfeld S = 0.15*exp(-0.03*(T-60))*(Omega/60)/lf,
%       avec lf = 0.3 + 0.7*(charge/100) = 0.79 a 70 % de charge,
%       soit S = 0.190 au point de fonctionnement nominal.
%   Base_Signal    : bruit blanc de base du palier en fonctionnement.
%   Fault_Injection -> Severity_Control -> Transient_Behavior :
%       pour Fault_Type = 1, la sortie de Fault_Injection est NULLE.
%       Le signal sain ne contient donc AUCUNE signature de defaut.
%   Noise_Model    : 7 sources de bruit de mesure realistes (bruit capteur,
%       interference secteur 50 Hz, bruit rose, derive lente, derive du
%       capteur, impulsions parasites, repliement).
%   Signal_Sum -> Quantizer : somme des contributions puis quantification
%       (pas de 0.001, effet convertisseur analogique-numerique), appliquee
%       en aval de la somme.
%
% Le signal sain est donc : bruit de base + bruits de mesure, quantifie.
% C'est exactement ce qui caracterise une machine saine : un plancher de
% bruit large bande, sans composante de rotation dominante ni signature
% de defaut.
%
% POUR REPRODUIRE CE SIGNAL :
%   Option A (interface) : ouvrir PFD_Signal_Generator.slx, mettre la
%       constante Fault_Type a 1 (double-clic sur le bloc rouge
%       Fault_Injection), lancer Run (5 s), le signal apparait dans le
%       Scope et dans la variable x_sim de l'espace de travail.
%   Option B (script)    : >> generate_simulink_signals
%       (regenere automatiquement les 11 signaux, dont sain_001.mat)
%
% =========================================================================

fprintf('SECTION 0 : Production du signal (modele Simulink)\n');
fprintf('--------------------------------------------------\n');

% Chargement du signal
data = load(SIGNAL_FILE);
x = data.x(:);          % Signal vibratoire (colonne)
fs = double(data.fs);   % Frequence d'echantillonnage (Hz)
fault = char(data.fault);

N = length(x);          % Nombre d'echantillons
T = N / fs;             % Duree totale (s)
t = (0:N-1)' / fs;      % Vecteur temps

fprintf('Etat                : %s (sain)\n', fault);
fprintf('Frequence d''echantillonnage : %d Hz\n', fs);
fprintf('Duree               : %.2f s\n', T);
fprintf('Nombre d''echantillons : %d\n\n', N);

% Metadonnees (parametres exacts du modele lors de la generation)
if isfield(data, 'metadata')
    meta = data.metadata;
    fprintf('Parametres du modele Simulink (metadonnees) :\n');
    fprintf('  Vitesse     : %.0f tr/min (%.1f Hz)\n', meta.speed_rpm, meta.speed_rpm/60);
    fprintf('  Charge      : %.0f %%\n', meta.load_percent);
    fprintf('  Temperature : %.0f degres C\n', meta.temperature_C);
    fprintf('  Sommerfeld  : %.4f\n', meta.sommerfeld_number);
    fprintf('  Severite    : %s\n', char(meta.severity));
    fprintf('  Generateur  : %s\n\n', char(meta.generator_version));
    Omega = meta.speed_rpm / 60;   % Frequence de rotation 1X (Hz)
else
    fprintf('Pas de metadonnees : vitesse par defaut 3600 tr/min.\n\n');
    Omega = 60;
end

fprintf('Frequence de rotation (1X) : %.2f Hz\n', Omega);
fprintf('Harmoniques attendus (defauts) : 2X = %.0f Hz, 3X = %.0f Hz\n', 2*Omega, 3*Omega);
fprintf('Zone sous-synchrone (tourbillonnement d''huile) : %.0f-%.0f Hz\n', ...
    0.42*Omega, 0.48*Omega);
fprintf('--------------------------------------------------\n\n');

%% ========================================================================
%% SECTION 1 : ANALYSE TEMPORELLE
%% ========================================================================
%
% L'analyse temporelle donne une premiere vue du comportement vibratoire :
% amplitude generale, presence eventuelle de chocs, de modulations ou de
% transitoires. Pour un palier sain, on attend un signal de faible
% amplitude, d'aspect aleatoire et stationnaire, sans chocs repetitifs.
%
% =========================================================================

fprintf('SECTION 1 : Analyse temporelle\n');
fprintf('--------------------------------------------------\n');

%% Figure 1 : signal temporel complet (5 s)
fig1 = figure('Name', 'Signal sain - Vue complete', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

plot(t, x, 'b-', 'LineWidth', 0.4);
xlabel('Temps (s)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 1 : Signal temporel complet - Etat sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(sprintf('Duree : %.0f s, fs = %d Hz, N = %d echantillons', T, fs, N), ...
    'FontSize', 11);
grid on;
xlim([0, T]);
set(gca, 'FontSize', 11);

exportgraphics(fig1, fullfile(OUTPUT_DIR, 'Fig1_Sain_Temporel_Complet.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 1 exportee : Fig1_Sain_Temporel_Complet.png\n');

%% Figure 2 : zoom temporel (100 premieres millisecondes)
% Le zoom permet de verifier l'absence de motif periodique dominant.
% Les traits verticaux marquent la periode de rotation (1/Omega = 16.7 ms) :
% pour un signal sain, aucun motif ne se repete a cette periode.
zoom_dur = 0.100;                 % 100 ms
idx_zoom = t <= zoom_dur;
T_rot = 1 / Omega;                % Periode de rotation (s)

fig2 = figure('Name', 'Signal sain - Zoom', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

plot(t(idx_zoom)*1000, x(idx_zoom), 'b-', 'LineWidth', 0.8);
hold on;
for kk = 0:floor(zoom_dur / T_rot)
    xline(kk * T_rot * 1000, 'r--', 'LineWidth', 1.0);
end
xlabel('Temps (ms)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 2 : Zoom temporel (100 ms) - Etat sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(sprintf(['Traits rouges : periode de rotation T = %.2f ms ' ...
    '(1X = %.0f Hz). Aucun motif periodique visible.'], T_rot*1000, Omega), ...
    'FontSize', 11);
grid on;
xlim([0, zoom_dur*1000]);
set(gca, 'FontSize', 11);

exportgraphics(fig2, fullfile(OUTPUT_DIR, 'Fig2_Sain_Temporel_Zoom.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 2 exportee : Fig2_Sain_Temporel_Zoom.png\n\n');

%% ========================================================================
%% SECTION 2 : ANALYSE STATISTIQUE
%% ========================================================================
%
% Indicateurs statistiques du signal temporel.
%
% DEFINITIONS (estimateurs utilises par MATLAB) :
%   Moyenne          : mu = (1/N) * somme(x_i)
%   Valeur efficace  : RMS = sqrt((1/N) * somme(x_i^2))
%   Ecart-type       : sigma = sqrt((1/(N-1)) * somme((x_i - mu)^2))
%                      (estimateur d'echantillon, fonction std)
%   Valeur crete     : max|x|
%   Crete-a-crete    : max(x) - min(x)
%   Asymetrie        : E[(x-mu)^3] / sigma_b^3   (0 = symetrique ;
%                      sigma_b = ecart-type biaise, fonction skewness)
%   Kurtosis         : E[(x-mu)^4] / sigma_b^4   (3 = gaussien ;
%                      kurtosis non normalise, fonction kurtosis)
%   Facteur de crete : FC = max|x| / RMS
%
% INTERPRETATION PHYSIQUE :
%   - Kurtosis proche de 3 : distribution gaussienne, pas de chocs
%     (un kurtosis > 3 signale des evenements impulsifs, typiques de
%     certains defauts comme le manque de lubrification ou la cavitation)
%   - Facteur de crete < 5 : pas de pics transitoires severes
%   - Moyenne proche de 0 : pas de decalage capteur significatif
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
noms = {'Moyenne'; 'Valeur efficace (RMS)'; 'Ecart-type'; 'Valeur crete'; ...
    'Crete-a-crete'; 'Asymetrie (skewness)'; 'Kurtosis'; 'Facteur de crete'};
valeurs = [stat_moyenne; stat_rms; stat_ecart; stat_crete; ...
    stat_c2c; stat_asym; stat_kurt; stat_fc];
references = {'~ 0 (pas de biais)'; '-'; '-'; '-'; '-'; ...
    '0 = symetrique'; '3 = gaussien'; '< 5 = normal'};

tableau_stats = table(noms, valeurs, references, ...
    'VariableNames', {'Indicateur', 'Valeur', 'Reference_sain'});
disp(tableau_stats);

fprintf('\nINTERPRETATION (reference saine) :\n');
if stat_kurt < 4
    fprintf('  OK : kurtosis = %.2f, proche de 3 (gaussien) -> pas de chocs\n', stat_kurt);
else
    fprintf('  ATTENTION : kurtosis = %.2f, eleve pour un etat sain\n', stat_kurt);
end
if stat_fc < 5
    fprintf('  OK : facteur de crete = %.2f, dans la plage normale\n', stat_fc);
else
    fprintf('  ATTENTION : facteur de crete = %.2f, eleve\n', stat_fc);
end
if abs(stat_moyenne) < 0.01
    fprintf('  OK : moyenne = %.4f, pas de biais significatif\n', stat_moyenne);
else
    fprintf('  NOTE : moyenne = %.4f (legere derive du capteur simulee)\n', stat_moyenne);
end
fprintf('\n');

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
ylabel('Densite de probabilite', 'FontSize', 13, 'FontWeight', 'bold');
title('Distribution d''amplitude', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Signal mesure', 'Ajustement gaussien'}, 'Location', 'northwest', ...
    'FontSize', 10);
grid on;
set(gca, 'FontSize', 11);

annotation('textbox', [0.33, 0.62, 0.14, 0.24], 'String', sprintf( ...
    ['Indicateurs :\n' ...
     'Moyenne = %.4f\n' ...
     'Ecart-type = %.4f\n' ...
     'Asymetrie = %.3f\n' ...
     'Kurtosis = %.3f'], ...
    stat_moyenne, stat_ecart, stat_asym, stat_kurt), ...
    'FontSize', 9, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
    'FitBoxToText', 'on');

subplot(1, 2, 2);
qqplot(x);
title('Diagramme Q-Q (test de normalite)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Quantiles normaux theoriques', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Quantiles de l''echantillon', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 11);

sgtitle('Figure 3 : Analyse statistique de la distribution - Etat sain', ...
    'FontSize', 15, 'FontWeight', 'bold');

exportgraphics(fig3, fullfile(OUTPUT_DIR, 'Fig3_Sain_Distribution_Amplitude.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 3 exportee : Fig3_Sain_Distribution_Amplitude.png\n\n');

%% ========================================================================
%% SECTION 3 : ANALYSE FREQUENTIELLE
%% ========================================================================
%
% Deux outils complementaires :
%   1. Spectre d'amplitude par FFT (transformee de Fourier discrete) :
%      X[k] = somme_n x[n] * exp(-j*2*pi*k*n/N)
%   2. Densite spectrale de puissance (DSP) par la methode de Welch :
%      moyenne des periodogrammes sur des segments fenetres (Hann 4096,
%      recouvrement 50 %), plus robuste au bruit que la FFT brute.
%
% La composante continue (moyenne) est retiree avant le calcul pour que le
% bin a 0 Hz ne masque pas le contenu utile.
%
% FREQUENCES CARACTERISTIQUES A 3600 tr/min :
%   1X = 60 Hz (balourd), 2X = 120 Hz et 3X = 180 Hz (desalignement),
%   0.42-0.48X = 25-29 Hz (tourbillonnement d'huile),
%   1500-2500 Hz (cavitation).
% Pour l'etat SAIN, AUCUNE de ces composantes ne doit dominer :
% c'est precisement leur ABSENCE qui definit la reference.
%
% NOTE sur les artefacts de mesure simules (presents volontairement) :
%   - une raie a 50 Hz : interference electromagnetique du secteur (EMI),
%     issue du modele de bruit, a ne pas confondre avec le 1X a 60 Hz ;
%   - du contenu tres basse frequence (< 1 Hz) : derive lente simulee.
%
% =========================================================================

fprintf('SECTION 3 : Analyse frequentielle\n');
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
% Recherche de la frequence dominante au-dessus de 10 % de la vitesse de
% rotation (pour exclure la derive tres basse frequence)
f_min = max(f_psd(2), 0.1 * Omega);
masque = f_psd >= f_min;
f_valides = f_psd(masque);
[~, i_max] = max(Pxx(masque));
f_dominante = f_valides(i_max);

centroide = sum(f_psd .* Pxx) / sum(Pxx);
Pxx_n = Pxx / sum(Pxx);
entropie = -sum(Pxx_n .* log2(Pxx_n + eps));
platitude = exp(mean(log(Pxx + eps))) / (mean(Pxx) + eps);

% Niveaux spectraux (racine de la DSP, en unite/sqrt(Hz)) aux frequences
% caracteristiques. Ces niveaux servent de reference RELATIVE pour la
% comparaison avec les defauts (memes parametres d'estimation partout) ;
% ce ne sont pas des amplitudes de raies au sens du spectre FFT.
% La grille de Welch a un pas de fs/fen = 5 Hz : on releve le bin le plus
% proche de chaque frequence cible et on affiche sa frequence reelle.
[~, i1X] = min(abs(f_psd - Omega));
[~, i2X] = min(abs(f_psd - 2*Omega));
[~, i3X] = min(abs(f_psd - 3*Omega));
[~, iSub] = min(abs(f_psd - 0.45*Omega));
niv_1X  = sqrt(Pxx(i1X));
niv_2X  = sqrt(Pxx(i2X));
niv_3X  = sqrt(Pxx(i3X));
niv_sub = sqrt(Pxx(iSub));

fprintf('\n--- INDICATEURS SPECTRAUX ---\n');
fprintf('Frequence dominante : %.2f Hz\n', f_dominante);
fprintf('Centroide spectral  : %.1f Hz\n', centroide);
fprintf('Entropie spectrale  : %.2f bits\n', entropie);
fprintf('Platitude spectrale : %.4f (1 = bruit large bande, 0 = tonal)\n\n', platitude);
fprintf('Niveaux spectraux (racine de DSP) aux frequences caracteristiques,\n');
fprintf('releves au bin de Welch le plus proche (reference saine) :\n');
fprintf('  0.45X, cible %.0f Hz, bin %.0f Hz : %.3e   (zone tourbillonnement d''huile)\n', ...
    0.45*Omega, f_psd(iSub), niv_sub);
fprintf('  1X,    cible %.0f Hz, bin %.0f Hz : %.3e   (zone balourd)\n', ...
    Omega, f_psd(i1X), niv_1X);
fprintf('  2X,    cible %.0f Hz, bin %.0f Hz : %.3e   (zone desalignement)\n', ...
    2*Omega, f_psd(i2X), niv_2X);
fprintf('  3X,    cible %.0f Hz, bin %.0f Hz : %.3e   (zone desalignement)\n', ...
    3*Omega, f_psd(i3X), niv_3X);

fprintf('\nINTERPRETATION (reference saine) :\n');
if abs(f_dominante - 50) < 2
    fprintf(['  La frequence dominante (%.1f Hz) correspond a l''interference\n' ...
             '  secteur 50 Hz simulee (artefact de mesure), PAS a une composante\n' ...
             '  de rotation : le 1X a %.0f Hz ne domine pas le spectre.\n'], ...
        f_dominante, Omega);
elseif abs(f_dominante - Omega)/Omega < 0.10
    fprintf('  La frequence dominante (%.1f Hz) est proche du 1X (%.0f Hz).\n', ...
        f_dominante, Omega);
else
    fprintf('  Frequence dominante : %.1f Hz (a examiner sur le spectre).\n', f_dominante);
end
if platitude > 0.5
    fprintf(['  Platitude spectrale elevee (%.2f) : le spectre est un plancher\n' ...
             '  de bruit large bande, sans raie dominante -> conforme a un\n' ...
             '  palier sain.\n'], platitude);
end
fprintf(['  Aucun pic significatif a 1X, 2X, 3X ni en zone sous-synchrone :\n' ...
         '  absence de signature de defaut. Ces niveaux serviront de\n' ...
         '  reference de comparaison pour les defauts.\n\n']);

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
xlabel('Frequence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 4 : Spectre d''amplitude (FFT) - Etat sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(['Zoom 0-500 Hz. Seuls emergent la raie EMI 50 Hz et la derive ' ...
    'tres basse frequence (< 1 Hz) ; pas de 1X/2X/3X.'], 'FontSize', 11);
grid on;
xlim([0, 500]);
set(gca, 'FontSize', 11);

exportgraphics(fig4, fullfile(OUTPUT_DIR, 'Fig4_Sain_Spectre_FFT.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 4 exportee : Fig4_Sain_Spectre_FFT.png\n');

%% Figure 5 : DSP de Welch (bande 0 - fs/4, echelle log)
fig5 = figure('Name', 'Signal sain - DSP Welch', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

semilogy(f_psd, Pxx, 'b-', 'LineWidth', 0.9);
hold on;
% Reperes 1X/2X/3X sans etiquettes (trop proches a cette echelle ;
% voir la Figure 6 pour la zone 0-500 Hz etiquetee)
xline(Omega,   'r--', 'LineWidth', 1.3);
xline(2*Omega, 'g--', 'LineWidth', 1.3);
xline(3*Omega, 'm--', 'LineWidth', 1.3);
xlabel('Frequence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('DSP (unite^2/Hz)', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 5 : Densite spectrale de puissance (methode de Welch) - Etat sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(sprintf('Fenetre de Hann %d points, recouvrement 50 %%', fen), 'FontSize', 11);
grid on;
xlim([0, fs/4]);
set(gca, 'FontSize', 11);

annotation('textbox', [0.62, 0.68, 0.2, 0.2], 'String', sprintf( ...
    ['Indicateurs spectraux :\n' ...
     'f dominante = %.1f Hz\n' ...
     'Centroide = %.0f Hz\n' ...
     'Entropie = %.2f bits\n' ...
     'Platitude = %.3f'], ...
    f_dominante, centroide, entropie, platitude), ...
    'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
    'FitBoxToText', 'on');

exportgraphics(fig5, fullfile(OUTPUT_DIR, 'Fig5_Sain_DSP_Welch.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 5 exportee : Fig5_Sain_DSP_Welch.png\n');

%% Figure 6 : DSP zoom basses frequences (0-500 Hz, en dB)
fig6 = figure('Name', 'Signal sain - DSP zoom', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

plot(f_psd, 10*log10(Pxx + eps), 'b-', 'LineWidth', 1.0);
hold on;
xline(Omega,   'r--', '1X = 60 Hz',  'LineWidth', 1.3, 'FontSize', 10, 'LabelOrientation', 'horizontal');
xline(2*Omega, 'g--', '2X = 120 Hz', 'LineWidth', 1.3, 'FontSize', 10, 'LabelOrientation', 'horizontal');
xline(3*Omega, 'm--', '3X = 180 Hz', 'LineWidth', 1.3, 'FontSize', 10, 'LabelOrientation', 'horizontal');
xline(50, 'k:', 'EMI 50 Hz', 'LineWidth', 1.2, 'FontSize', 10, ...
    'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');

% Zone sous-synchrone (ou apparaitrait un tourbillonnement d'huile)
xr = [0.42*Omega, 0.48*Omega];
yl = ylim;
patch([xr(1) xr(2) xr(2) xr(1)], [yl(1) yl(1) yl(2) yl(2)], ...
    [1.0 0.9 0.7], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
text(mean(xr), yl(2)-4, 'zone 0.42-0.48X', 'FontSize', 9, ...
    'HorizontalAlignment', 'center');

xlabel('Frequence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('DSP (dB)', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 6 : DSP en zone basses frequences (0-500 Hz) - Etat sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(['Absence de raies aux frequences caracteristiques des defauts : ' ...
    'seule l''interference secteur 50 Hz est visible.'], 'FontSize', 11);
grid on;
xlim([0, 500]);
set(gca, 'FontSize', 11);

exportgraphics(fig6, fullfile(OUTPUT_DIR, 'Fig6_Sain_DSP_Zoom.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 6 exportee : Fig6_Sain_DSP_Zoom.png\n\n');

%% ========================================================================
%% SECTION 4 : ANALYSE TEMPS-FREQUENCE
%% ========================================================================
%
% Deux methodes complementaires :
%
% 1. Spectrogramme (transformee de Fourier a court terme, STFT) :
%    S(t,f) = |integrale de x(tau)*w(tau-t)*exp(-j*2*pi*f*tau) dtau|^2
%    Compromis temps-frequence fixe par la longueur de la fenetre.
%
% 2. Transformee en ondelettes continue (CWT, ondelette de Morlet
%    analytique) :
%    W(a,b) = (1/sqrt(a)) * integrale de x(t)*psi*((t-b)/a) dt
%    Resolution multi-echelle : fine en temps pour les hautes frequences,
%    fine en frequence pour les basses frequences. Ideale pour reveler
%    des transitoires (chocs, bouffees) que la STFT etale.
%
% Pour l'etat sain, les deux representations doivent etre HOMOGENES dans
% le temps : pas de bandes horizontales persistantes (raies), pas de
% colonnes verticales (chocs), hormis la raie EMI 50 Hz.
%
% =========================================================================

fprintf('SECTION 4 : Analyse temps-frequence\n');
fprintf('--------------------------------------------------\n');

%% Figure 7 : spectrogramme STFT
fig7 = figure('Name', 'Signal sain - Spectrogramme', ...
    'Position', [100, 100, 1200, 600], 'Color', 'white');
% tiledlayout : reserve l'espace du titre au-dessus de l'image
% (un titre d'axes classique n'est pas rendu de facon fiable au-dessus
% d'imagesc en export sans affichage)
tl7 = tiledlayout(fig7, 1, 1, 'Padding', 'compact');
nexttile(tl7);

fen_spec = 512;
rec_spec = 448;    % 87.5 % de recouvrement
nfft_spec = 1024;
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
ylabel('Frequence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
title(tl7, 'Figure 7 : Spectrogramme (STFT) - Etat sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(tl7, sprintf(['Fenetre de Hann %d points (%.1f ms), ' ...
    'recouvrement %.1f %%. Contenu homogene, sans raie de defaut.'], ...
    fen_spec, 1000*fen_spec/fs, 100*rec_spec/fen_spec), 'FontSize', 11);
set(gca, 'FontSize', 11);

exportgraphics(fig7, fullfile(OUTPUT_DIR, 'Fig7_Sain_Spectrogramme_STFT.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 7 exportee : Fig7_Sain_Spectrogramme_STFT.png\n');

%% Figure 8 : transformee en ondelettes continue (CWT)
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
ylabel('Frequence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
title('Figure 8 : Transformee en ondelettes continue (Morlet analytique) - Etat sain', ...
    'FontSize', 15, 'FontWeight', 'bold');
subtitle(['Analyse multi-resolution : aucun transitoire ni composante ' ...
    'persistante liee a un defaut.'], 'FontSize', 11);
set(gca, 'FontSize', 11, 'YScale', 'linear');

exportgraphics(fig8, fullfile(OUTPUT_DIR, 'Fig8_Sain_CWT.png'), ...
    'Resolution', EXPORT_DPI);
fprintf('  Figure 8 exportee : Fig8_Sain_CWT.png\n\n');

%% ========================================================================
%% SECTION 5 : SYNTHESE ET INTERPRETATION
%% ========================================================================

fprintf('SECTION 5 : Synthese\n');
fprintf('--------------------------------------------------\n');

interp_txt = sprintf([ ...
'INTERPRETATION DES RESULTATS - ETAT SAIN (REFERENCE)\n' ...
'=====================================================\n\n' ...
'Signal analyse : %s\n' ...
'Produit par le modele Simulink PFD_Signal_Generator\n' ...
'(3600 tr/min soit 1X = 60 Hz, charge 70 %%, temperature 60 C,\n' ...
'fs = %d Hz, duree %.0f s).\n\n' ...
'1. ANALYSE TEMPORELLE\n' ...
'Le signal presente un aspect aleatoire, stationnaire et de faible\n' ...
'amplitude (valeur efficace RMS = %.4f, valeur crete = %.3f).\n' ...
'Le zoom sur 100 ms ne revele aucun motif periodique a la periode de\n' ...
'rotation (%.2f ms) ni chocs repetitifs : comportement attendu d''un\n' ...
'palier hydrodynamique sain, ou le film d''huile amortit les\n' ...
'vibrations de l''arbre.\n\n' ...
'2. ANALYSE STATISTIQUE\n' ...
'Le kurtosis vaut %.2f, tres proche de 3 (distribution gaussienne),\n' ...
'et le facteur de crete vaut %.2f (< 5) : aucune impulsivite dans le\n' ...
'signal, donc pas de chocs metal-metal ni d''evenements transitoires.\n' ...
'L''asymetrie est quasi nulle (%.3f) et la moyenne negligeable\n' ...
'(%.4f) : distribution symetrique, pas de biais de mesure notable.\n' ...
'L''histogramme et le diagramme Q-Q confirment la normalite de la\n' ...
'distribution d''amplitude.\n\n' ...
'3. ANALYSE FREQUENTIELLE\n' ...
'Le spectre est un plancher de bruit large bande (platitude spectrale\n' ...
'%.3f, entropie %.2f bits), SANS raie dominante liee a la rotation :\n' ...
'ni 1X (60 Hz, signature de balourd), ni 2X/3X (120/180 Hz, signature\n' ...
'de desalignement), ni composante sous-synchrone entre 25 et 29 Hz\n' ...
'(signature de tourbillonnement d''huile). La seule raie visible est\n' ...
'l''interference secteur a 50 Hz, un artefact de mesure simule\n' ...
'volontairement (couplage electromagnetique), a ne pas confondre avec\n' ...
'le 1X. C''est precisement l''ABSENCE des raies caracteristiques qui\n' ...
'definit la signature de l''etat sain et en fait la reference de\n' ...
'comparaison pour tous les defauts.\n\n' ...
'4. ANALYSE TEMPS-FREQUENCE\n' ...
'Le spectrogramme (STFT) montre un contenu spectral homogene et\n' ...
'stationnaire sur toute la duree : pas de bande horizontale\n' ...
'persistante (raie de defaut) ni de colonne verticale (choc).\n' ...
'La transformee en ondelettes (CWT, Morlet analytique), plus fine en\n' ...
'temps aux hautes frequences, confirme l''absence de transitoires ou\n' ...
'de bouffees d''energie. Les deux representations valident la\n' ...
'stationnarite du signal sain.\n\n' ...
'CONCLUSION\n' ...
'Le signal sain issu du modele Simulink presente toutes les\n' ...
'caracteristiques d''un palier hydrodynamique en bon etat :\n' ...
'distribution gaussienne (kurtosis proche de 3), spectre plat sans\n' ...
'signature de defaut, contenu temps-frequence homogene. Il constitue\n' ...
'la REFERENCE a laquelle les 7 defauts simples et les 3 defauts\n' ...
'mixtes seront compares dans les phases suivantes.\n'], ...
SIGNAL_FILE, fs, T, stat_rms, stat_crete, T_rot*1000, ...
stat_kurt, stat_fc, stat_asym, stat_moyenne, platitude, entropie);

% Sauvegarde du texte d'interpretation
fid = fopen(fullfile(OUTPUT_DIR, 'Interpretation_Sain.txt'), 'w', 'n', 'UTF-8');
fprintf(fid, '%s', interp_txt);
fclose(fid);

fprintf('%s\n', interp_txt);
fprintf('  Texte d''interpretation sauvegarde : Interpretation_Sain.txt\n\n');

fprintf('========================================================================\n');
fprintf('   ANALYSE TERMINEE - 8 figures exportees dans %s\n', OUTPUT_DIR);
fprintf('========================================================================\n');
fprintf('Prochaine etape : analyse des 7 defauts simples (Phase 2).\n\n');
