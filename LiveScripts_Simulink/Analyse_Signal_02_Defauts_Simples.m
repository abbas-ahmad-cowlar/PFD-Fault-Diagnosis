%% ANALYSE DU SIGNAL 02 : LES 7 DÉFAUTS SIMPLES
%%
%% Analyse vibratoire multi-domaine d'un système palier hydrodynamique
%% Signaux produits par le modèle Simulink PFD_Signal_Generator (v3.1)
%%
%% Ce script analyse les 7 défauts simples avec les mêmes quatre familles
%% d'analyse que l'état sain (Phase 1), et compare chaque défaut à la
%% référence saine : désalignement, déséquilibre, jeu, lubrification,
%% cavitation, usure, tourbillonnement d'huile.
%%
%% CONTENU PAR DÉFAUT :
%%   1. Analyse temporelle (vue complète + zoom)
%%   2. Analyse statistique (8 indicateurs + comparaison au sain)
%%   3. Analyse fréquentielle (FFT + DSP de Welch + zoom annoté)
%%   4. Analyse temps-fréquence (spectrogramme STFT + ondelettes CWT)
%%   5. Vérification des indicateurs attendus (règles documentées) et
%%      texte d'interprétation
%% Puis un tableau comparatif global (sain + 7 défauts) et une figure
%% comparative.
%%
%% Version : 1.0 (Phase 2 - Défauts simples) - 2026-08
%% Compatible : MATLAB R2024b, Signal Processing Toolbox, Statistics and
%%              Machine Learning Toolbox, Wavelet Toolbox
%% ========================================================================

clear; clc; close all;

% Désactive la barre d'outils des axes pour tous les exports (elle peut
% sinon apparaître dans les PNG exportés en mode sans affichage)
set(groot, 'DefaultAxesCreateFcn', @(ax, ~) set(ax.Toolbar, 'Visible', 'off'));
nettoyage = onCleanup(@() set(groot, 'DefaultAxesCreateFcn', ''));

scriptPath = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptPath);
cd(projectRoot);
fprintf('Répertoire de travail : %s\n\n', pwd);

%% Configuration
% =========================================================================
DATA_DIR   = 'data_signaux_simulink';
OUT_ROOT   = 'Figures_Simulink/Defauts';
EXPORT_DPI = 300;

if ~exist(OUT_ROOT, 'dir'), mkdir(OUT_ROOT); end

% Les 7 défauts simples : code fichier, nom affiché (accentué), bande
% d'affichage fréquentielle [Hz], description courte de l'indicateur
% attendu (issue de la physique du modèle), fenêtre du zoom temporel [s]
% (choisie pour montrer un événement caractéristique du défaut), et note
% de zoom pour le sous-titre.
DEFAUTS = {
% code            nom affiché                  bande     indicateur attendu                                                          zoom [s]      note de zoom
 'desalignement', 'Désalignement',             [0 500],  'harmoniques 2X (120 Hz) et 3X (180 Hz)',                                   [0 0.100],    ''
 'desequilibre',  'Déséquilibre',              [0 500],  'composante 1X dominante (60 Hz)',                                          [0 0.100],    ''
 'jeu',           'Jeu',                       [0 500],  'composante sous-synchrone ~0.43X + 1X + 2X',                               [0 0.100],    ''
 'lubrification', 'Lubrification',             [0 500],  'adhérence-glissement très basse fréquence (~3.5 Hz) + impacts métal-métal', [0.70 1.00],  'Fenêtre 0.70-1.00 s : impact métal-métal modélisé à t = 0.8 s + ondulation d''adhérence-glissement.'
 'cavitation',    'Cavitation',                [0 3000], 'bouffées haute fréquence 1500-2500 Hz',                                    [0.45 0.60],  'Fenêtre 0.45-0.60 s : première bouffée de cavitation modélisée à t = 0.5 s.'
 'usure',         'Usure',                     [0 3000], 'bruit blanc large bande (4 bandes disjointes 500-2500 Hz) + harmoniques modulés', [0 0.100],   ''
 'oilwhirl',      'Tourbillonnement d''huile', [0 500],  'composante sous-synchrone ~0.45X (~27 Hz) dominante',                      [0 0.100],    ''
};

fen = 4096;              % fenêtre de Welch (grille 5 Hz)
SEUIL_DB = 3;            % seuil de signification (au moins 3 dB), identique Phase 1
% Tolérance indicative propre à cette étude pour l'étendue (max - min) des
% quatre bandes disjointes du test d'élévation "plate" (règle usure) : le
% bruit blanc du modèle donne une étendue mesurée d'environ 0.3 dB ; la
% cavitation, dont l'énergie est concentrée en 1500-2500 Hz, la dépasse
% largement.
TOL_PLAT_DB = 1.5;

fprintf('========================================================================\n');
fprintf('   ANALYSE DES 7 DÉFAUTS SIMPLES - SIGNAUX ISSUS DU MODÈLE SIMULINK\n');
fprintf('========================================================================\n\n');

%% ========================================================================
%% SECTION 0 : RÉFÉRENCE SAINE
%% ========================================================================
% La référence saine (Phase 1) est recalculée ici avec exactement les
% mêmes paramètres d'estimation, pour que toutes les comparaisons soient
% à méthode identique.

fprintf('SECTION 0 : Calcul de la référence saine...\n');
ref = analyser_signal(fullfile(DATA_DIR, 'sain_001.mat'), fen);
fprintf('  Sain : RMS = %.4f, kurtosis = %.2f, plancher médian (10-300 Hz) = %.3e\n\n', ...
    ref.rms, ref.kurt, ref.plancher);

% Accumulateur du tableau comparatif (le sain d'abord). dBF = écart au
% sain au bin de Welch le plus proche de 3.5 Hz (indicateur principal de
% la lubrification) ; dB4 = écarts au sain dans les 4 bandes disjointes.
comp = struct('nom', {}, 'rms', {}, 'kurt', {}, 'fc', {}, ...
    'ex1X', {}, 'ex2X', {}, 'ex3X', {}, 'exSub', {}, 'dHF', {}, 'dMid', {}, ...
    'dBF', {}, 'dB4', {});
comp(1) = struct('nom', 'Sain', 'rms', ref.rms, 'kurt', ref.kurt, ...
    'fc', ref.fc, 'ex1X', ref.ex1X, 'ex2X', ref.ex2X, 'ex3X', ref.ex3X, ...
    'exSub', ref.exSub, 'dHF', 0, 'dMid', 0, 'dBF', 0, 'dB4', zeros(1,4));

% Bin de Welch le plus proche de la cible 3.5 Hz (grille de 5 Hz -> bin 5 Hz)
i_bin35 = find(abs(ref.f_psd - 3.5) == min(abs(ref.f_psd - 3.5)), 1);

%% ========================================================================
%% BOUCLE SUR LES 7 DÉFAUTS
%% ========================================================================

nb_observes = 0;
verdicts = cell(size(DEFAUTS, 1), 1);

for kd = 1:size(DEFAUTS, 1)
    code   = DEFAUTS{kd, 1};
    nomAff = DEFAUTS{kd, 2};
    bande  = DEFAUTS{kd, 3};
    signat = DEFAUTS{kd, 4};
    zoomw  = DEFAUTS{kd, 5};
    znote  = DEFAUTS{kd, 6};

    fprintf('========================================================================\n');
    fprintf('DÉFAUT %d/7 : %s\n', kd, upper(nomAff));
    fprintf('  Indicateur attendu : %s\n', signat);
    fprintf('------------------------------------------------------------------------\n');

    outDir = fullfile(OUT_ROOT, code);
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    % ---- Analyse complète du signal ----
    S = analyser_signal(fullfile(DATA_DIR, [code '_001.mat']), fen);
    x = S.x; fs = S.fs; t = S.t; N = S.N; T = S.T; Omega = S.Omega;
    T_rot = 1 / Omega;

    fprintf('  Signal chargé : N = %d, fs = %d Hz, 1X = %.0f Hz\n', N, fs, Omega);

    % ---- Écarts par rapport au sain (mêmes bandes, mêmes méthodes) ----
    dHF  = 10*log10(S.p_hf  / ref.p_hf);    % bande cavitation 1500-2500 Hz
    dMid = 10*log10(S.p_mid / ref.p_mid);   % bande de mesure 500-2000 Hz
    dB4  = 10*log10(S.p_b4 ./ ref.p_b4);    % 4 bandes disjointes (règle usure)
    dBF  = 10*log10(S.Pxx(i_bin35) / ref.Pxx(i_bin35)); % bin 5 Hz vs sain

    %% Figure 1 : signal temporel complet
    fig1 = figure('Name', [nomAff ' - Vue complète'], ...
        'Position', [100, 100, 1200, 500], 'Color', 'white');
    plot(t, x, 'b-', 'LineWidth', 0.4);
    xlabel('Temps (s)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Figure 1 : Signal temporel complet - %s', nomAff), ...
        'FontSize', 15, 'FontWeight', 'bold');
    subtitle(sprintf('Durée : %.0f s, fs = %d Hz. RMS = %.4f (sain : %.4f)', ...
        T, fs, S.rms, ref.rms), 'FontSize', 11);
    grid on; xlim([0, T]); set(gca, 'FontSize', 11);
    exporter_figure(fig1, fullfile(outDir, sprintf('Fig1_%s_Temporel_Complet.png', code)), EXPORT_DPI);

    %% Figure 2 : zoom temporel (fenêtre caractéristique du défaut)
    idxz = t >= zoomw(1) & t <= zoomw(2);
    fig2 = figure('Name', [nomAff ' - Zoom'], ...
        'Position', [100, 100, 1200, 500], 'Color', 'white');
    plot(t(idxz)*1000, x(idxz), 'b-', 'LineWidth', 0.8);
    hold on;
    for kk = ceil(zoomw(1) / T_rot):floor(zoomw(2) / T_rot)
        xline(kk * T_rot * 1000, 'r--', 'LineWidth', 1.0);
    end
    xlabel('Temps (ms)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Figure 2 : Zoom temporel (%.0f-%.0f ms) - %s', ...
        zoomw(1)*1000, zoomw(2)*1000, nomAff), 'FontSize', 15, 'FontWeight', 'bold');
    if isempty(znote)
        subtitle(sprintf('Traits rouges : période de rotation T = %.2f ms (1X = %.0f Hz)', ...
            T_rot*1000, Omega), 'FontSize', 11);
    else
        subtitle(sprintf('%s Traits rouges : période de rotation (%.2f ms).', ...
            znote, T_rot*1000), 'FontSize', 10);
    end
    grid on; xlim([zoomw(1)*1000, zoomw(2)*1000]); set(gca, 'FontSize', 11);
    exporter_figure(fig2, fullfile(outDir, sprintf('Fig2_%s_Temporel_Zoom.png', code)), EXPORT_DPI);

    %% Figure 3 : distribution d'amplitude
    fig3 = figure('Name', [nomAff ' - Distribution'], ...
        'Position', [100, 100, 1200, 550], 'Color', 'white');
    subplot(1, 2, 1);
    histogram(x, 100, 'Normalization', 'pdf', 'FaceColor', [0.3, 0.5, 0.8], ...
        'EdgeColor', 'none', 'FaceAlpha', 0.7);
    hold on;
    xr = linspace(min(x), max(x), 200);
    plot(xr, normpdf(xr, S.moy, S.ecart), 'r-', 'LineWidth', 2.2);
    xlabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Densité de probabilité', 'FontSize', 13, 'FontWeight', 'bold');
    title('Distribution d''amplitude', 'FontSize', 14, 'FontWeight', 'bold');
    legend({'Signal', 'Ajustement gaussien'}, 'Location', 'northwest', 'FontSize', 10);
    grid on; set(gca, 'FontSize', 11);
    subplot(1, 2, 2);
    qqplot(x);
    title('Diagramme Q-Q (comparaison à la loi normale)', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Quantiles normaux théoriques', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Quantiles de l''échantillon', 'FontSize', 13, 'FontWeight', 'bold');
    grid on; set(gca, 'FontSize', 11);
    sgtitle(sprintf('Figure 3 : Analyse statistique de la distribution - %s', nomAff), ...
        'FontSize', 15, 'FontWeight', 'bold');
    exporter_figure(fig3, fullfile(outDir, sprintf('Fig3_%s_Distribution_Amplitude.png', code)), EXPORT_DPI);

    % ---- Tableau statistique (avec référence saine) ----
    noms = {'Moyenne'; 'Valeur efficace (RMS)'; 'Écart-type'; 'Valeur crête'; ...
        'Crête-à-crête'; 'Asymétrie (skewness)'; 'Kurtosis'; 'Facteur de crête'};
    v_def  = [S.moy;  S.rms;  S.ecart;  S.crete;  S.c2c;  S.asym;  S.kurt;  S.fc];
    v_sain = [ref.moy; ref.rms; ref.ecart; ref.crete; ref.c2c; ref.asym; ref.kurt; ref.fc];
    tableau = table(noms, v_def, v_sain, ...
        'VariableNames', {'Indicateur', nomVarValide(code), 'Sain_reference'});
    disp(tableau);
    writetable(tableau, fullfile(outDir, sprintf('Tableau_Statistiques_%s.csv', code)), ...
        'Encoding', 'UTF-8');

    %% Figure 4 : spectre d'amplitude FFT (bande du défaut)
    fig4 = figure('Name', [nomAff ' - Spectre FFT'], ...
        'Position', [100, 100, 1200, 500], 'Color', 'white');
    plot(S.f_fft, S.X_mag, 'b-', 'LineWidth', 0.8);
    hold on;
    tracer_reperes(code, Omega);
    xlabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Figure 4 : Spectre d''amplitude (FFT) - %s', nomAff), ...
        'FontSize', 15, 'FontWeight', 'bold');
    subtitle(sprintf('Zoom %d-%d Hz. Indicateur attendu : %s.', ...
        bande(1), bande(2), signat), 'FontSize', 10);
    grid on; xlim(bande); set(gca, 'FontSize', 11);
    exporter_figure(fig4, fullfile(outDir, sprintf('Fig4_%s_Spectre_FFT.png', code)), EXPORT_DPI);

    %% Figure 5 : DSP de Welch (bande complète, échelle log)
    fig5 = figure('Name', [nomAff ' - DSP Welch'], ...
        'Position', [100, 100, 1200, 500], 'Color', 'white');
    semilogy(S.f_psd, S.Pxx, 'b-', 'LineWidth', 0.9);
    hold on;
    semilogy(ref.f_psd, ref.Pxx, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.7);
    % Raie de repliement simulée (artefact, présent dans tous les états)
    xline(10040, 'k:', 'repliement ~10.04 kHz', 'LineWidth', 1.0, 'FontSize', 9, ...
        'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'top', ...
        'LabelHorizontalAlignment', 'left');
    legend({nomAff, 'Sain (référence)'}, 'Location', 'northeast', 'FontSize', 10);
    xlabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('DSP (unité^2/Hz)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Figure 5 : DSP (Welch) sur la bande complète - %s vs sain', nomAff), ...
        'FontSize', 15, 'FontWeight', 'bold');
    subtitle(sprintf('Fenêtre de Hann %d points, recouvrement 50 %%, bande 0-%d Hz', ...
        fen, fs/2), 'FontSize', 11);
    grid on; xlim([0, fs/2]); set(gca, 'FontSize', 11);
    annotation('textbox', [0.55, 0.62, 0.2, 0.2], 'String', sprintf( ...
        ['Indicateurs spectraux :\n' ...
         'f dominante (>= 6 Hz) = %.1f Hz\n' ...
         'Entropie = %.2f bits\n' ...
         'Platitude = %.3f (sain : %.3f)'], ...
        S.fdom, S.entropie, S.platitude, ref.platitude), ...
        'FontSize', 9, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
        'FitBoxToText', 'on');
    exporter_figure(fig5, fullfile(outDir, sprintf('Fig5_%s_DSP_Welch.png', code)), EXPORT_DPI);

    %% Figure 6 : DSP zoom (bande du défaut, en dB, annotée)
    fig6 = figure('Name', [nomAff ' - DSP zoom'], ...
        'Position', [100, 100, 1200, 500], 'Color', 'white');
    plot(S.f_psd, 10*log10(S.Pxx + eps), 'b-', 'LineWidth', 1.0);
    hold on;
    plot(ref.f_psd, 10*log10(ref.Pxx + eps), '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.7);
    tracer_reperes(code, Omega);
    legend({nomAff, 'Sain (référence)'}, 'Location', 'northeast', 'FontSize', 10);
    xlabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('DSP (dB)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Figure 6 : DSP en zone caractéristique - %s vs sain', nomAff), ...
        'FontSize', 15, 'FontWeight', 'bold');
    subtitle(sprintf('Bande %d-%d Hz. Indicateur attendu : %s.', ...
        bande(1), bande(2), signat), 'FontSize', 10);
    grid on; xlim(bande); set(gca, 'FontSize', 11);
    exporter_figure(fig6, fullfile(outDir, sprintf('Fig6_%s_DSP_Zoom.png', code)), EXPORT_DPI);

    %% Figure 7 : spectrogramme STFT
    fig7 = figure('Name', [nomAff ' - Spectrogramme'], ...
        'Position', [100, 100, 1200, 600], 'Color', 'white');
    tl7 = tiledlayout(fig7, 1, 1, 'Padding', 'compact');
    nexttile(tl7);
    fen_spec = 2048; rec_spec = round(0.875 * fen_spec); nfft_spec = 4096;
    [Sg, F, T_spec] = spectrogram(x, hann(fen_spec), rec_spec, nfft_spec, fs);
    imagesc(T_spec, F, 10*log10(abs(Sg).^2 + eps));
    axis xy; colormap('jet');
    cb = colorbar; cb.Label.String = 'Puissance (dB)'; cb.Label.FontSize = 11;
    ylim(bande);
    xlabel('Temps (s)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
    title(tl7, sprintf('Figure 7 : Spectrogramme (STFT) - %s', nomAff), ...
        'FontSize', 15, 'FontWeight', 'bold');
    subtitle(tl7, sprintf(['Hann %d points (%.0f ms), recouvrement %.1f %%, ' ...
        'affichage 5 Hz, bande %d-%d Hz.'], fen_spec, 1000*fen_spec/fs, ...
        100*rec_spec/fen_spec, bande(1), bande(2)), 'FontSize', 10);
    set(gca, 'FontSize', 11);
    exporter_figure(fig7, fullfile(outDir, sprintf('Fig7_%s_Spectrogramme_STFT.png', code)), EXPORT_DPI);

    %% Figure 8 : transformée en ondelettes continue (CWT)
    fig8 = figure('Name', [nomAff ' - CWT'], ...
        'Position', [100, 100, 1200, 600], 'Color', 'white');
    [cfs, frq] = cwt(x, 'amor', fs);
    surface(t, frq, abs(cfs));
    axis tight; shading interp; view(0, 90); colormap('parula');
    cb = colorbar; cb.Label.String = 'Module'; cb.Label.FontSize = 11;
    ylim(bande);
    xlabel('Temps (s)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Figure 8 : Transformée en ondelettes continue (Morlet analytique) - %s', nomAff), ...
        'FontSize', 15, 'FontWeight', 'bold');
    subtitle(sprintf('Bande %d-%d Hz.', bande(1), bande(2)), 'FontSize', 11);
    set(gca, 'FontSize', 11, 'YScale', 'linear');
    exporter_figure(fig8, fullfile(outDir, sprintf('Fig8_%s_CWT.png', code)), EXPORT_DPI);

    close all;

    %% Vérification des indicateurs attendus + interprétation
    fprintf('\n  --- VÉRIFICATION DES INDICATEURS ATTENDUS (seuil : au moins %d dB) ---\n', SEUIL_DB);
    fprintf('  Excès au-dessus du plancher médian local du défaut :\n');
    fprintf('    zone sous-synchrone : %+.1f dB | 1X : %+.1f dB | 2X : %+.1f dB | 3X : %+.1f dB\n', ...
        S.exSub, S.ex1X, S.ex2X, S.ex3X);
    fprintf('  Écarts par rapport au sain : bin 5 Hz : %+.1f dB | bandes disjointes\n', dBF);
    fprintf('    500-1000 : %+.2f | 1000-1500 : %+.2f | 1500-2000 : %+.2f | 2000-2500 : %+.2f dB\n', ...
        dB4(1), dB4(2), dB4(3), dB4(4));
    fprintf('  Kurtosis : %.2f (sain : %.2f) | RMS : %.4f (sain : %.4f)\n', ...
        S.kurt, ref.kurt, S.rms, ref.rms);

    [verdict, resume, regle] = verifier_signature(code, S, ref, dHF, dBF, dB4, SEUIL_DB, TOL_PLAT_DB);
    if verdict
        fprintf('  INDICATEURS ATTENDUS OBSERVÉS (état simulé connu) : %s\n\n', resume);
        nb_observes = nb_observes + 1;
    else
        fprintf('  ATTENTION : indicateurs attendus NON observés (%s)\n\n', resume);
    end
    verdicts{kd} = struct('nom', nomAff, 'ok', verdict, 'resume', resume);

    % ---- Texte d'interprétation ----
    interp = interpretation_defaut(code, nomAff, S, ref, dHF, dBF, dB4, resume, regle, verdict);
    fid = fopen(fullfile(outDir, sprintf('Interpretation_%s.txt', code)), 'w', 'n', 'UTF-8');
    fprintf(fid, '%s', interp);
    fclose(fid);
    fprintf('  8 figures + tableau CSV + interprétation exportés dans %s\n\n', outDir);

    % ---- Ligne du tableau comparatif ----
    comp(end+1) = struct('nom', nomAff, 'rms', S.rms, 'kurt', S.kurt, ...
        'fc', S.fc, 'ex1X', S.ex1X, 'ex2X', S.ex2X, 'ex3X', S.ex3X, ...
        'exSub', S.exSub, 'dHF', dHF, 'dMid', dMid, 'dBF', dBF, ...
        'dB4', dB4); %#ok<SAGROW>
end

%% ========================================================================
%% TABLEAU COMPARATIF GLOBAL + FIGURE COMPARATIVE
%% ========================================================================

fprintf('========================================================================\n');
fprintf('SYNTHÈSE COMPARATIVE (sain + 7 défauts)\n');
fprintf('------------------------------------------------------------------------\n');

dB4mat = vertcat(comp.dB4);
Tcomp = table({comp.nom}', [comp.rms]', [comp.kurt]', [comp.fc]', ...
    [comp.ex1X]', [comp.ex2X]', [comp.ex3X]', [comp.exSub]', ...
    [comp.dBF]', dB4mat(:,1), dB4mat(:,2), dB4mat(:,3), dB4mat(:,4), ...
    [comp.dHF]', [comp.dMid]', ...
    'VariableNames', {'Etat', 'RMS', 'Kurtosis', 'Facteur_crete', ...
    'Exces_1X_dB', 'Exces_2X_dB', 'Exces_3X_dB', 'Exces_sous_sync_dB', ...
    'Bin5Hz_vs_sain_dB', 'B500_1000_vs_sain_dB', 'B1000_1500_vs_sain_dB', ...
    'B1500_2000_vs_sain_dB', 'B2000_2500_vs_sain_dB', ...
    'Bande_HF_vs_sain_dB', 'Bande_mid_vs_sain_dB'});
disp(Tcomp);
writetable(Tcomp, fullfile(OUT_ROOT, 'Tableau_Comparatif_Simples.csv'), ...
    'Encoding', 'UTF-8');
fprintf('  Tableau comparatif exporté : Tableau_Comparatif_Simples.csv\n');

% Figure comparative : 6 indicateurs comparatifs en barres (chaque défaut
% y trouve son indicateur principal, y compris la lubrification)
figC = figure('Name', 'Comparaison des états', ...
    'Position', [40, 40, 1700, 850], 'Color', 'white');
etats = {comp.nom};
subplot(2, 3, 1);
bar([comp.rms], 'FaceColor', [0.3 0.5 0.8]);
title('Valeur efficace (RMS)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('RMS', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 8); xtickangle(35); grid on;
subplot(2, 3, 2);
bar([comp.kurt], 'FaceColor', [0.85 0.5 0.3]);
hold on; yline(3, 'k--', 'gaussien = 3', 'FontSize', 8);
title('Kurtosis', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Kurtosis', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 8); xtickangle(35); grid on;
subplot(2, 3, 3);
bar([comp.ex1X; comp.ex2X; comp.ex3X]', 'grouped');
hold on; yline(3, 'k--', 'seuil 3 dB', 'FontSize', 8);
legend({'1X', '2X', '3X'}, 'Location', 'northwest', 'FontSize', 8);
title('Excès aux harmoniques 1X, 2X et 3X (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('dB au-dessus du plancher', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 8); xtickangle(35); grid on;
subplot(2, 3, 4);
bar([comp.exSub], 'FaceColor', [0.5 0.7 0.4]);
hold on; yline(3, 'k--', 'seuil 3 dB', 'FontSize', 8);
title('Excès en zone sous-synchrone (bins 25-30 Hz) (dB)', ...
    'FontSize', 12, 'FontWeight', 'bold');
ylabel('dB au-dessus du plancher', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 8); xtickangle(35); grid on;
subplot(2, 3, 5);
bar([comp.dBF], 'FaceColor', [0.6 0.4 0.7]);
hold on; yline(3, 'k--', 'seuil 3 dB', 'FontSize', 8);
title('Écart au sain au bin 5 Hz, cible ~3.5 Hz (dB)', ...
    'FontSize', 12, 'FontWeight', 'bold');
ylabel('dB par rapport au sain', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 8); xtickangle(35); grid on;
subplot(2, 3, 6);
bar(vertcat(comp.dB4), 'grouped');
hold on; yline(3, 'k--', 'seuil 3 dB', 'FontSize', 8);
legend({'500-1000', '1000-1500', '1500-2000', '2000-2500 Hz'}, ...
    'Location', 'northwest', 'FontSize', 7);
title('Écarts au sain des 4 bandes disjointes (dB)', ...
    'FontSize', 12, 'FontWeight', 'bold');
ylabel('dB par rapport au sain', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 8); xtickangle(35); grid on;
sgtitle('Figure C1 : Indicateurs comparatifs - sain et 7 défauts simples', ...
    'FontSize', 15, 'FontWeight', 'bold');
exporter_figure(figC, fullfile(OUT_ROOT, 'FigC1_Comparatif_Indicateurs.png'), EXPORT_DPI);
close all;
fprintf('  Figure comparative exportée : FigC1_Comparatif_Indicateurs.png\n\n');

fprintf('========================================================================\n');
if nb_observes == size(DEFAUTS, 1)
    fprintf('   ANALYSE TERMINÉE - indicateurs attendus observés pour les %d défauts\n', nb_observes);
else
    fprintf('   ANALYSE TERMINÉE AVEC RÉSERVES : indicateurs observés pour %d/%d défauts.\n', ...
        nb_observes, size(DEFAUTS, 1));
    for kv = 1:numel(verdicts)
        if ~verdicts{kv}.ok
            fprintf('   NON OBSERVÉ - %s : %s\n', verdicts{kv}.nom, verdicts{kv}.resume);
        end
    end
end
fprintf('   7 défauts x (8 figures + tableau + interprétation) + tableau et\n');
fprintf('   figure comparatifs dans %s\n', OUT_ROOT);
fprintf('========================================================================\n');
fprintf(['Cette analyse couvre uniquement les sept défauts simples ; les trois\n' ...
    'défauts mixtes ne sont pas inclus et restent à confirmer séparément.\n\n']);

%% ========================================================================
%% FONCTIONS LOCALES
%% ========================================================================

function S = analyser_signal(fichier, fen)
% Charge un signal et calcule tous les indicateurs de l'étude
% (mêmes méthodes et paramètres que l'analyse de l'état sain, Phase 1).
    d = load(fichier);
    x = d.x(:);
    fs = double(d.fs);
    N = length(x);
    S.x = x; S.fs = fs; S.N = N;
    S.T = (N-1) / fs;
    S.t = (0:N-1)' / fs;
    % Paramètres de génération lus dans les métadonnées : les textes de
    % sortie en dérivent, et les champs requis sont EXIGÉS (pas de valeur
    % par défaut silencieuse - cohérent avec la validation canonique
    % ci-dessous).
    champs = {'speed_rpm', 'load_percent', 'temperature_C', 'severity_factor'};
    if ~isfield(d, 'metadata') || ~all(isfield(d.metadata, champs))
        error(['Métadonnées absentes ou incomplètes dans %s : les champs ' ...
            'speed_rpm, load_percent, temperature_C et severity_factor ' ...
            'sont requis (fichiers du livrable v3.1).'], fichier);
    end
    m = d.metadata;
    S.Omega    = double(m.speed_rpm) / 60;
    S.load_pct = double(m.load_percent);
    S.temp_C   = double(m.temperature_C);
    S.sev      = double(m.severity_factor);

    % Statistiques
    S.moy   = mean(x);
    S.rms   = rms(x);
    S.ecart = std(x);
    S.crete = max(abs(x));
    S.c2c   = max(x) - min(x);
    S.asym  = skewness(x);
    S.kurt  = kurtosis(x);
    S.fc    = S.crete / S.rms;

    % FFT (composante continue retirée)
    x_ac = x - mean(x);
    NFFT = 2^nextpow2(N);
    X = fft(x_ac, NFFT);
    S.f_fft = (0:NFFT/2)' * fs / NFFT;
    Xm = abs(X(1:NFFT/2+1)) / N;
    Xm(2:end-1) = 2 * Xm(2:end-1);
    S.X_mag = Xm;

    % DSP de Welch
    [S.Pxx, S.f_psd] = pwelch(x_ac, hann(fen), fen/2, fen, fs);

    % Plancher médian local et excès aux fréquences caractéristiques
    % (critère opérationnel identique à la Phase 1 : significatif si
    % l'excès atteint au moins 3 dB)
    S.plancher = median(S.Pxx(S.f_psd >= 10 & S.f_psd <= 300));
    ex = @(fcible) 10*log10(S.Pxx(find(abs(S.f_psd - fcible) == ...
        min(abs(S.f_psd - fcible)), 1)) / S.plancher);
    S.ex1X = ex(S.Omega);
    S.ex2X = ex(2*S.Omega);
    S.ex3X = ex(3*S.Omega);
    izone = S.f_psd >= 5*floor(0.42*S.Omega/5) & S.f_psd <= 5*ceil(0.48*S.Omega/5);
    S.exSub = 10*log10(max(S.Pxx(izone)) / S.plancher);
    % Indicateurs spectraux globaux (parité avec la Phase 1)
    f_min = max(S.f_psd(2), 0.1 * S.Omega);
    masque = S.f_psd >= f_min;
    fv = S.f_psd(masque);
    [~, imax] = max(S.Pxx(masque));
    S.fdom = fv(imax);
    Pn = S.Pxx / sum(S.Pxx);
    S.entropie = -sum(Pn .* log2(Pn + eps));
    S.platitude = exp(mean(log(S.Pxx + eps))) / (mean(S.Pxx) + eps);

    % Puissances de bande (comparaison entre états, méthode identique)
    S.p_hf  = bandpower(S.Pxx, S.f_psd, [1500 2500], 'psd');
    S.p_mid = bandpower(S.Pxx, S.f_psd, [500 2000], 'psd');
    % Quatre bandes DISJOINTES pour le test d'élévation large bande plate
    % (règle usure) : elles ne se recouvrent pas, contrairement aux bandes
    % de mesure 500-2000 / 1500-2500 ci-dessus.
    bandes4 = [500 1000; 1000 1500; 1500 2000; 2000 2500];
    S.p_b4 = zeros(1, 4);
    for kb = 1:4
        S.p_b4(kb) = bandpower(S.Pxx, S.f_psd, bandes4(kb, :), 'psd');
    end

    % Validation des entrées canoniques : les textes, repères de fréquence
    % et bandes de ce script sont calibrés pour les signaux du livrable
    % (3600 tr/min soit 1X = 60 Hz, fs = 20 480 Hz, 102 401 échantillons).
    % Échec explicite plutôt que sorties trompeuses si un signal non
    % conforme est fourni.
    if fs ~= 20480 || abs(S.Omega - 60) > 1e-6 || N ~= 102401
        error(['Signal non conforme aux paramètres canoniques du livrable ' ...
            '(attendu : fs = 20480 Hz, 3600 tr/min, 102401 échantillons ; ' ...
            'reçu : fs = %g Hz, 1X = %g Hz, N = %d) : %s'], ...
            fs, S.Omega, N, fichier);
    end
end

function tracer_reperes(code, Omega)
% Trace les repères de fréquences adaptés au défaut sur la figure active.
    switch code
        case 'desalignement'
            xline(Omega,   'r--', '1X', 'LineWidth', 1.2, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(2*Omega, 'g-',  '2X = 120 Hz', 'LineWidth', 1.6, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(3*Omega, 'm-',  '3X = 180 Hz', 'LineWidth', 1.6, 'FontSize', 10, 'LabelOrientation', 'horizontal');
        case 'desequilibre'
            xline(Omega,   'r-',  '1X = 60 Hz', 'LineWidth', 1.6, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(2*Omega, 'g--', '2X', 'LineWidth', 1.2, 'FontSize', 10, 'LabelOrientation', 'horizontal');
        case 'jeu'
            xline(0.43*Omega, 'c-', '~0.43X', 'LineWidth', 1.6, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(Omega,   'r-',  '1X', 'LineWidth', 1.4, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(2*Omega, 'g-',  '2X', 'LineWidth', 1.4, 'FontSize', 10, 'LabelOrientation', 'horizontal');
        case 'lubrification'
            xline(3.5, 'c-', '~3.5 Hz', 'LineWidth', 1.6, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(Omega, 'r--', '1X', 'LineWidth', 1.2, 'FontSize', 10, 'LabelOrientation', 'horizontal');
        case 'cavitation'
            xline(1500, 'c-', '1500 Hz', 'LineWidth', 1.4, 'FontSize', 10, ...
                'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
            xline(2500, 'c-', '2500 Hz', 'LineWidth', 1.4, 'FontSize', 10, ...
                'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
        case 'usure'
            xline(Omega,   'r-', '1X', 'LineWidth', 1.4, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(2*Omega, 'g-', '2X', 'LineWidth', 1.4, 'FontSize', 10, ...
                'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'middle');
        case 'oilwhirl'
            xline(0.45*Omega, 'c-', '0.45X', 'LineWidth', 1.8, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(Omega, 'r--', '1X', 'LineWidth', 1.2, 'FontSize', 10, ...
                'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'middle');
    end
    xline(50, 'k:', 'EMI 50 Hz', 'LineWidth', 1.0, 'FontSize', 9, ...
        'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
end

function [ok, resume, regle] = verifier_signature(code, S, ref, dHF, dBF, dB4, seuil, tol_plat)
% Vérifie que les INDICATEURS ATTENDUS du défaut sont observés. Mêmes
% méthodes d'estimation partout, mais règles indicatives PROPRES À CHAQUE
% DÉFAUT (excès sur le plancher local, niveau sain au même bin, ou
% comparaison de bandes ; seuil indicatif : au moins 3 dB). Chaque règle
% est documentée dans la sortie `regle`. Ces règles constatent des
% indicateurs pour des états simulés CONNUS ; elles ne constituent pas un
% diagnostic exclusif (plusieurs défauts peuvent élever un même
% indicateur).
    switch code
        case 'desalignement'
            regle = sprintf(['excès 2X >= %d dB ET excès 3X >= %d dB ' ...
                'au-dessus du plancher médian local'], seuil, seuil);
            ok = S.ex2X >= seuil && S.ex3X >= seuil;
            resume = sprintf('2X à %+.1f dB et 3X à %+.1f dB au-dessus du plancher', S.ex2X, S.ex3X);
        case 'desequilibre'
            regle = sprintf(['excès 1X >= %d dB ET dominance : excès 1X ' ...
                'supérieur aux excès 2X et sous-synchrone'], seuil);
            ok = S.ex1X >= seuil && S.ex1X > S.ex2X && S.ex1X > S.exSub;
            resume = sprintf(['1X à %+.1f dB, dominant sur 2X (%+.1f dB) et ' ...
                'sur la zone sous-synchrone (%+.1f dB)'], S.ex1X, S.ex2X, S.exSub);
        case 'jeu'
            regle = sprintf(['excès sous-synchrone (bins 25-30 Hz), 1X et 2X ' ...
                'tous >= %d dB au-dessus du plancher médian local'], seuil);
            ok = S.exSub >= seuil && S.ex1X >= seuil && S.ex2X >= seuil;
            resume = sprintf(['sous-synchrone (bins 25-30 Hz) à %+.1f dB, ' ...
                '1X à %+.1f dB et 2X à %+.1f dB'], S.exSub, S.ex1X, S.ex2X);
        case 'lubrification'
            % Le bin de Welch le plus proche de 3.5 Hz est le bin 5 Hz (grille
            % de 5 Hz), contaminé par la dérive < 1 Hz via le lobe de la
            % fenêtre : l'état sain y mesure déjà +17 dB au-dessus du plancher.
            % Le critère est donc référencé au NIVEAU SAIN AU MÊME BIN, qui
            % isole l'énergie propre au défaut. Les impacts déclarés ne sont
            % pas vérifiés par un indicateur global (masqués par la
            % composante d'adhérence-glissement à cette sévérité) ; ils sont
            % montrés au zoom temporel (Figure 2).
            regle = sprintf(['écart au NIVEAU SAIN au bin de Welch le plus ' ...
                'proche de 3.5 Hz (bin 5 Hz) >= %d dB'], seuil);
            ok = dBF >= seuil;
            resume = sprintf(['composante basse fréquence (bin 5 Hz, cible ' ...
                '~3.5 Hz) à %+.1f dB au-dessus du niveau sain au même bin'], dBF);
        case 'cavitation'
            % Énergie de bande ET impulsivité (bouffées) : distingue d'une
            % élévation large bande continue
            regle = sprintf(['écart bande 1500-2500 Hz vs sain >= %d dB ET ' ...
                'kurtosis > 4 (impulsivité des bouffées)'], seuil);
            ok = dHF >= seuil && S.kurt > 4;
            resume = sprintf(['bande 1500-2500 Hz à %+.2f dB au-dessus du sain ' ...
                'ET kurtosis %.2f (bouffées impulsives)'], dHF, S.kurt);
        case 'usure'
            % Élévation large bande PLATE testée sur QUATRE bandes DISJOINTES
            % (500-1000, 1000-1500, 1500-2000, 2000-2500 Hz) : toutes
            % élevées d'au moins `seuil` ET étendue (max - min) sous la
            % tolérance indicative `tol_plat` propre à cette étude. La
            % cavitation, concentrée en 1500-2500 Hz, échoue nettement à ce
            % test de platitude.
            etendue = max(dB4) - min(dB4);
            regle = sprintf(['écarts vs sain des 4 bandes disjointes ' ...
                '(500-1000/1000-1500/1500-2000/2000-2500 Hz) tous >= %d dB ' ...
                'ET étendue (max - min) <= %.1f dB (tolérance indicative de ' ...
                'platitude propre à cette étude)'], seuil, tol_plat);
            ok = all(dB4 >= seuil) && etendue <= tol_plat;
            resume = sprintf(['élévation large bande plate : %+.2f / %+.2f / ' ...
                '%+.2f / %+.2f dB sur les 4 bandes disjointes, étendue ' ...
                '%.2f dB (<= %.1f dB)'], dB4(1), dB4(2), dB4(3), dB4(4), ...
                etendue, tol_plat);
        case 'oilwhirl'
            regle = sprintf(['excès sous-synchrone (bins 25-30 Hz) >= %d dB ' ...
                'ET dominance : excès sous-synchrone supérieur à l''excès 1X'], seuil);
            ok = S.exSub >= seuil && S.exSub > S.ex1X;
            resume = sprintf(['sous-synchrone (bins 25-30 Hz, ~0.45X) à %+.1f dB, ' ...
                'dominante sur 1X (%+.1f dB)'], S.exSub, S.ex1X);
        otherwise
            ok = false; resume = 'défaut inconnu'; regle = 'aucune';
    end
end

function txt = interpretation_defaut(code, nomAff, S, ref, dHF, dBF, dB4, resume, regle, verdict)
% Construit le texte d'interprétation du défaut (valeurs calculées +
% physique du défaut), prêt à adapter pour le rapport.
    switch code
        case 'desalignement'
            phys = sprintf([ ...
'Physique du défaut : un désalignement angulaire ou parallèle de\n' ...
'l''arbre génère des efforts de flexion qui se répètent deux et trois\n' ...
'fois par tour. La signature classique est donc l''élévation des\n' ...
'harmoniques 2X (120 Hz) et 3X (180 Hz) de la fréquence de rotation,\n' ...
'visible sur les Figures 4 et 6.\n']);
        case 'desequilibre'
            phys = sprintf([ ...
'Physique du défaut : un balourd (répartition inégale des masses)\n' ...
'crée une force centrifuge tournante qui excite le palier une fois\n' ...
'par tour. La signature est une raie dominante à la fréquence de\n' ...
'rotation 1X (60 Hz), visible sur les Figures 4 et 6.\n']);
        case 'jeu'
            phys = sprintf([ ...
'Physique du défaut : un jeu excessif dans le palier autorise des\n' ...
'mouvements de l''arbre à des fréquences inférieures à la rotation\n' ...
'(composante sous-synchrone vers 0.43X) accompagnés d''harmoniques\n' ...
'1X et 2X, visibles sur les Figures 4 et 6.\n']);
        case 'lubrification'
            phys = sprintf([ ...
'Physique du défaut : un manque de lubrification provoque un régime\n' ...
'd''adhérence-glissement (stick-slip) très basse fréquence (~3.5 Hz)\n' ...
'et des contacts métal-métal intermittents. À cette sévérité, la\n' ...
'composante sinusoïdale de stick-slip domine le signal : elle abaisse\n' ...
'le kurtosis global en dessous de 3 (une sinusoïde pure a un kurtosis\n' ...
'de 1.5) et masque, dans cet indicateur global, les impacts localisés,\n' ...
'qui sont montrés au zoom temporel (Figure 2, fenêtre 0.70-1.00 s,\n' ...
'impact modélisé à t = 0.8 s). L''indicateur spectral à ~3.5 Hz\n' ...
'(Figure 6) est le marqueur principal.\n']);
        case 'cavitation'
            phys = sprintf([ ...
'Physique du défaut : l''implosion des bulles de cavitation dans le\n' ...
'film d''huile produit des bouffées d''énergie haute fréquence dans\n' ...
'la bande 1500-2500 Hz : la première bouffée est montrée au zoom\n' ...
'temporel (Figure 2, fenêtre 0.45-0.60 s), l''élévation de bande sur\n' ...
'les Figures 5-6, et les bouffées apparaissent comme des colonnes\n' ...
'verticales au spectrogramme (Figure 7).\n']);
        case 'usure'
            phys = sprintf([ ...
'Physique du défaut : l''usure des surfaces augmente le frottement et\n' ...
'produit, tel qu''encodé par le modèle, un bruit blanc LARGE BANDE\n' ...
'qui élève le plancher sur toute la bande analysée (mesuré ici sur\n' ...
'les quatre bandes disjointes 500-1000, 1000-1500, 1500-2000 et\n' ...
'2000-2500 Hz de la règle de platitude), ainsi que des harmoniques de\n' ...
'rotation (1X, 2X, visibles sur les Figures 4 et 6) modulés en\n' ...
'amplitude à ~1 Hz. Cette modulation lente est visible dans\n' ...
'l''enveloppe du signal temporel (Figure 1) ; ses bandes latérales à\n' ...
'±1 Hz ne sont pas résolues par la grille de Welch de 5 Hz des\n' ...
'Figures 5-6. L''élévation étant plate et large bande, elle touche\n' ...
'aussi la bande 1500-2500 Hz utilisée pour la cavitation : la\n' ...
'distinction se fait par la FORME de l''élévation (plancher plat en\n' ...
'continu pour l''usure ; bouffées localisées dans le temps pour la\n' ...
'cavitation, visibles au spectrogramme). Les bandes du critère sont\n' ...
'des bandes de MESURE, pas des signatures exclusives.\n']);
        case 'oilwhirl'
            phys = sprintf([ ...
'Physique du défaut : le tourbillonnement d''huile est une instabilité\n' ...
'du film qui entraîne l''arbre à un peu moins de la moitié de la\n' ...
'vitesse de rotation (~0.45X soit ~27 Hz), signature sous-synchrone\n' ...
'dominante visible sur les Figures 4, 6 et 7.\n']);
        otherwise
            phys = '';
    end
    if verdict
        vtxt = sprintf(['Indicateurs attendus observés pour cet état simulé ' ...
            'connu : %s.'], resume);
    else
        vtxt = sprintf('Indicateurs à examiner : %s.', resume);
    end
    txt = sprintf([ ...
'INTERPRÉTATION DES RÉSULTATS - %s\n' ...
'=====================================================\n\n' ...
'Signal : data_signaux_simulink/%s_001.mat (modèle Simulink, %.0f\n' ...
'tr/min soit 1X = %.0f Hz, charge %.0f %%, température %.0f °C,\n' ...
'sévérité %.1f, fs = %d Hz, durée %.0f s). Comparaison à la référence\n' ...
'saine de la Phase 1 : mêmes méthodes d''estimation, mais règles\n' ...
'indicatives PROPRES À CHAQUE DÉFAUT (excès sur le plancher médian\n' ...
'local, niveau sain au même bin, ou comparaison de bandes ; seuil\n' ...
'indicatif : au moins 3 dB).\n\n' ...
'%s\n' ...
'Résultats mesurés :\n' ...
'  RMS = %.4f (sain : %.4f)\n' ...
'  Kurtosis = %.2f (sain : %.2f) ; facteur de crête = %.2f\n' ...
'  Excès au-dessus du plancher médian local : 1X %+.1f dB,\n' ...
'  2X %+.1f dB, 3X %+.1f dB, zone sous-synchrone (bins de Welch\n' ...
'  25-30 Hz, soit ~0.42-0.50X sur cette grille) %+.1f dB\n' ...
'  Écarts par rapport au sain : bin 5 Hz (cible ~3.5 Hz) %+.1f dB ;\n' ...
'  bandes disjointes 500-1000 : %+.2f dB, 1000-1500 : %+.2f dB,\n' ...
'  1500-2000 : %+.2f dB, 2000-2500 : %+.2f dB ;\n' ...
'  bande cavitation 1500-2500 Hz : %+.2f dB\n\n' ...
'Note de lecture du kurtosis (propre à ces signaux simulés) :\n' ...
'lorsqu''une composante quasi sinusoïdale domine le signal (balourd,\n' ...
'tourbillonnement, adhérence-glissement), le kurtosis global descend\n' ...
'EN DESSOUS de 3 (une sinusoïde pure a un kurtosis de 1.5) ; dans ce\n' ...
'cadre, un kurtosis inférieur à 3 est donc cohérent avec une\n' ...
'composante périodique dominante, et non le signe d''un signal plus\n' ...
'sain. Les défauts impulsifs (cavitation) l''élèvent nettement\n' ...
'au-dessus de 3 ; une modulation d''amplitude (usure) peut aussi le\n' ...
'porter légèrement au-dessus de 3 sans impulsivité.\n\n' ...
'Règle appliquée (indicative, propre à cette étude) : %s\n\n' ...
'%s\n\n' ...
'Ces indicateurs sont COMPARATIFS et non exclusifs : plusieurs\n' ...
'défauts peuvent élever un même indicateur (l''usure élève par\n' ...
'exemple la bande 1500-2500 Hz davantage que la cavitation).\n' ...
'L''identification d''un état inconnu s''appuierait sur leur\n' ...
'combinaison (voir le tableau comparatif), pas sur un indicateur\n' ...
'isolé.\n\n' ...
'Les conclusions décrivent le comportement du défaut tel qu''encodé\n' ...
'par le modèle de simulation. Cette livraison Phase 2 couvre\n' ...
'uniquement les sept défauts simples ; l''analyse des trois défauts\n' ...
'mixtes n''est pas incluse dans cette archive et reste à confirmer\n' ...
'séparément.\n'], ...
    upper(nomAff), code, S.Omega*60, S.Omega, S.load_pct, S.temp_C, S.sev, ...
    S.fs, S.T, phys, S.rms, ref.rms, S.kurt, ref.kurt, ...
    S.fc, S.ex1X, S.ex2X, S.ex3X, S.exSub, dBF, dB4(1), dB4(2), dB4(3), ...
    dB4(4), dHF, regle, vtxt);
end

function nv = nomVarValide(code)
% Nom de variable de table valide à partir du code du défaut
    nv = matlab.lang.makeValidName(code);
end

function exporter_figure(figH, chemin, dpi)
% Exporte une figure en supprimant d'abord la barre d'outils de chaque
% axe. En mode sans affichage, MATLAB peut malgré tout incruster la
% barre d'outils dans les premiers exports d'une session (avertissement
% "Exported image displays axes toolbar") ; le remède documenté est de
% réexporter. On détecte donc cet avertissement et on réexporte
% automatiquement, ce qui garantit un PNG propre.
    drawnow;
    for axh = reshape(findall(figH, 'Type', 'axes'), 1, [])
        try
            delete(axh.Toolbar);
        catch
        end
    end
    lastwarn('');
    evalc('exportgraphics(figH, chemin, ''Resolution'', dpi)');
    if contains(lower(lastwarn), 'toolbar')
        lastwarn('');
        evalc('exportgraphics(figH, chemin, ''Resolution'', dpi)');
    end
end
