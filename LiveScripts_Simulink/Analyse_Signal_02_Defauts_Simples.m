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
%%   5. Vérification de la signature attendue (critère >= 3 dB) et
%%      texte d'interprétation
%% Puis un tableau comparatif global (sain + 7 défauts) et une figure
%% comparative.
%%
%% Version : 1.0 (Phase 2 - Défauts simples) - 2026-08
%% Compatible : MATLAB R2024b, Signal Processing Toolbox, Statistics and
%%              Machine Learning Toolbox, Wavelet Toolbox
%% ========================================================================

clear; clc; close all;

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
% d'affichage fréquentielle [Hz], et description courte de la signature
% attendue (issue de la physique du modèle).
DEFAUTS = {
% code            nom affiché                 bande     signature attendue
 'desalignement', 'Désalignement',            [0 500],  'harmoniques 2X (120 Hz) et 3X (180 Hz)'
 'desequilibre',  'Déséquilibre',             [0 500],  'composante 1X dominante (60 Hz)'
 'jeu',           'Jeu',                      [0 500],  'composante sous-synchrone ~0.43X + 1X + 2X'
 'lubrification', 'Lubrification',            [0 500],  'adhérence-glissement très basse fréquence (~3.5 Hz) + impacts métal-métal'
 'cavitation',    'Cavitation',               [0 3000], 'bouffées haute fréquence 1500-2500 Hz'
 'usure',         'Usure',                    [0 500],  'bruit large bande (500-2000 Hz) + harmoniques modulés'
 'oilwhirl',      'Tourbillonnement d''huile', [0 500],  'composante sous-synchrone ~0.45X (~27 Hz) dominante'
};

fen = 4096;              % fenêtre de Welch (grille 5 Hz)
SEUIL_DB = 3;            % seuil de signification (au moins 3 dB), identique Phase 1

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

% Accumulateur du tableau comparatif (le sain d'abord)
comp = struct('nom', {}, 'rms', {}, 'kurt', {}, 'fc', {}, ...
    'ex1X', {}, 'ex2X', {}, 'ex3X', {}, 'exSub', {}, 'dHF', {}, 'dMid', {});
comp(1) = struct('nom', 'Sain', 'rms', ref.rms, 'kurt', ref.kurt, ...
    'fc', ref.fc, 'ex1X', ref.ex1X, 'ex2X', ref.ex2X, 'ex3X', ref.ex3X, ...
    'exSub', ref.exSub, 'dHF', 0, 'dMid', 0);

%% ========================================================================
%% BOUCLE SUR LES 7 DÉFAUTS
%% ========================================================================

for kd = 1:size(DEFAUTS, 1)
    code   = DEFAUTS{kd, 1};
    nomAff = DEFAUTS{kd, 2};
    bande  = DEFAUTS{kd, 3};
    signat = DEFAUTS{kd, 4};

    fprintf('========================================================================\n');
    fprintf('DÉFAUT %d/7 : %s\n', kd, upper(nomAff));
    fprintf('  Signature attendue : %s\n', signat);
    fprintf('------------------------------------------------------------------------\n');

    outDir = fullfile(OUT_ROOT, code);
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    % ---- Analyse complète du signal ----
    S = analyser_signal(fullfile(DATA_DIR, [code '_001.mat']), fen);
    x = S.x; fs = S.fs; t = S.t; N = S.N; T = S.T; Omega = S.Omega;
    T_rot = 1 / Omega;

    fprintf('  Signal charge : N = %d, fs = %d Hz, 1X = %.0f Hz\n', N, fs, Omega);

    % ---- Écarts par rapport au sain (mêmes bandes, mêmes méthodes) ----
    dHF  = 10*log10(S.p_hf  / ref.p_hf);    % bande cavitation 1500-2500 Hz
    dMid = 10*log10(S.p_mid / ref.p_mid);   % bande usure 500-2000 Hz

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
    exportgraphics(fig1, fullfile(outDir, sprintf('Fig1_%s_Temporel_Complet.png', code)), ...
        'Resolution', EXPORT_DPI);

    %% Figure 2 : zoom temporel (100 ms, repères de rotation)
    zoom_dur = 0.100;
    idxz = t <= zoom_dur;
    fig2 = figure('Name', [nomAff ' - Zoom'], ...
        'Position', [100, 100, 1200, 500], 'Color', 'white');
    plot(t(idxz)*1000, x(idxz), 'b-', 'LineWidth', 0.8);
    hold on;
    for kk = 0:floor(zoom_dur / T_rot)
        xline(kk * T_rot * 1000, 'r--', 'LineWidth', 1.0);
    end
    xlabel('Temps (ms)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Figure 2 : Zoom temporel (100 ms) - %s', nomAff), ...
        'FontSize', 15, 'FontWeight', 'bold');
    subtitle(sprintf('Traits rouges : période de rotation T = %.2f ms (1X = %.0f Hz)', ...
        T_rot*1000, Omega), 'FontSize', 11);
    grid on; xlim([0, zoom_dur*1000]); set(gca, 'FontSize', 11);
    exportgraphics(fig2, fullfile(outDir, sprintf('Fig2_%s_Temporel_Zoom.png', code)), ...
        'Resolution', EXPORT_DPI);

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
    exportgraphics(fig3, fullfile(outDir, sprintf('Fig3_%s_Distribution.png', code)), ...
        'Resolution', EXPORT_DPI);

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
    subtitle(sprintf('Zoom %d-%d Hz. Signature attendue : %s.', ...
        bande(1), bande(2), signat), 'FontSize', 10);
    grid on; xlim(bande); set(gca, 'FontSize', 11);
    exportgraphics(fig4, fullfile(outDir, sprintf('Fig4_%s_Spectre_FFT.png', code)), ...
        'Resolution', EXPORT_DPI);

    %% Figure 5 : DSP de Welch (bande complète, échelle log)
    fig5 = figure('Name', [nomAff ' - DSP Welch'], ...
        'Position', [100, 100, 1200, 500], 'Color', 'white');
    semilogy(S.f_psd, S.Pxx, 'b-', 'LineWidth', 0.9);
    hold on;
    semilogy(ref.f_psd, ref.Pxx, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.7);
    legend({nomAff, 'Sain (référence)'}, 'Location', 'northeast', 'FontSize', 10);
    xlabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('DSP (unité^2/Hz)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Figure 5 : DSP (Welch) sur la bande complète - %s vs sain', nomAff), ...
        'FontSize', 15, 'FontWeight', 'bold');
    subtitle(sprintf('Fenêtre de Hann %d points, recouvrement 50 %%, bande 0-%d Hz', ...
        fen, fs/2), 'FontSize', 11);
    grid on; xlim([0, fs/2]); set(gca, 'FontSize', 11);
    exportgraphics(fig5, fullfile(outDir, sprintf('Fig5_%s_DSP_Welch.png', code)), ...
        'Resolution', EXPORT_DPI);

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
    subtitle(sprintf('Bande %d-%d Hz. Signature attendue : %s.', ...
        bande(1), bande(2), signat), 'FontSize', 10);
    grid on; xlim(bande); set(gca, 'FontSize', 11);
    exportgraphics(fig6, fullfile(outDir, sprintf('Fig6_%s_DSP_Zoom.png', code)), ...
        'Resolution', EXPORT_DPI);

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
    exportgraphics(fig7, fullfile(outDir, sprintf('Fig7_%s_Spectrogramme_STFT.png', code)), ...
        'Resolution', EXPORT_DPI);

    %% Figure 8 : transformée en ondelettes continue (CWT)
    fig8 = figure('Name', [nomAff ' - CWT'], ...
        'Position', [100, 100, 1200, 600], 'Color', 'white');
    [cfs, frq] = cwt(x, 'amor', fs);
    surface(t, frq, abs(cfs));
    axis tight; shading interp; view(0, 90); colormap('parula');
    cb = colorbar; cb.Label.String = 'Module'; cb.Label.FontSize = 11;
    ylim(bande);
    ax8 = gca; ax8.Toolbar = [];   % évite la barre d'outils dans l'export
    xlabel('Temps (s)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Figure 8 : Ondelettes continues (Morlet analytique) - %s', nomAff), ...
        'FontSize', 15, 'FontWeight', 'bold');
    subtitle(sprintf('Bande %d-%d Hz.', bande(1), bande(2)), 'FontSize', 11);
    set(gca, 'FontSize', 11, 'YScale', 'linear');
    exportgraphics(fig8, fullfile(outDir, sprintf('Fig8_%s_CWT.png', code)), ...
        'Resolution', EXPORT_DPI);

    close all;

    %% Vérification de la signature attendue + interprétation
    fprintf('\n  --- VÉRIFICATION DE LA SIGNATURE (critère : au moins %d dB) ---\n', SEUIL_DB);
    fprintf('  Excès au-dessus du plancher médian local du défaut :\n');
    fprintf('    zone sous-synchrone : %+.1f dB | 1X : %+.1f dB | 2X : %+.1f dB | 3X : %+.1f dB\n', ...
        S.exSub, S.ex1X, S.ex2X, S.ex3X);
    fprintf('  Écarts par rapport au sain : bande 1500-2500 Hz : %+.1f dB | bande 500-2000 Hz : %+.1f dB\n', ...
        dHF, dMid);
    fprintf('  Kurtosis : %.2f (sain : %.2f) | RMS : %.4f (sain : %.4f)\n', ...
        S.kurt, ref.kurt, S.rms, ref.rms);

    [verdict, resume] = verifier_signature(code, S, ref, dHF, dMid, SEUIL_DB);
    if verdict
        fprintf('  SIGNATURE CONFIRMÉE : %s\n\n', resume);
    else
        fprintf('  ATTENTION : signature non confirmée (%s)\n\n', resume);
    end

    % ---- Texte d'interprétation ----
    interp = interpretation_defaut(code, nomAff, S, ref, dHF, dMid, resume, verdict);
    fid = fopen(fullfile(outDir, sprintf('Interpretation_%s.txt', code)), 'w', 'n', 'UTF-8');
    fprintf(fid, '%s', interp);
    fclose(fid);
    fprintf('  8 figures + tableau CSV + interprétation exportés dans %s\n\n', outDir);

    % ---- Ligne du tableau comparatif ----
    comp(end+1) = struct('nom', nomAff, 'rms', S.rms, 'kurt', S.kurt, ...
        'fc', S.fc, 'ex1X', S.ex1X, 'ex2X', S.ex2X, 'ex3X', S.ex3X, ...
        'exSub', S.exSub, 'dHF', dHF, 'dMid', dMid); %#ok<SAGROW>
end

%% ========================================================================
%% TABLEAU COMPARATIF GLOBAL + FIGURE COMPARATIVE
%% ========================================================================

fprintf('========================================================================\n');
fprintf('SYNTHÈSE COMPARATIVE (sain + 7 défauts)\n');
fprintf('------------------------------------------------------------------------\n');

Tcomp = table({comp.nom}', [comp.rms]', [comp.kurt]', [comp.fc]', ...
    [comp.ex1X]', [comp.ex2X]', [comp.ex3X]', [comp.exSub]', ...
    [comp.dHF]', [comp.dMid]', ...
    'VariableNames', {'Etat', 'RMS', 'Kurtosis', 'Facteur_crete', ...
    'Exces_1X_dB', 'Exces_2X_dB', 'Exces_3X_dB', 'Exces_sous_sync_dB', ...
    'Bande_HF_vs_sain_dB', 'Bande_mid_vs_sain_dB'});
disp(Tcomp);
writetable(Tcomp, fullfile(OUT_ROOT, 'Tableau_Comparatif_Simples.csv'), ...
    'Encoding', 'UTF-8');
fprintf('  Tableau comparatif exporté : Tableau_Comparatif_Simples.csv\n');

% Figure comparative : 4 indicateurs discriminants en barres
figC = figure('Name', 'Comparaison des états', ...
    'Position', [60, 60, 1400, 800], 'Color', 'white');
etats = {comp.nom};
subplot(2, 2, 1);
bar([comp.rms], 'FaceColor', [0.3 0.5 0.8]);
title('Valeur efficace (RMS)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('RMS', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 9); xtickangle(35); grid on;
subplot(2, 2, 2);
bar([comp.kurt], 'FaceColor', [0.85 0.5 0.3]);
hold on; yline(3, 'k--', 'gaussien = 3', 'FontSize', 9);
title('Kurtosis', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Kurtosis', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 9); xtickangle(35); grid on;
subplot(2, 2, 3);
bar([comp.ex1X; comp.ex2X]', 'grouped');
hold on; yline(3, 'k--', 'seuil 3 dB', 'FontSize', 9);
legend({'1X', '2X'}, 'Location', 'northwest', 'FontSize', 9);
title('Excès aux harmoniques 1X et 2X (dB)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('dB au-dessus du plancher', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 9); xtickangle(35); grid on;
subplot(2, 2, 4);
bar([comp.exSub], 'FaceColor', [0.5 0.7 0.4]);
hold on; yline(3, 'k--', 'seuil 3 dB', 'FontSize', 9);
title('Excès en zone sous-synchrone 0.42-0.48X (dB)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('dB au-dessus du plancher', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 9); xtickangle(35); grid on;
sgtitle('Figure C1 : Indicateurs discriminants - sain et 7 défauts simples', ...
    'FontSize', 15, 'FontWeight', 'bold');
exportgraphics(figC, fullfile(OUT_ROOT, 'FigC1_Comparatif_Indicateurs.png'), ...
    'Resolution', EXPORT_DPI);
close all;
fprintf('  Figure comparative exportée : FigC1_Comparatif_Indicateurs.png\n\n');

fprintf('========================================================================\n');
fprintf('   ANALYSE TERMINÉE - 7 défauts x (8 figures + tableau + interprétation)\n');
fprintf('   + tableau comparatif global + figure comparative dans %s\n', OUT_ROOT);
fprintf('========================================================================\n');
fprintf('Prochaine étape : analyse des 3 défauts mixtes (Phase 3).\n\n');

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
    if isfield(d, 'metadata') && isfield(d.metadata, 'speed_rpm')
        S.Omega = d.metadata.speed_rpm / 60;
    else
        S.Omega = 60;
    end

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
    % Adhérence-glissement (lubrification) : bin le plus proche de 3.5 Hz
    S.exBF = 10*log10(S.Pxx(find(abs(S.f_psd - 3.5) == ...
        min(abs(S.f_psd - 3.5)), 1)) / S.plancher);

    % Puissances de bande (comparaison entre états, méthode identique)
    S.p_hf  = bandpower(S.Pxx, S.f_psd, [1500 2500], 'psd');
    S.p_mid = bandpower(S.Pxx, S.f_psd, [500 2000], 'psd');
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
            xline(1500, 'c-', '1500 Hz', 'LineWidth', 1.4, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(2500, 'c-', '2500 Hz', 'LineWidth', 1.4, 'FontSize', 10, 'LabelOrientation', 'horizontal');
        case 'usure'
            xline(Omega,   'r-', '1X', 'LineWidth', 1.4, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(2*Omega, 'g-', '2X', 'LineWidth', 1.4, 'FontSize', 10, 'LabelOrientation', 'horizontal');
        case 'oilwhirl'
            xline(0.45*Omega, 'c-', '~0.45X = 27 Hz', 'LineWidth', 1.8, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(Omega, 'r--', '1X', 'LineWidth', 1.2, 'FontSize', 10, 'LabelOrientation', 'horizontal');
    end
    xline(50, 'k:', 'EMI 50 Hz', 'LineWidth', 1.0, 'FontSize', 9, ...
        'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
end

function [ok, resume] = verifier_signature(code, S, ref, dHF, dMid, seuil)
% Vérifie la signature attendue du défaut (règles indicatives du cadre
% d'étude, appliquées à l'identique : significatif si au moins 3 dB).
    switch code
        case 'desalignement'
            ok = S.ex2X >= seuil && S.ex3X >= seuil;
            resume = sprintf('2X à %+.1f dB et 3X à %+.1f dB au-dessus du plancher', S.ex2X, S.ex3X);
        case 'desequilibre'
            ok = S.ex1X >= seuil;
            resume = sprintf('1X à %+.1f dB au-dessus du plancher', S.ex1X);
        case 'jeu'
            ok = S.exSub >= seuil && S.ex1X >= seuil;
            resume = sprintf('sous-synchrone à %+.1f dB et 1X à %+.1f dB', S.exSub, S.ex1X);
        case 'lubrification'
            ok = S.exBF >= seuil || S.kurt > ref.kurt + 0.5;
            resume = sprintf('composante ~3.5 Hz à %+.1f dB ; kurtosis %.2f (sain %.2f)', ...
                S.exBF, S.kurt, ref.kurt);
        case 'cavitation'
            ok = dHF >= seuil;
            resume = sprintf('bande 1500-2500 Hz à %+.1f dB au-dessus du sain', dHF);
        case 'usure'
            ok = dMid >= seuil;
            resume = sprintf('bande 500-2000 Hz à %+.1f dB au-dessus du sain', dMid);
        case 'oilwhirl'
            ok = S.exSub >= seuil;
            resume = sprintf('sous-synchrone (~0.45X) à %+.1f dB au-dessus du plancher', S.exSub);
        otherwise
            ok = false; resume = 'défaut inconnu';
    end
end

function txt = interpretation_defaut(code, nomAff, S, ref, dHF, dMid, resume, verdict)
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
'de 1.5) et masque, dans cet indicateur global, les impacts localisés\n' ...
'qui restent visibles dans la vue temporelle (Figure 1). La signature\n' ...
'spectrale à ~3.5 Hz (Figure 6) est le marqueur principal.\n']);
        case 'cavitation'
            phys = sprintf([ ...
'Physique du défaut : l''implosion des bulles de cavitation dans le\n' ...
'film d''huile produit des bouffées d''énergie haute fréquence dans\n' ...
'la bande 1500-2500 Hz, visibles sur les Figures 5 à 8 (bande élargie\n' ...
'à 0-3000 Hz) et sous forme de colonnes dans le spectrogramme.\n']);
        case 'usure'
            phys = sprintf([ ...
'Physique du défaut : l''usure des surfaces augmente le frottement et\n' ...
'produit un bruit large bande (500-2000 Hz) ainsi que des harmoniques\n' ...
'de rotation modulés en amplitude, visibles sur les Figures 5 et 6.\n']);
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
        vtxt = sprintf('SIGNATURE CONFIRMÉE : %s.', resume);
    else
        vtxt = sprintf('Signature à examiner : %s.', resume);
    end
    txt = sprintf([ ...
'INTERPRÉTATION DES RÉSULTATS - %s\n' ...
'=====================================================\n\n' ...
'Signal : data_signaux_simulink/%s_001.mat (modèle Simulink, 3600\n' ...
'tr/min soit 1X = 60 Hz, charge 70 %%, température 60 °C, sévérité\n' ...
'0.7, fs = %d Hz, durée %.0f s). Comparaison à la référence saine de\n' ...
'la Phase 1, mêmes méthodes et mêmes critères (significatif à partir\n' ...
'de 3 dB ; règles indicatives du cadre d''étude).\n\n' ...
'%s\n' ...
'Résultats mesurés :\n' ...
'  RMS = %.4f (sain : %.4f)\n' ...
'  Kurtosis = %.2f (sain : %.2f) ; facteur de crête = %.2f\n' ...
'  Excès au-dessus du plancher médian local : 1X %+.1f dB,\n' ...
'  2X %+.1f dB, 3X %+.1f dB, zone sous-synchrone %+.1f dB\n' ...
'  Écarts par rapport au sain : bande 1500-2500 Hz %+.1f dB,\n' ...
'  bande 500-2000 Hz %+.1f dB\n\n' ...
'Note de lecture du kurtosis : lorsqu''une composante périodique\n' ...
'domine le signal (balourd, tourbillonnement, stick-slip), le\n' ...
'kurtosis global descend EN DESSOUS de 3 (une sinusoïde pure a un\n' ...
'kurtosis de 1.5) ; un kurtosis inférieur à 3 signale donc ici une\n' ...
'composante périodique dominante, et non un signal plus sain. Seuls\n' ...
'les défauts impulsifs (cavitation) l''élèvent au-dessus de 3.\n\n' ...
'%s\n\n' ...
'Les conclusions décrivent le comportement du défaut tel qu''encodé\n' ...
'par le modèle de simulation ; elles servent de référence interne\n' ...
'pour la suite de l''étude (défauts mixtes, Phase 3).\n'], ...
    upper(nomAff), code, S.fs, S.T, phys, S.rms, ref.rms, S.kurt, ref.kurt, ...
    S.fc, S.ex1X, S.ex2X, S.ex3X, S.exSub, dHF, dMid, vtxt);
end

function nv = nomVarValide(code)
% Nom de variable de table valide à partir du code du défaut
    nv = matlab.lang.makeValidName(code);
end
