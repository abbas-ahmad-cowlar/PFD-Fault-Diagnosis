%% ANALYSE DU SIGNAL 03 : LES 3 DÉFAUTS MIXTES
%%
%% Analyse vibratoire multi-domaine d'un système palier hydrodynamique
%% Signaux produits par le modèle Simulink PFD_Signal_Generator (v3.1)
%%
%% Ce script analyse les 3 défauts mixtes avec les mêmes quatre familles
%% d'analyse que les Phases 1 et 2, et compare chaque défaut mixte à la
%% référence saine ET à ses deux défauts simples constitutifs :
%%   - Désalignement + Déséquilibre
%%   - Usure + Lubrification
%%   - Cavitation + Jeu
%%
%% CONTENU PAR DÉFAUT MIXTE :
%%   1. Analyse temporelle (vue complète + zoom sur un événement)
%%   2. Analyse statistique (8 indicateurs + comparaison au sain)
%%   3. Analyse fréquentielle (FFT + DSP de Welch + zoom annoté)
%%   4. Analyse temps-fréquence (spectrogramme STFT + ondelettes CWT)
%%   5. Figure de superposition : mixte vs ses deux constituants vs sain
%%      (le coeur de l'originalité de l'étude)
%%   6. Vérification des indicateurs attendus (règles documentées : les
%%      indicateurs principaux des DEUX constituants) et interprétation
%% Puis un tableau comparatif global et une figure comparative.
%%
%% Version : 1.0 (Phase 3 - Défauts mixtes) - 2026-08
%% Compatible : MATLAB R2024b, Signal Processing Toolbox, Statistics and
%%              Machine Learning Toolbox, Wavelet Toolbox
%% ========================================================================

clear; clc; close all;

set(groot, 'DefaultAxesCreateFcn', @(ax, ~) set(ax.Toolbar, 'Visible', 'off'));
nettoyage = onCleanup(@() set(groot, 'DefaultAxesCreateFcn', ''));

scriptPath = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptPath);
cd(projectRoot);
fprintf('Répertoire de travail : %s\n\n', pwd);

%% Configuration
% =========================================================================
DATA_DIR   = 'data_signaux_simulink';
OUT_ROOT   = 'Figures_Simulink/Mixtes';
EXPORT_DPI = 300;

if ~exist(OUT_ROOT, 'dir'), mkdir(OUT_ROOT); end

% Les 3 défauts mixtes : code fichier, nom affiché, codes des deux
% constituants, noms affichés des constituants, bande d'affichage [Hz],
% indicateurs attendus (les indicateurs principaux des DEUX constituants),
% fenêtre du zoom temporel [s], note de zoom.
MIXTES = {
 'mixed_misalign_imbalance', 'Désalignement + Déséquilibre', ...
   {'desalignement', 'desequilibre'}, {'Désalignement', 'Déséquilibre'}, ...
   [0 500], 'harmoniques 2X et 3X (désalignement) ET composante 1X (déséquilibre)', ...
   [0 0.100], ''
 'mixed_wear_lube', 'Usure + Lubrification', ...
   {'usure', 'lubrification'}, {'Usure', 'Lubrification'}, ...
   [0 3000], 'élévation large bande plate + harmoniques 1X/2X (usure) ET composante ~3.5 Hz (lubrification)', ...
   [0.90 1.20], 'Fenêtre 0.90-1.20 s : ondulation d''adhérence-glissement ; l''impact modélisé à t = 1.0 s (1 ms) reste noyé dans le bruit large bande.'
 'mixed_cavit_jeu', 'Cavitation + Jeu', ...
   {'cavitation', 'jeu'}, {'Cavitation', 'Jeu'}, ...
   [0 3000], 'bouffées 1500-2500 Hz (cavitation) ET sous-synchrone + 1X (jeu ; sans le 2X du jeu simple)', ...
   [0.55 0.70], 'Fenêtre 0.55-0.70 s : première bouffée de cavitation modélisée à t = 0.6 s.'
};

fen = 4096;              % fenêtre de Welch (grille 5 Hz)
SEUIL_DB = 3;            % seuil de signification (au moins 3 dB), identique Phases 1-2
TOL_PLAT_DB = 1.5;       % tolérance de platitude (règle usure), identique Phase 2

fprintf('========================================================================\n');
fprintf('   ANALYSE DES 3 DÉFAUTS MIXTES - SIGNAUX ISSUS DU MODÈLE SIMULINK\n');
fprintf('========================================================================\n\n');

%% ========================================================================
%% SECTION 0 : RÉFÉRENCES (SAIN + DÉFAUTS SIMPLES CONSTITUTIFS)
%% ========================================================================
% Le sain et les défauts simples concernés sont recalculés avec exactement
% les mêmes paramètres d'estimation, pour des comparaisons à méthode
% identique.

fprintf('SECTION 0 : Calcul des références (sain + constituants)...\n');
ref = analyser_signal(fullfile(DATA_DIR, 'sain_001.mat'), fen);
fprintf('  Sain : RMS = %.4f, kurtosis = %.2f\n', ref.rms, ref.kurt);

% Défauts simples nécessaires (constituants des 3 mixtes)
codes_simples = {'desalignement', 'desequilibre', 'usure', 'lubrification', ...
    'cavitation', 'jeu'};
SIMPLES = struct();
for ks = 1:numel(codes_simples)
    c = codes_simples{ks};
    SIMPLES.(c) = analyser_signal(fullfile(DATA_DIR, [c '_001.mat']), fen);
    fprintf('  %s : RMS = %.4f\n', c, SIMPLES.(c).rms);
end
fprintf('\n');

i_bin35 = find(abs(ref.f_psd - 3.5) == min(abs(ref.f_psd - 3.5)), 1);

% Accumulateur comparatif : sain + 3 mixtes (les indicateurs des simples
% figurent déjà dans le tableau de la Phase 2 ; on les rappelle ici dans
% les colonnes 'constituants' des interprétations et la figure C2)
comp = struct('nom', {}, 'rms', {}, 'kurt', {}, 'fc', {}, ...
    'ex1X', {}, 'ex2X', {}, 'ex3X', {}, 'exSub', {}, 'dHF', {}, ...
    'dBF', {}, 'dB4', {});
comp(1) = struct('nom', 'Sain', 'rms', ref.rms, 'kurt', ref.kurt, ...
    'fc', ref.fc, 'ex1X', ref.ex1X, 'ex2X', ref.ex2X, 'ex3X', ref.ex3X, ...
    'exSub', ref.exSub, 'dHF', 0, 'dBF', 0, 'dB4', zeros(1,4));

%% ========================================================================
%% BOUCLE SUR LES 3 DÉFAUTS MIXTES
%% ========================================================================

nb_observes = 0;
verdicts = cell(size(MIXTES, 1), 1);

for kd = 1:size(MIXTES, 1)
    code    = MIXTES{kd, 1};
    nomAff  = MIXTES{kd, 2};
    cCodes  = MIXTES{kd, 3};
    cNoms   = MIXTES{kd, 4};
    bande   = MIXTES{kd, 5};
    signat  = MIXTES{kd, 6};
    zoomw   = MIXTES{kd, 7};
    znote   = MIXTES{kd, 8};
    C1 = SIMPLES.(cCodes{1});
    C2 = SIMPLES.(cCodes{2});

    fprintf('========================================================================\n');
    fprintf('DÉFAUT MIXTE %d/3 : %s\n', kd, upper(nomAff));
    fprintf('  Indicateurs attendus : %s\n', signat);
    fprintf('------------------------------------------------------------------------\n');

    outDir = fullfile(OUT_ROOT, code);
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    S = analyser_signal(fullfile(DATA_DIR, [code '_001.mat']), fen);
    x = S.x; fs = S.fs; t = S.t; N = S.N; T = S.T; Omega = S.Omega;
    T_rot = 1 / Omega;

    fprintf('  Signal chargé : N = %d, fs = %d Hz, 1X = %.0f Hz\n', N, fs, Omega);

    dHF = 10*log10(S.p_hf  / ref.p_hf);
    dB4 = 10*log10(S.p_b4 ./ ref.p_b4);
    dBF = 10*log10(S.Pxx(i_bin35) / ref.Pxx(i_bin35));

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

    %% Figure 2 : zoom temporel (fenêtre caractéristique)
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

    % ---- Tableau statistique (mixte + les deux constituants + sain) ----
    noms = {'Moyenne'; 'Valeur efficace (RMS)'; 'Écart-type'; 'Valeur crête'; ...
        'Crête-à-crête'; 'Asymétrie (skewness)'; 'Kurtosis'; 'Facteur de crête'};
    vM  = [S.moy;  S.rms;  S.ecart;  S.crete;  S.c2c;  S.asym;  S.kurt;  S.fc];
    v1  = [C1.moy; C1.rms; C1.ecart; C1.crete; C1.c2c; C1.asym; C1.kurt; C1.fc];
    v2  = [C2.moy; C2.rms; C2.ecart; C2.crete; C2.c2c; C2.asym; C2.kurt; C2.fc];
    vS  = [ref.moy; ref.rms; ref.ecart; ref.crete; ref.c2c; ref.asym; ref.kurt; ref.fc];
    tableau = table(noms, vM, v1, v2, vS, 'VariableNames', ...
        {'Indicateur', 'Mixte', matlab.lang.makeValidName(cCodes{1}), ...
         matlab.lang.makeValidName(cCodes{2}), 'Sain_reference'});
    disp(tableau);
    writetable(tableau, fullfile(outDir, sprintf('Tableau_Statistiques_%s.csv', code)), ...
        'Encoding', 'UTF-8');

    %% Figure 4 : spectre d'amplitude FFT
    fig4 = figure('Name', [nomAff ' - Spectre FFT'], ...
        'Position', [100, 100, 1200, 500], 'Color', 'white');
    plot(S.f_fft, S.X_mag, 'b-', 'LineWidth', 0.8);
    hold on;
    tracer_reperes_mixte(code, Omega, bande(2) > 500);
    xlabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Figure 4 : Spectre d''amplitude (FFT) - %s', nomAff), ...
        'FontSize', 15, 'FontWeight', 'bold');
    subtitle(sprintf('Zoom %d-%d Hz. Indicateurs attendus : %s.', ...
        bande(1), bande(2), signat), 'FontSize', 9);
    grid on; xlim(bande); set(gca, 'FontSize', 11);
    exporter_figure(fig4, fullfile(outDir, sprintf('Fig4_%s_Spectre_FFT.png', code)), EXPORT_DPI);

    %% Figure 5 : DSP de Welch (bande complète) vs sain
    fig5 = figure('Name', [nomAff ' - DSP Welch'], ...
        'Position', [100, 100, 1200, 500], 'Color', 'white');
    semilogy(S.f_psd, S.Pxx, 'b-', 'LineWidth', 0.9);
    hold on;
    semilogy(ref.f_psd, ref.Pxx, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.7);
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

    %% Figure 6 : DSP zoom annoté vs sain
    fig6 = figure('Name', [nomAff ' - DSP zoom'], ...
        'Position', [100, 100, 1200, 500], 'Color', 'white');
    plot(S.f_psd, 10*log10(S.Pxx + eps), 'b-', 'LineWidth', 1.0);
    hold on;
    plot(ref.f_psd, 10*log10(ref.Pxx + eps), '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.7);
    tracer_reperes_mixte(code, Omega, bande(2) > 500);
    legend({nomAff, 'Sain (référence)'}, 'Location', 'northeast', 'FontSize', 10);
    xlabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('DSP (dB)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Figure 6 : DSP en zone caractéristique - %s vs sain', nomAff), ...
        'FontSize', 15, 'FontWeight', 'bold');
    subtitle(sprintf('Bande %d-%d Hz. Indicateurs attendus : %s.', ...
        bande(1), bande(2), signat), 'FontSize', 9);
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

    %% Figure 9 : superposition mixte vs constituants vs sain (originalité)
    % Pour les défauts mixtes à bande large (usure+lubrification,
    % cavitation+jeu), la signature basse fréquence du constituant serait
    % illisible sur l'axe complet : la figure est alors en deux panneaux
    % (zoom basse fréquence + bande large).
    if bande(2) > 500
        fig9 = figure('Name', [nomAff ' - Superposition'], ...
            'Position', [60, 100, 1500, 550], 'Color', 'white');
        subplot(1, 2, 1);
        trace_superposition(S, C1, C2, ref);
        tracer_reperes_mixte(code, Omega, false);
        xlabel('Fréquence (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('DSP (dB)', 'FontSize', 12, 'FontWeight', 'bold');
        title('Zone basse fréquence (0-200 Hz)', 'FontSize', 13, 'FontWeight', 'bold');
        grid on; xlim([0, 200]); set(gca, 'FontSize', 10);
        subplot(1, 2, 2);
        trace_superposition(S, C1, C2, ref);
        if strcmp(code, 'mixed_cavit_jeu')
            xline(1500, 'c-', '1500 Hz', 'LineWidth', 1.2, 'FontSize', 9, ...
                'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
            xline(2500, 'c-', '2500 Hz', 'LineWidth', 1.2, 'FontSize', 9, ...
                'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
        end
        legend({['Mixte : ' nomAff], cNoms{1}, cNoms{2}, 'Sain (référence)'}, ...
            'Location', 'northeast', 'FontSize', 8);
        xlabel('Fréquence (Hz)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('DSP (dB)', 'FontSize', 12, 'FontWeight', 'bold');
        title(sprintf('Bande large (0-%d Hz)', bande(2)), 'FontSize', 13, 'FontWeight', 'bold');
        grid on; xlim(bande); set(gca, 'FontSize', 10);
        sgtitle(sprintf('Figure 9 : Superposition - %s vs ses constituants et le sain', ...
            nomAff), 'FontSize', 14, 'FontWeight', 'bold');
    else
        fig9 = figure('Name', [nomAff ' - Superposition'], ...
            'Position', [100, 100, 1200, 550], 'Color', 'white');
        trace_superposition(S, C1, C2, ref);
        tracer_reperes_mixte(code, Omega, false);
        legend({['Mixte : ' nomAff], cNoms{1}, cNoms{2}, 'Sain (référence)'}, ...
            'Location', 'northeast', 'FontSize', 9);
        xlabel('Fréquence (Hz)', 'FontSize', 13, 'FontWeight', 'bold');
        ylabel('DSP (dB)', 'FontSize', 13, 'FontWeight', 'bold');
        title(sprintf('Figure 9 : Superposition - %s vs ses constituants', nomAff), ...
            'FontSize', 15, 'FontWeight', 'bold');
        subtitle(['Composantes reprises, réduites ou omises par le modèle mixte : ' ...
            'bilan détaillé dans l''interprétation.'], 'FontSize', 10);
        grid on; xlim(bande); set(gca, 'FontSize', 11);
    end
    exporter_figure(fig9, fullfile(outDir, sprintf('Fig9_%s_Superposition.png', code)), EXPORT_DPI);

    close all;

    %% Vérification des indicateurs attendus + interprétation
    fprintf('\n  --- VÉRIFICATION DES INDICATEURS ATTENDUS (seuil : au moins %d dB) ---\n', SEUIL_DB);
    fprintf('  Excès plancher local : sous-sync %+.1f | 1X %+.1f | 2X %+.1f | 3X %+.1f dB\n', ...
        S.exSub, S.ex1X, S.ex2X, S.ex3X);
    fprintf('  Écarts vs sain : bin 5 Hz %+.1f dB | bandes disjointes %+.2f / %+.2f / %+.2f / %+.2f dB\n', ...
        pz(dBF), pz(dB4(1)), pz(dB4(2)), pz(dB4(3)), pz(dB4(4)));
    fprintf('  Kurtosis : %.2f (sain : %.2f) | RMS : %.4f (sain : %.4f)\n', ...
        S.kurt, ref.kurt, S.rms, ref.rms);

    [verdict, resume, regle] = verifier_mixte(code, S, dHF, dBF, dB4, SEUIL_DB, TOL_PLAT_DB);
    if verdict
        fprintf('  INDICATEURS ATTENDUS OBSERVÉS (état simulé connu) : %s\n\n', resume);
        nb_observes = nb_observes + 1;
    else
        fprintf('  ATTENTION : indicateurs attendus NON observés (%s)\n\n', resume);
    end
    verdicts{kd} = struct('nom', nomAff, 'ok', verdict, 'resume', resume);

    interp = interpretation_mixte(code, nomAff, cNoms, S, C1, C2, ref, ...
        dHF, dBF, dB4, resume, regle, verdict);
    fid = fopen(fullfile(outDir, sprintf('Interpretation_%s.txt', code)), 'w', 'n', 'UTF-8');
    fprintf(fid, '%s', interp);
    fclose(fid);
    fprintf('  9 figures + tableau CSV + interprétation exportés dans %s\n\n', outDir);

    comp(end+1) = struct('nom', nomAff, 'rms', S.rms, 'kurt', S.kurt, ...
        'fc', S.fc, 'ex1X', S.ex1X, 'ex2X', S.ex2X, 'ex3X', S.ex3X, ...
        'exSub', S.exSub, 'dHF', dHF, 'dBF', dBF, 'dB4', dB4); %#ok<SAGROW>
end

%% ========================================================================
%% TABLEAU COMPARATIF + FIGURE COMPARATIVE
%% ========================================================================

fprintf('========================================================================\n');
fprintf('SYNTHÈSE COMPARATIVE (sain + 3 défauts mixtes)\n');
fprintf('------------------------------------------------------------------------\n');

dB4mat = vertcat(comp.dB4);
Tcomp = table({comp.nom}', [comp.rms]', [comp.kurt]', [comp.fc]', ...
    [comp.ex1X]', [comp.ex2X]', [comp.ex3X]', [comp.exSub]', ...
    [comp.dBF]', dB4mat(:,1), dB4mat(:,2), dB4mat(:,3), dB4mat(:,4), ...
    [comp.dHF]', ...
    'VariableNames', {'Etat', 'RMS', 'Kurtosis', 'Facteur_crete', ...
    'Exces_1X_dB', 'Exces_2X_dB', 'Exces_3X_dB', 'Exces_sous_sync_dB', ...
    'Bin5Hz_vs_sain_dB', 'B500_1000_vs_sain_dB', 'B1000_1500_vs_sain_dB', ...
    'B1500_2000_vs_sain_dB', 'B2000_2500_vs_sain_dB', 'Bande_HF_vs_sain_dB'});
disp(Tcomp);
writetable(Tcomp, fullfile(OUT_ROOT, 'Tableau_Comparatif_Mixtes.csv'), ...
    'Encoding', 'UTF-8');
fprintf('  Tableau comparatif exporté : Tableau_Comparatif_Mixtes.csv\n');

figC = figure('Name', 'Comparaison des états mixtes', ...
    'Position', [40, 40, 1700, 500], 'Color', 'white');
etats = {comp.nom};
subplot(1, 3, 1);
bar([comp.ex1X; comp.ex2X; comp.ex3X]', 'grouped');
hold on; yline(3, 'k--', 'seuil 3 dB', 'FontSize', 8);
legend({'1X', '2X', '3X'}, 'Location', 'northwest', 'FontSize', 8);
title('Excès aux harmoniques (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('dB au-dessus du plancher', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 8); xtickangle(25); grid on;
subplot(1, 3, 2);
bar([comp.exSub; comp.dBF]', 'grouped');
hold on; yline(3, 'k--', 'seuil 3 dB', 'FontSize', 8);
legend({'Sous-synchrone (bins 25-30 Hz)', 'Bin 5 Hz vs sain'}, ...
    'Location', 'northwest', 'FontSize', 8);
title('Composantes basse fréquence (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('dB', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 8); xtickangle(25); grid on;
subplot(1, 3, 3);
bar(vertcat(comp.dB4), 'grouped');
hold on; yline(3, 'k--', 'seuil 3 dB', 'FontSize', 8);
legend({'500-1000', '1000-1500', '1500-2000', '2000-2500 Hz'}, ...
    'Location', 'northwest', 'FontSize', 7);
title('Écarts au sain des 4 bandes disjointes (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('dB par rapport au sain', 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTickLabel', etats, 'FontSize', 8); xtickangle(25); grid on;
sgtitle('Figure C2 : Indicateurs comparatifs - sain et 3 défauts mixtes', ...
    'FontSize', 15, 'FontWeight', 'bold');
exporter_figure(figC, fullfile(OUT_ROOT, 'FigC2_Comparatif_Mixtes.png'), EXPORT_DPI);
close all;
fprintf('  Figure comparative exportée : FigC2_Comparatif_Mixtes.png\n\n');

fprintf('========================================================================\n');
if nb_observes == size(MIXTES, 1)
    fprintf('   ANALYSE TERMINÉE - indicateurs attendus observés pour les %d défauts mixtes\n', nb_observes);
else
    fprintf('   ANALYSE TERMINÉE AVEC RÉSERVES : indicateurs observés pour %d/%d défauts mixtes.\n', ...
        nb_observes, size(MIXTES, 1));
    for kv = 1:numel(verdicts)
        if ~verdicts{kv}.ok
            fprintf('   NON OBSERVÉ - %s : %s\n', verdicts{kv}.nom, verdicts{kv}.resume);
        end
    end
end
fprintf('   3 défauts mixtes x (9 figures + tableau + interprétation) + tableau\n');
fprintf('   et figure comparatifs dans %s\n', OUT_ROOT);
fprintf('========================================================================\n\n');

%% ========================================================================
%% FONCTIONS LOCALES
%% ========================================================================

function S = analyser_signal(fichier, fen)
% Identique aux Phases 1-2 : charge un signal, exige les métadonnées,
% valide les paramètres canoniques et calcule tous les indicateurs.
    d = load(fichier);
    x = d.x(:);
    fs = double(d.fs);
    N = length(x);
    S.x = x; S.fs = fs; S.N = N;
    S.T = (N-1) / fs;
    S.t = (0:N-1)' / fs;

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

    S.moy   = mean(x);
    S.rms   = rms(x);
    S.ecart = std(x);
    S.crete = max(abs(x));
    S.c2c   = max(x) - min(x);
    S.asym  = skewness(x);
    S.kurt  = kurtosis(x);
    S.fc    = S.crete / S.rms;

    x_ac = x - mean(x);
    NFFT = 2^nextpow2(N);
    X = fft(x_ac, NFFT);
    S.f_fft = (0:NFFT/2)' * fs / NFFT;
    Xm = abs(X(1:NFFT/2+1)) / N;
    Xm(2:end-1) = 2 * Xm(2:end-1);
    S.X_mag = Xm;

    [S.Pxx, S.f_psd] = pwelch(x_ac, hann(fen), fen/2, fen, fs);

    S.plancher = median(S.Pxx(S.f_psd >= 10 & S.f_psd <= 300));
    ex = @(fcible) 10*log10(S.Pxx(find(abs(S.f_psd - fcible) == ...
        min(abs(S.f_psd - fcible)), 1)) / S.plancher);
    S.ex1X = ex(S.Omega);
    S.ex2X = ex(2*S.Omega);
    S.ex3X = ex(3*S.Omega);
    izone = S.f_psd >= 5*floor(0.42*S.Omega/5) & S.f_psd <= 5*ceil(0.48*S.Omega/5);
    S.exSub = 10*log10(max(S.Pxx(izone)) / S.plancher);

    f_min = max(S.f_psd(2), 0.1 * S.Omega);
    masque = S.f_psd >= f_min;
    fv = S.f_psd(masque);
    [~, imax] = max(S.Pxx(masque));
    S.fdom = fv(imax);
    Pn = S.Pxx / sum(S.Pxx);
    S.entropie = -sum(Pn .* log2(Pn + eps));
    S.platitude = exp(mean(log(S.Pxx + eps))) / (mean(S.Pxx) + eps);

    S.p_hf  = bandpower(S.Pxx, S.f_psd, [1500 2500], 'psd');
    S.p_mid = bandpower(S.Pxx, S.f_psd, [500 2000], 'psd');
    bandes4 = [500 1000; 1000 1500; 1500 2000; 2000 2500];
    S.p_b4 = zeros(1, 4);
    for kb = 1:4
        S.p_b4(kb) = bandpower(S.Pxx, S.f_psd, bandes4(kb, :), 'psd');
    end

    if fs ~= 20480 || abs(S.Omega - 60) > 1e-6 || N ~= 102401
        error(['Signal non conforme aux paramètres canoniques du livrable ' ...
            '(attendu : fs = 20480 Hz, 3600 tr/min, 102401 échantillons ; ' ...
            'reçu : fs = %g Hz, 1X = %g Hz, N = %d) : %s'], ...
            fs, S.Omega, N, fichier);
    end
end

function trace_superposition(S, C1, C2, ref)
% Trace les quatre DSP (mixte, constituants, sain) sur l'axe courant.
    plot(S.f_psd, 10*log10(S.Pxx + eps), 'b-', 'LineWidth', 1.3);
    hold on;
    plot(C1.f_psd, 10*log10(C1.Pxx + eps), '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 0.9);
    plot(C2.f_psd, 10*log10(C2.Pxx + eps), '-', 'Color', [0.47 0.67 0.19], 'LineWidth', 0.9);
    plot(ref.f_psd, 10*log10(ref.Pxx + eps), '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.7);
end

function tracer_reperes_mixte(code, Omega, large)
% Repères de fréquences adaptés à chaque défaut mixte. En mode `large`
% (axe > 500 Hz), les repères basse fréquence sont tracés SANS étiquette
% (elles se chevaucheraient) ; les fréquences sont rappelées dans le
% sous-titre et étiquetées sur les vues zoomées (Figure 9, panneau
% gauche).
    if nargin < 3, large = false; end
    if large, et = @(txt) ''; else, et = @(txt) txt; end
    switch code
        case 'mixed_misalign_imbalance'
            xline(Omega,   'r-', et('1X'), 'LineWidth', 1.6, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(2*Omega, 'g-', et('2X'), 'LineWidth', 1.6, 'FontSize', 10, ...
                'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'middle');
            xline(3*Omega, 'm-', et('3X'), 'LineWidth', 1.6, 'FontSize', 10, 'LabelOrientation', 'horizontal');
        case 'mixed_wear_lube'
            xline(3.5, 'c-', et('~3.5 Hz'), 'LineWidth', 1.6, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(Omega,   'r-', et('1X'), 'LineWidth', 1.4, 'FontSize', 10, ...
                'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'middle');
            xline(2*Omega, 'g-', et('2X'), 'LineWidth', 1.4, 'FontSize', 10, ...
                'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
        case 'mixed_cavit_jeu'
            xline(0.43*Omega, 'c-', et('0.43X'), 'LineWidth', 1.6, 'FontSize', 10, 'LabelOrientation', 'horizontal');
            xline(Omega, 'r-', et('1X'), 'LineWidth', 1.4, 'FontSize', 10, ...
                'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'middle');
            xline(1500, 'c-', '1500 Hz', 'LineWidth', 1.2, 'FontSize', 9, ...
                'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
            xline(2500, 'c-', '2500 Hz', 'LineWidth', 1.2, 'FontSize', 9, ...
                'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
    end
    xline(50, 'k:', et('EMI 50 Hz'), 'LineWidth', 1.0, 'FontSize', 9, ...
        'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
end

function v = pz(v)
% Élimine le zéro négatif de l'affichage (-0.00 -> 0.00).
    if abs(v) < 0.005, v = 0; end
end

function [ok, resume, regle] = verifier_mixte(code, S, dHF, dBF, dB4, seuil, tol_plat)
% Vérifie que les indicateurs principaux des DEUX constituants du défaut
% mixte sont observés. Règles indicatives propres à chaque défaut mixte,
% documentées dans `regle`. Ces règles constatent des indicateurs pour des
% états simulés CONNUS ; elles ne constituent pas un diagnostic exclusif.
    switch code
        case 'mixed_misalign_imbalance'
            regle = sprintf(['part désalignement : excès 2X ET 3X >= %d dB ; ' ...
                'part déséquilibre : excès 1X >= %d dB (pas de condition de ' ...
                'dominance : les deux constituants coexistent)'], seuil, seuil);
            ok = S.ex2X >= seuil && S.ex3X >= seuil && S.ex1X >= seuil;
            resume = sprintf('2X à %+.1f dB, 3X à %+.1f dB ET 1X à %+.1f dB', ...
                S.ex2X, S.ex3X, S.ex1X);
        case 'mixed_wear_lube'
            % Kurtosis non utilisé : la composante d'adhérence-glissement
            % domine la statistique globale (voir note de lecture). Les
            % impacts modélisés (1 ms) restent noyés dans le bruit large
            % bande de la part usure à cette sévérité : ils ne sont pas
            % vérifiés ni montrés comme discernables.
            etendue = max(dB4) - min(dB4);
            regle = sprintf(['part usure : écarts vs sain des 4 bandes ' ...
                'disjointes tous >= %d dB ET étendue <= %.1f dB ; ' ...
                'part lubrification : écart au sain au bin 5 Hz >= %d dB'], ...
                seuil, tol_plat, seuil);
            ok = all(dB4 >= seuil) && etendue <= tol_plat && dBF >= seuil;
            resume = sprintf(['bandes disjointes %+.2f / %+.2f / %+.2f / %+.2f dB ' ...
                '(étendue %.2f dB) ET bin 5 Hz à %+.1f dB'], ...
                dB4(1), dB4(2), dB4(3), dB4(4), max(dB4)-min(dB4), dBF);
        case 'mixed_cavit_jeu'
            % Kurtosis non utilisé : la composante sous-synchrone du jeu
            % domine la statistique globale ; l'impulsivité des bouffées est
            % montrée au zoom temporel (Figure 2) et au spectrogramme.
            regle = sprintf(['part cavitation : écart bande 1500-2500 Hz vs ' ...
                'sain >= %d dB ; part jeu : excès sous-synchrone ET 1X ' ...
                '>= %d dB'], seuil, seuil);
            ok = dHF >= seuil && S.exSub >= seuil && S.ex1X >= seuil;
            resume = sprintf(['bande 1500-2500 Hz à %+.2f dB vs sain ET ' ...
                'sous-synchrone à %+.1f dB, 1X à %+.1f dB'], dHF, S.exSub, S.ex1X);
        otherwise
            ok = false; resume = 'défaut inconnu'; regle = 'aucune';
    end
end

function txt = interpretation_mixte(code, nomAff, cNoms, S, C1, C2, ref, ...
    dHF, dBF, dB4, resume, regle, verdict)
% Texte d'interprétation du défaut mixte : physique de la superposition,
% valeurs mesurées, comparaison aux constituants.
    switch code
        case 'mixed_misalign_imbalance'
            phys = sprintf([ ...
'Physique du défaut : le modèle superpose les signatures du\n' ...
'désalignement (harmoniques 2X et 3X) et du déséquilibre (composante\n' ...
'1X). Le spectre combine donc les trois raies 1X, 2X et 3X\n' ...
'(Figures 4, 6 et 9), là où chaque défaut simple n''en présente\n' ...
'qu''une partie : c''est la coexistence des deux familles de raies\n' ...
'qui caractérise ce défaut mixte.\n' ...
'Bilan de la superposition encodée par le modèle : toutes les\n' ...
'composantes des deux constituants sont reprises, avec des\n' ...
'coefficients réduits (2X : 0.25 contre 0.35 ; 3X : 0.15 contre\n' ...
'0.20 ; 1X : 0.35 contre 0.50) ; les trois raies restent très\n' ...
'au-dessus du seuil de 3 dB.\n']);
        case 'mixed_wear_lube'
            phys = sprintf([ ...
'Physique du défaut : le modèle superpose la part usure - bruit blanc\n' ...
'large bande ET harmoniques d''aspérités 1X + 2X (ce sont elles qui\n' ...
'produisent les raies annotées à 60 et 120 Hz sur les Figures 4 et\n' ...
'6) - et la part lubrification : composante d''adhérence-glissement à\n' ...
'~3.5 Hz et impacts métal-métal. Le plancher s''élève de façon plate\n' ...
'sur les quatre bandes disjointes ET le bin basse fréquence monte\n' ...
'nettement au-dessus du niveau sain - voir la superposition en\n' ...
'Figure 9.\n' ...
'Bilan de la superposition encodée par le modèle : bruit d''usure\n' ...
'réduit (0.18 contre 0.25), harmoniques d''aspérités réduits (0.08\n' ...
'contre 0.12) et SANS la modulation ~1 Hz du défaut simple ;\n' ...
'adhérence-glissement réduit (0.20 contre 0.30). Les impacts\n' ...
'modélisés (1 ms, le premier à t = 1.0 s) restent noyés dans le\n' ...
'bruit large bande de la part usure à cette sévérité : ils ne sont\n' ...
'pas discernables au zoom temporel (Figure 2), contrairement au\n' ...
'défaut simple de lubrification où le fond est plus faible.\n']);
        case 'mixed_cavit_jeu'
            phys = sprintf([ ...
'Physique du défaut : le modèle superpose les bouffées haute\n' ...
'fréquence de la cavitation (5 bouffées, la première à t = 0.6 s,\n' ...
'montrée au zoom temporel, Figure 2, et visibles comme des colonnes\n' ...
'au spectrogramme, Figure 7) et une partie de la signature du jeu\n' ...
'(composante sous-synchrone ~0.43X + 1X). Le spectre combine donc\n' ...
'l''élévation de la bande 1500-2500 Hz et les raies basse fréquence\n' ...
'du jeu (Figures 4, 6 et 9).\n' ...
'Bilan de la superposition encodée par le modèle : bouffées de\n' ...
'cavitation réduites (amplitude 0.4, 5 bouffées : écart de bande\n' ...
'+3.75 dB contre +5.31 dB pour le défaut simple) ; sous-synchrone\n' ...
'réduite (0.22 contre 0.25) et 1X réduit (0.15 contre 0.18). En\n' ...
'revanche, le modèle mixte N''INCLUT PAS la composante 2X du jeu\n' ...
'simple : le 2X mesure %+.1f dB (niveau sain), et la Figure 9\n' ...
'(panneau gauche) montre la raie 2X du constituant jeu à 120 Hz sans\n' ...
'équivalent sur le défaut mixte. La superposition est donc PARTIELLE\n' ...
'pour la part jeu, telle qu''encodée par le modèle.\n'], pz(S.ex2X));
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
'saine ET aux deux défauts simples constitutifs (%s, %s), avec les\n' ...
'mêmes méthodes d''estimation que les Phases 1 et 2 et des règles\n' ...
'indicatives propres à chaque défaut mixte.\n\n' ...
'%s\n' ...
'Résultats mesurés :\n' ...
'  RMS = %.4f (constituants : %.4f et %.4f ; sain : %.4f)\n' ...
'  Kurtosis = %.2f (constituants : %.2f et %.2f ; sain : %.2f)\n' ...
'  Excès au-dessus du plancher médian local : 1X %+.1f dB,\n' ...
'  2X %+.1f dB, 3X %+.1f dB, zone sous-synchrone (bins de Welch\n' ...
'  25-30 Hz) %+.1f dB\n' ...
'  Écarts par rapport au sain : bin 5 Hz %+.1f dB ; bandes disjointes\n' ...
'  500-1000 : %+.2f dB, 1000-1500 : %+.2f dB, 1500-2000 : %+.2f dB,\n' ...
'  2000-2500 : %+.2f dB ; bande cavitation 1500-2500 Hz : %+.2f dB\n\n' ...
'Note de lecture du kurtosis (propre à ces signaux simulés) :\n' ...
'lorsqu''une composante quasi sinusoïdale domine le signal, le\n' ...
'kurtosis global descend en dessous de 3 (une sinusoïde pure a un\n' ...
'kurtosis de 1.5) ; dans ce cadre, un kurtosis inférieur à 3 est\n' ...
'cohérent avec une composante périodique dominante. Pour les défauts\n' ...
'mixtes, le kurtosis global reflète le mélange des contributions des\n' ...
'deux constituants ; il n''est utilisé dans aucune règle de cette\n' ...
'phase. La visibilité éventuelle des événements impulsifs est\n' ...
'discutée au cas par cas dans le paragraphe de physique ci-dessus.\n\n' ...
'Règle appliquée (indicative, propre à cette étude) : %s\n\n' ...
'%s\n\n' ...
'Ces indicateurs sont COMPARATIFS et non exclusifs ; l''identification\n' ...
'd''un état inconnu s''appuierait sur leur combinaison (voir le\n' ...
'tableau comparatif et la Figure 9 de superposition), pas sur un\n' ...
'indicateur isolé. La Figure 9 montre directement comment le défaut\n' ...
'mixte reprend les signatures de ses constituants - avec les\n' ...
'réductions et, le cas échéant, les omissions détaillées dans le\n' ...
'bilan de superposition ci-dessus - telles qu''encodées par le modèle\n' ...
'de simulation : c''est l''argument central de l''originalité de\n' ...
'l''étude (défauts combinés, plus proches des conditions\n' ...
'industrielles réelles).\n'], ...
    upper(nomAff), code, S.Omega*60, S.Omega, S.load_pct, S.temp_C, S.sev, ...
    S.fs, S.T, cNoms{1}, cNoms{2}, phys, ...
    S.rms, C1.rms, C2.rms, ref.rms, S.kurt, C1.kurt, C2.kurt, ref.kurt, ...
    S.ex1X, S.ex2X, S.ex3X, S.exSub, pz(dBF), pz(dB4(1)), pz(dB4(2)), ...
    pz(dB4(3)), pz(dB4(4)), pz(dHF), regle, vtxt);
end

function exporter_figure(figH, chemin, dpi)
% Export avec suppression de la barre d'outils des axes et réexport
% automatique si l'avertissement d'incrustation apparaît (Phases 1-2).
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
