%% CAPTURE_ECRANS_SAIN  Captures d'écran du modèle configuré pour l'état sain
%
% Configure le modèle PFD_Signal_Generator pour l'état SAIN
% (Fault_Type = 1, sévérité 1.0, sans évolution ni transitoire), exporte
% les captures des vues du modèle, lance la simulation et exporte la
% figure du signal obtenu. Le fichier .slx n'est PAS modifié sur le
% disque (le modèle est fermé sans sauvegarde).
%
% Sortie : LiveScripts_Simulink/screenshots_sain/*.png (150 dpi)

clear; clc; close all;

scriptPath = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptPath);
cd(projectRoot);

outDir = fullfile(scriptPath, 'screenshots_sain');
if ~exist(outDir, 'dir'), mkdir(outDir); end

modelName = 'PFD_Signal_Generator';
modelFile = fullfile(projectRoot, [modelName '.slx']);
if ~exist(modelFile, 'file')
    error('Modèle introuvable : %s', modelFile);
end
if bdIsLoaded(modelName), close_system(modelName, 0); end
load_system(modelFile);
fprintf('Modèle chargé : %s\n', modelFile);

%% Configuration de l'état sain
set_param([modelName '/Fault_Injection/Fault_Type'], 'Value', '1');
set_param([modelName '/Severity_Control/Severity_Level'], 'Value', '1.0');
set_param([modelName '/Severity_Control/Enable_Evolution'], 'Value', '0');
set_param([modelName '/Transient_Behavior/Transient_Type'], 'Value', '1');
set_param([modelName '/Operating_Conditions/Speed_RPM'], 'Value', '3600');
set_param([modelName '/Operating_Conditions/Load_Percent'], 'Value', '70');
set_param([modelName '/Operating_Conditions/Temperature_C'], 'Value', '60');
fprintf('Configuration état sain appliquée (Fault_Type = 1).\n');

%% Captures des vues du modèle
captures = {
    modelName,                            '01_modele_complet'
    [modelName '/Fault_Injection'],       '02_fault_injection_sain'
    [modelName '/Operating_Conditions'],  '03_conditions_fonctionnement'
    [modelName '/Severity_Control'],      '04_controle_severite'
    [modelName '/Noise_Model'],           '05_modele_de_bruit'
};
for k = 1:size(captures, 1)
    sysPath = captures{k, 1};
    open_system(sysPath);
    print(['-s' sysPath], fullfile(outDir, [captures{k,2} '.png']), ...
        '-dpng', '-r150');
    fprintf('  Capture : %s.png\n', captures{k,2});
end

%% Simulation et figure du résultat
fprintf('Simulation de l''état sain (5 s)...\n');
simOut = sim(modelName, 'ReturnWorkspaceOutputs', 'on');
try
    x = simOut.get('x_sim');
catch
    x = evalin('base', 'x_sim');
end
x = x(:);
fs = 20480;
t = (0:length(x)-1)' / fs;

figR = figure('Name', 'Résultat simulation sain', ...
    'Position', [100, 100, 1200, 600], 'Color', 'white');
subplot(2, 1, 1);
plot(t, x, 'b-', 'LineWidth', 0.4);
xlabel('Temps (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
title('Signal obtenu dans x\_sim : vue complète (5 s)', ...
    'FontSize', 13, 'FontWeight', 'bold');
grid on; xlim([0, 5]);
subplot(2, 1, 2);
idx = t <= 0.1;
plot(t(idx)*1000, x(idx), 'b-', 'LineWidth', 0.8);
xlabel('Temps (ms)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
title('Zoom : 100 premières millisecondes', 'FontSize', 13, 'FontWeight', 'bold');
grid on; xlim([0, 100]);
sgtitle('Résultat de la simulation - État sain (Fault\_Type = 1)', ...
    'FontSize', 15, 'FontWeight', 'bold');

exportgraphics(figR, fullfile(outDir, '06_resultat_simulation.png'), ...
    'Resolution', 150);
fprintf('  Capture : 06_resultat_simulation.png\n');

% Vérification rapide du signal obtenu
fprintf('Signal : N = %d, RMS = %.4f, crête = %.4f\n', ...
    length(x), rms(x), max(abs(x)));

%% Fermeture SANS sauvegarde (le .slx sur disque reste inchangé)
close_system(modelName, 0);
fprintf('Terminé. Captures dans : %s\n', outDir);
