# Production du signal sain a partir du modele Simulink

Ce document explique, etape par etape, comment le signal de reference de
l'etat sain (`sain_001.mat`) est produit par le modele Simulink
`PFD_Signal_Generator`. Deux methodes sont possibles : par l'interface
Simulink (methode A) ou par script (methode B). Les deux donnent
exactement le meme signal.

---

## 1. Ce que fait le modele pour l'etat sain

Le modele est constitue de six sous-systemes (voir la capture
`screenshots/01_top_level.png` du livrable) :

| Sous-systeme | Role | Cas de l'etat sain |
|---|---|---|
| Operating_Conditions (bleu) | Convertit vitesse, charge et temperature en parametres physiques et calcule le nombre de Sommerfeld | 3600 tr/min, 70 %, 60 C : facteur de charge 0.79, Sommerfeld S = 0.190 |
| Base_Signal (vert) | Bruit de base du palier en fonctionnement | Present (amplitude faible) |
| Fault_Injection (rouge) | Genere la signature du defaut selectionne par la constante Fault_Type (1 a 11) | Fault_Type = 1 : sortie NULLE, aucun defaut |
| Severity_Control (violet) | Multiplie la signature du defaut par la severite (0 a 1) | Sans effet pour l'etat sain (le defaut est nul). Le script de generation utilise 1.0 (nominal) ; la constante du modele vaut 0.7 par defaut |
| Transient_Behavior (beige) | Applique un transitoire optionnel (rampe de vitesse, echelon de charge, thermique) | Transient_Type = 1 : aucun |
| Noise_Model (orange) | 7 sources de bruit de mesure realistes | Present : bruit capteur, interference secteur 50 Hz, bruit rose, derives, impulsions, repliement |

Le signal final est la somme :

```
x = Base_Signal + (Defaut x Severite x Transitoire) + Bruits de mesure
```

puis une quantification (pas de 0.001) simule le convertisseur
analogique-numerique. Pour l'etat sain, le terme central est nul : le
signal sain est donc le bruit de base plus les bruits de mesure. C'est
exactement la signature d'une machine en bon etat : un plancher de bruit
large bande, sans raie de rotation dominante ni signature de defaut.

Parametres de simulation : duree 5 s, frequence d'echantillonnage
20 480 Hz, solveur a pas fixe (ode4), soit 102 401 echantillons.

---

## 2. Methode A : par l'interface Simulink

1. Ouvrir MATLAB R2024b dans le dossier du projet.
2. Ouvrir le modele : double-clic sur `PFD_Signal_Generator.slx`
   (ou `>> open_system('PFD_Signal_Generator')`).
3. Double-cliquer sur le sous-systeme rouge **Fault_Injection**, puis sur
   la constante **Fault_Type**, et mettre sa valeur a **1** (= Sain).
   Valider avec OK.
4. (Facultatif, valeurs deja par defaut) Verifier dans
   **Operating_Conditions** : Speed_RPM = 3600, Load_Percent = 70,
   Temperature_C = 60. Dans **Severity_Control** : Enable_Evolution = 0.
   Dans **Transient_Behavior** : Transient_Type = 1 (aucun).
5. Lancer la simulation : bouton **Run** (ou Ctrl+T). La simulation dure
   5 secondes de temps simule.
6. Observer le signal dans le **Scope**. Le signal complet est disponible
   dans l'espace de travail MATLAB sous la variable **x_sim**.
7. Sauvegarder le signal au format .mat :

```matlab
x = x_sim(:);
fs = 20480;
fault = 'sain';
save('sain_001.mat', 'x', 'fs', 'fault');
```

## 3. Methode B : par script (automatique, recommandee)

Le script `generate_simulink_signals.m` fait tout automatiquement :
il reconstruit le modele, le configure pour chacun des 11 etats (dont
l'etat sain avec les parametres ci-dessus), lance les 11 simulations et
sauvegarde les fichiers dans `data_signaux_simulink/`, avec les
metadonnees completes (vitesse, charge, temperature, nombre de
Sommerfeld, severite, version du generateur).

```matlab
>> generate_simulink_signals
```

Resultat : `data_signaux_simulink/sain_001.mat` (et les 10 signaux de
defauts). Chaque fichier contient :

- `x` : le signal vibratoire (102 401 echantillons, colonne)
- `fs` : la frequence d'echantillonnage (20 480 Hz)
- `fault` : le nom de l'etat (`'sain'`)
- `metadata` : les parametres exacts du modele lors de la generation

## 4. Utilisation comme reference d'analyse

Le signal sain sert de reference a tout le travail. Pour l'analyser :

```matlab
>> run('LiveScripts_Simulink/Analyse_Signal_01_Sain.m')
```

Ce script effectue les quatre familles d'analyse convenues (temporelle,
statistique, frequentielle, temps-frequence STFT + CWT) et exporte les
figures en francais (300 DPI) dans `Figures_Simulink/Sain/`.
