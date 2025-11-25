# Figure Naming Scheme - Consistent Sequential Numbering

**Date**: November 25, 2025
**Purpose**: Establish consistent sequential figure naming (Fig1 through FigN) for both pipeline.m and Technical_report.tex

---

## 🎯 Mapping Table: Old → New Names

| New Name | Old Name | Description | Section | Status |
|----------|----------|-------------|---------|--------|
| **Fig1_Basic_Faults.png** | Fig0a_Basic_Faults.png | Visualization of 8 basic fault types | Data Visualization | ✓ Rename |
| **Fig2_Mixed_Faults.png** | Fig0b_Mixed_Faults.png | Visualization of 3 mixed fault types | Data Visualization | ✓ Rename |
| **Fig3_Feature_Correlation.png** | Fig1_Feature_Correlation.png | Correlation matrix of 15 features | Feature Analysis | ✓ Rename |
| **Fig4_Feature_Distributions.png** | Fig2_Feature_Distributions.png | Distribution histograms | Feature Analysis | ✓ Rename |
| **Fig5_tSNE_Clusters.png** | Fig3_tSNE_Clusters.png | t-SNE visualization | Feature Analysis | ✓ Rename |
| **Fig6_Class_Distribution.png** | Fig7_Class_Distribution.png | Train/Val/Test split | Data Split | ✓ Rename |
| **Fig7_SVM_Confusion_Matrix.png** | Fig4_SVM_Confusion_Matrix.png | SVM confusion matrix | SVM Performance | ✓ Rename |
| **Fig8_SVM_ROC_Curves.png** | Fig5_SVM_ROC_Curves.png | SVM ROC curves (11 classes) | SVM Performance | ✓ Rename |
| **Fig9_RandomForest_Confusion_Matrix.png** | Fig4_RandomForest_Confusion_Matrix.png | RF confusion matrix | RF Performance | ✓ Rename |
| **Fig10_RandomForest_ROC_Curves.png** | Fig5_RandomForest_ROC_Curves.png | RF ROC curves (11 classes) | RF Performance | ✓ Rename |
| **Fig11_NeuralNetwork_Confusion_Matrix.png** | Fig4_NeuralNetwork_Confusion_Matrix.png | NN confusion matrix | NN Performance | ✓ Rename |
| **Fig12_NeuralNetwork_ROC_Curves.png** | Fig5_NeuralNetwork_ROC_Curves.png | NN ROC curves (11 classes) | NN Performance | ✓ Rename |
| **Fig13_Performance_Comparison.png** | Fig8_Performance_Comparison.png | Bar chart comparing all models | Comparative | ✓ Rename |
| **Fig14_All_Models_Comparison.png** | Fig11_All_Models_Performance_Comparison.png | 4-panel detailed comparison | Comparative | ✓ Rename |
| **Fig15_Feature_Importance.png** | Fig12_Feature_Importance_Comparison.png | Feature importance across models | Comparative | ✓ Rename |

### Figures to REMOVE (Replaced by Model-Specific Versions):
- ~~Fig4_Confusion_Matrix.png~~ → **REMOVED** (now have Fig7, Fig9, Fig11 for each model)
- ~~Fig5_ROC_Curves.png~~ → **REMOVED** (now have Fig8, Fig10, Fig12 for each model)

### Optional Figures (Conditionally Generated):
- Fig16_Learning_Curves.png (was Fig6) - Generated if step5b enabled
- Fig17_Old_Feature_Importance.png (was Fig9) - Legacy, may not be used

---

## 📝 Implementation Strategy

### Step 1: Update pipeline.m
Replace all `saveas()` calls with new sequential names:

#### Static Figure Names (Direct Replacement):
```matlab
Line 524:  'Fig0a_Basic_Faults.png'           → 'Fig1_Basic_Faults.png'
Line 585:  'Fig0b_Mixed_Faults.png'           → 'Fig2_Mixed_Faults.png'
Line 605:  'Fig1_Feature_Correlation.png'     → 'Fig3_Feature_Correlation.png'
Line 640:  'Fig2_Feature_Distributions.png'   → 'Fig4_Feature_Distributions.png'
Line 664:  'Fig3_tSNE_Clusters.png'           → 'Fig5_tSNE_Clusters.png'
Line 3717: 'Fig7_Class_Distribution.png'      → 'Fig6_Class_Distribution.png'
Line 3761: 'Fig8_Performance_Comparison.png'  → 'Fig13_Performance_Comparison.png'
Line 2724: 'Fig11_All_Models_Performance_Comparison.png' → 'Fig14_All_Models_Comparison.png'
Line 2886: 'Fig12_Feature_Importance_Comparison.png'    → 'Fig15_Feature_Importance.png'
Line 3354: 'Fig6_Learning_Curves.png'         → 'Fig16_Learning_Curves.png'
Line 3506: 'Fig9_Feature_Importance.png'      → 'Fig17_Old_Feature_Importance.png'
```

#### Dynamic Figure Names (Model-Specific):
```matlab
Line 2450: sprintf('Fig4_%s_Confusion_Matrix.png', modelFieldName)
           → sprintf('Fig%d_%s_Confusion_Matrix.png', figNum, modelFieldName)
           where figNum = 7 (SVM), 9 (RandomForest), 11 (NeuralNetwork)

Line 2568: sprintf('Fig5_%s_ROC_Curves.png', modelFieldName)
           → sprintf('Fig%d_%s_ROC_Curves.png', figNum+1, modelFieldName)
           where figNum+1 = 8 (SVM), 10 (RandomForest), 12 (NeuralNetwork)
```

#### Figures to REMOVE:
```matlab
Line 2004: saveas(fig4, ..., 'Fig4_Confusion_Matrix.png') → COMMENT OUT or REMOVE
Line 2249: saveas(fig5, ..., 'Fig5_ROC_Curves.png')       → COMMENT OUT or REMOVE
```

---

### Step 2: Update Technical_report.tex
Replace all `\includegraphics{}` and `\ref{fig:...}` with new names:

```latex
\includegraphics{Fig0a_Basic_Faults.png}     → \includegraphics{Fig1_Basic_Faults.png}
\includegraphics{Fig0b_Mixed_Faults.png}     → \includegraphics{Fig2_Mixed_Faults.png}
\includegraphics{Fig1_Feature_Correlation}   → \includegraphics{Fig3_Feature_Correlation}
\includegraphics{Fig2_Feature_Distributions} → \includegraphics{Fig4_Feature_Distributions}
\includegraphics{Fig3_tSNE_Clusters}         → \includegraphics{Fig5_tSNE_Clusters}
\includegraphics{Fig7_Class_Distribution}    → \includegraphics{Fig6_Class_Distribution}
\includegraphics{Fig8_Performance_Comparison}→ \includegraphics{Fig13_Performance_Comparison}

% NEW: Add model-specific figures
\includegraphics{Fig7_SVM_Confusion_Matrix}
\includegraphics{Fig8_SVM_ROC_Curves}
\includegraphics{Fig9_RandomForest_Confusion_Matrix}
\includegraphics{Fig10_RandomForest_ROC_Curves}
\includegraphics{Fig11_NeuralNetwork_Confusion_Matrix}
\includegraphics{Fig12_NeuralNetwork_ROC_Curves}
\includegraphics{Fig14_All_Models_Comparison}
\includegraphics{Fig15_Feature_Importance}
```

---

## ✅ Benefits of New Naming Scheme

1. **Sequential Clarity**: Fig1 through Fig15 (or Fig17) - easy to track
2. **Logical Grouping**:
   - Fig1-2: Raw data visualization
   - Fig3-5: Feature analysis
   - Fig6: Data split
   - Fig7-12: Model-specific performance (grouped by model)
   - Fig13-15: Comparative analysis
3. **LaTeX Compatibility**: Can directly copy figures from output directory to LaTeX directory
4. **Professional Presentation**: Standard figure numbering convention
5. **Eliminates Confusion**: No more "why does Fig4 have 3 variants?"

---

## 🔍 Implementation Checklist

- [x] Update pipeline.m static figure names (11 replacements) - **COMPLETED**
- [x] Update pipeline.m dynamic figure names (add figNum mapping) - **COMPLETED**
- [x] Mark old Fig4 and Fig5 as legacy (best model only) - **COMPLETED**
- [x] Update Technical_report.tex \includegraphics paths - **COMPLETED**
- [ ] Update Technical_report.tex figure labels and references - **DEFERRED TO PHASE 2**
- [ ] Add NEW figure sections in Technical Report for model-specific figures - **PHASE 2 TASK**
- [ ] Verify no broken references in LaTeX - **PENDING**
- [ ] Test pipeline run (optional - figures can be renamed manually) - **USER TASK**
- [x] Commit and push changes - **NEXT STEP**

---

## 🎓 Model-Specific Figure Number Assignment

### Logic for Dynamic Naming:
When iterating through models (SVM, RandomForest, NeuralNetwork), assign:

```matlab
modelFigureMap = struct();
modelFigureMap.SVM = struct('confusionMatrix', 7, 'rocCurves', 8);
modelFigureMap.RandomForest = struct('confusionMatrix', 9, 'rocCurves', 10);
modelFigureMap.NeuralNetwork = struct('confusionMatrix', 11, 'rocCurves', 12);

% Usage:
figNum = modelFigureMap.(modelFieldName).confusionMatrix;
figFilename = sprintf('Fig%d_%s_Confusion_Matrix.png', figNum, modelFieldName);
```

---

**Document Prepared By**: PFD Diagnostics Team
**Status**: Ready for Implementation
