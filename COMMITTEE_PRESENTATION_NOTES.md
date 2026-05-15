# SENTIRA Committee Presentation Notes

## Positioning

Present SENTIRA as a research-grade multimodal screening prototype, not as a diagnostic medical device. The strongest professional framing is:

- It combines text, audio, visual behavior, and PHQ-8 questionnaire signals.
- It uses leakage-aware cross-validation, class-imbalance handling, and clinical metrics.
- It prioritizes responsible screening support: high sensitivity, interpretability, and clear limitations.

## What To Emphasize

1. **Clinical Responsibility**
   - The system is a screening aid, not a diagnosis.
   - False negatives are more concerning than false positives, so sensitivity/recall is reported alongside accuracy.
   - PHQ-8 remains visible as a validated clinical questionnaire signal.

2. **Validation Quality**
   - Use stratified cross-validation because the dataset is small and imbalanced.
   - Keep scaling, PCA, feature selection, and SMOTE inside the training fold to avoid leakage.
   - Report accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, macro-F1, AUC, and average precision.

3. **Multimodal Design**
   - Text is the strongest modality in the current extracted-feature setup.
   - Audio and visual signals are kept because they add behavioral context, but fusion weights prevent weak modalities from dominating.
   - The final result should be explained as risk estimation, not a binary clinical judgment.

4. **Interpretability**
   - Show modality contribution weights.
   - Show confusion matrix and probability distribution by class.
   - Mention clinically meaningful text markers: negative affect, first-person singular pronouns, absolutist words, response brevity, and sentiment variation.

5. **Known Limitations**
   - Dataset size is limited.
   - Audio and visual quality depend on recording conditions.
   - Live web inference uses browser-compatible approximations for audio/visual behavior.
   - More diverse external validation is required before any real clinical deployment.

## Recommended Slide Flow

1. Problem and motivation.
2. Dataset and label definition: PHQ-8 >= 10 as depressed.
3. Architecture: text, audio, visual, PHQ-8, fusion.
4. Feature extraction examples per modality.
5. Leakage-safe training and validation protocol.
6. Results table with clinical metrics, not only accuracy.
7. Confusion matrix and ROC/PR plots.
8. Demo of Flask web app.
9. Limitations, ethics, and future work.

## Future Work That Sounds Professional

- External validation on another clinical interview dataset.
- Model calibration and decision-curve analysis.
- Explainability with feature attribution for text and tabular models.
- Fairness analysis across demographic groups, if metadata is available.
- Stronger audio/visual inference alignment between offline training and browser demo.
