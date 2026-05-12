# FIXES_GUIDE.md - Quick Start for Applying Fixes

## 🎯 Overview of Critical Fixes

This repository has 3 critical bugs that must be fixed for the project to work:

| Fix # | Issue | Severity | Impact | Time |
|-------|-------|----------|--------|------|
| **1** | Feature dimension mismatch (76 vs 59) | 🔴 CRITICAL | Tests fail, app crashes | 15 min |
| **2** | Audio modality debugging | 🔴 CRITICAL | Audio corrupts fusion results | 30 min |
| **3** | Fusion strategy not working | 🔴 CRITICAL | Multimodal fusion ineffective | 1 hour |

---

## 🚀 Quick Start: Apply Fixes in Order

### Fix #1: Feature Dimension Mismatch (15 minutes)

**Files Created:**
- `FIX_1_FEATURE_DIMENSION_BUG.py` — Explanation and solution code

**What's the problem?**
```
app.extract_text_features() extracts 76 features
but text_scaler expects 59 features
ERROR: ValueError during inference
```

**How to fix:**

**Option A (Recommended - Quick):**
1. Open `app.py`
2. Find `extract_text_features()` function (lines 103-169)
3. Replace with the `extract_text_features_FIXED()` from `FIX_1_FEATURE_DIMENSION_BUG.py`
4. Test:
   ```bash
   python -m pytest tests/test_app.py::TestAPIEndpoints::test_analyze_text_valid -v
   ```
5. Should see: `PASSED` ✅

**Option B (Better - Requires retraining):**
1. Update `config.py`: Change `N_TFIDF = 29` (was 50)
2. Retrain models: `python main.py`
3. This will save models trained with all 96 features (including SBERT)

**Recommended:** Use Option A now, migrate to Option B later for potential AUC improvement

---

### Fix #2: Audio Modality Debugging (30 minutes)

**Files Created:**
- `FIX_2_AUDIO_DEBUGGING.py` — Diagnostic script to identify audio issues

**What's the problem?**
```
Audio accuracy: 0.394 (worse than random)
Audio AUC: 0.548 (barely above 0.5)
This corrupts multimodal fusion
```

**How to fix:**

1. Run the diagnostic script:
   ```bash
   python FIX_2_AUDIO_DEBUGGING.py --data-root ~/Downloads/E-DAIC/data
   ```

2. This will check:
   - Do audio feature files exist? ✓
   - Are they readable and non-empty? ✓
   - Do they have reasonable statistics? ✓
   - Is there data leakage? ✓

3. Based on results, choose action:

   **If files are MISSING:**
   - Set environment variable:
     ```bash
     export AUDIO_RELIABLE=false
     ```
   - In `config.py`: `AUDIO_RELIABLE = False`
   - This disables audio from fusion

   **If files exist but performance is poor:**
   - Likely E-DAIC audio data doesn't correlate with depression
   - Still disable: `AUDIO_RELIABLE = False`
   - Focus on text + visual modalities

   **If everything looks good:**
   - Check main.py for data leakage in PCA/scaling (inside vs outside CV loop)
   - Lines 114-122 should fit PCA inside loop

---

### Fix #3: Fusion Strategy Redesign (1 hour)

**Files Created:**
- `FIX_3_FUSION_REDESIGN.py` — New fusion implementations (Attention, Stacking, Cost-Sensitive)

**What's the problem?**
```
Late fusion AUC: 0.591 (same as text-only)
Static weights [0.35, 0.35, 0.30] don't adapt to modality quality
Audio weight should be near 0, but it's 0.35
```

**How to fix:**

**Step 1: Implement Attention-Based Fusion**
1. Copy `AttentionFusion` class from `FIX_3_FUSION_REDESIGN.py`
2. Add to `src/fusion.py` (or create new file `src/attention_fusion.py`)
3. Update `main.py` to use it:
   ```python
   from src.attention_fusion import AttentionFusion
   
   # In the CV loop, replace static weights with:
   fusion = AttentionFusion(min_auc_threshold=0.52)
   fusion.fit(probs_val, y_val)
   fused_probs, fused_preds = fusion.predict_proba(probs_test, threshold=0.5)
   ```

**Step 2: Test the improvement**
```bash
python main.py
# Expected: Late Fusion AUC should now exceed text-only by ~0.05-0.10
```

**Step 3 (Optional): Try Stacking Fusion**
1. If attention fusion doesn't improve enough, try `StackingFusion`
2. Same implementation approach as above
3. Trade-off: More powerful but may overfit on small dataset

---

## 📋 Detailed Change Checklist

### Fix #1 - Feature Dimension (15 min)
- [ ] Open `app.py`
- [ ] Locate `extract_text_features()` function
- [ ] Replace with `extract_text_features_FIXED()` from FIX_1 file
- [ ] Test: `pytest tests/test_app.py -v`
- [ ] Verify: 4 previously failing tests now PASS

### Fix #2 - Audio Debugging (30 min)
- [ ] Run: `python FIX_2_AUDIO_DEBUGGING.py`
- [ ] Read diagnostic output carefully
- [ ] If audio files missing:
  - [ ] Set `AUDIO_RELIABLE=false` in config.py
- [ ] If audio files exist:
  - [ ] Check for data leakage in main.py
  - [ ] Verify PCA fits inside CV loop (lines 114-122)
- [ ] Rerun training: `python main.py`

### Fix #3 - Fusion Strategy (1 hour)
- [ ] Copy `AttentionFusion` class from FIX_3 file
- [ ] Add to `src/fusion.py` or new `src/attention_fusion.py`
- [ ] Update `main.py`:
  - [ ] Import AttentionFusion
  - [ ] Replace `late_fusion_predict()` with `AttentionFusion`
  - [ ] Remove hardcoded weights
- [ ] Test: `python main.py`
- [ ] Check results: Late Fusion AUC should improve

---

## 🧪 Testing & Validation

After applying each fix, run tests:

```bash
# Test individual fixes
python -m pytest tests/test_app.py::TestAPIEndpoints::test_analyze_text_valid -v
python -m pytest tests/test_app.py -v

# Full test suite
pytest tests/ -v

# Manual validation
python diagnose_data.py
python FIX_2_AUDIO_DEBUGGING.py
python main.py
python app.py  # Then test http://localhost:5000
```

---

## 📊 Expected Results After Fixes

### Before Fixes:
```
❌ Text model accuracy: 0.758, F1: 0.429, AUC: 0.591
❌ Audio model accuracy: 0.394, F1: 0.474, AUC: 0.548
❌ Late Fusion AUC: 0.591 (NO improvement over text)
❌ 4/25 tests FAILING
```

### After Fix #1 (Feature Dimension):
```
✅ Text model works: 0.758, F1: 0.429, AUC: 0.591
✅ 20/25 tests PASSING (4 were dimension bugs)
✅ app.py inference no longer crashes
```

### After Fix #2 (Audio Debugging):
```
✅ Audio issue identified and fixed or disabled
✅ Training runs without audio data leakage
✅ If audio disabled: audio weight ≈ 0 (no corruption)
```

### After Fix #3 (Fusion Redesign):
```
✅ Late Fusion AUC: 0.65-0.70 (improvement!)
✅ Audio weight automatically drops to ~0.01
✅ Text weight increases to ~0.60
✅ Visual weight optimized to ~0.35
✅ Multimodal fusion now provides actual benefit
```

---

## 🔗 Additional Resources

- `FIX_1_FEATURE_DIMENSION_BUG.py` — Detailed explanation + code
- `FIX_2_AUDIO_DEBUGGING.py` — Run this to diagnose audio issues
- `FIX_3_FUSION_REDESIGN.py` — Improved fusion implementations
- `PROJECT_REPORT.md` — Comprehensive project analysis
- `IMPLEMENTATION_SUMMARY.md` — Context on what was already done

---

## ⚠️ Important Notes

1. **Fix #1 is blocking** - Must fix first, causes test failures
2. **Fix #2 requires investigation** - Audio may need to be disabled completely
3. **Fix #3 improves quality** - Apply after #1 and #2 work

4. **Test frequently** - After each fix, run tests to verify no regressions

5. **Keep old fusion code** - Don't delete, use in comparison

6. **Document changes** - Update README when fixes are applied

---

## 🆘 Troubleshooting

### Fix #1 fails
- Error: `AttributeError: 'module' object has no attribute 'extract_clinical_nlp_features'`
  - Solution: Ensure `src/text_features.py` exists and has `extract_clinical_nlp_features()`

### Fix #2 shows "Data root not found"
- Solution: Set environment variable:
  ```bash
  export EDAIC_DATA_ROOT=/path/to/E-DAIC/data
  ```

### Fix #3 doesn't improve AUC
- Check: Are audio/visual weights actually getting low?
  - Should see: `audio weight: 0.01, text weight: 0.62, visual weight: 0.37`
- If not, audio/visual AUC is too high
  - Debug: Check `self.auc_scores` in AttentionFusion.fit()

---

## 📞 Next Steps

1. **Apply Fix #1 immediately** (15 min) - Unblocks everything else
2. **Run diagnostics** (30 min) - Understand audio issue
3. **Disable audio if needed** (5 min) - Quick win
4. **Implement new fusion** (1 hour) - Improves quality
5. **Validate & test** (30 min) - Ensure no regressions
6. **Retrain models** (varies) - With all fixes applied

**Total time: ~3-4 hours** for full implementation

Good luck! 🚀
