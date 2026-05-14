# RAMDEOBABA UNIVERSITY, NAGPUR  
## DEPARTMENT OF DATA SCIENCE  

### PROJECT REPORT  

**Machine Learning based Multimodal Depression Detection System**  

**Submitted by:**  
Rishil (Student Roll Number)  

**Under the Guidance of:**  
Prof. Guide Name / Dr. Guide Name  

**Department of Data Science**  
**Ramdeobaba University, Nagpur**  

**Academic Year: 2025-2026**  

---

# Certificate  

This is to certify that the project report entitled **"Machine Learning based Multimodal Depression Detection System"** submitted by **Rishil (Roll Number)** in partial fulfillment of the requirements for the degree of Bachelor of Technology in Data Science at Ramdeobaba University, Nagpur, is a record of bonafide work carried out by him under my supervision and guidance.  

The project work has not been submitted elsewhere for any other degree or diploma.  

**Guide:**  
Prof. Guide Name / Dr. Guide Name  
Department of Data Science  
Ramdeobaba University, Nagpur  

**Head of Department:**  
Prof. Head Name  
Department of Data Science  
Ramdeobaba University, Nagpur  

**Date:** May 14, 2026  

---

# Declaration  

I, **Rishil (Roll Number)**, hereby declare that the project report entitled **"Machine Learning based Multimodal Depression Detection System"** submitted to Ramdeobaba University, Nagpur, for the partial fulfillment of the requirements for the degree of Bachelor of Technology in Data Science, is my original work and has not been submitted for any other degree or diploma.  

All the information and data presented in this report are true to the best of my knowledge and belief.  

**Date:** May 14, 2026  
**Place:** Nagpur  

**Rishil**  
(Roll Number)  

---

# Acknowledgement  

I would like to express my sincere gratitude to my project guide, **Prof. Guide Name / Dr. Guide Name**, for his/her invaluable guidance, encouragement, and support throughout the project.  

I am also thankful to the Head of the Department, **Prof. Head Name**, and all faculty members of the Department of Data Science at Ramdeobaba University, Nagpur, for providing the necessary facilities and resources.  

Special thanks to my family and friends for their constant support and motivation.  

**Rishil**  

---

# Abstract  

Depression is a prevalent mental health disorder affecting millions worldwide. Early detection and intervention are crucial for effective treatment. This project develops a multimodal depression detection system using machine learning techniques that analyze text, audio, and visual modalities from patient interviews.  

The system employs advanced feature extraction methods, including SBERT for text, MFCCs for audio, and facial expression analysis for visual data. A late fusion approach combines these modalities with dynamic weighting based on cross-validation AUC scores. The model achieves an AUC of 0.655 on the E-DAIC dataset.  

A web application integrates the model with PHQ-8 questionnaire for comprehensive assessment. The system ensures privacy, clinical relevance, and real-time analysis.  

**Keywords:** Multimodal AI, Depression Detection, Machine Learning, Fusion Models, Mental Health  

---

# Table of Contents  

1. [Introduction](#chapter-1-introduction)  
   1.1 Overview  
   1.2 Objectives  
   1.3 Problem Statement  
   1.4 Scope of the Project  

2. [Literature Review](#chapter-2-literature-review)  

3. [System Requirements](#chapter-3-system-requirements)  
   3.1 Hardware Requirements  
   3.2 Software Requirements  
   3.3 Functional Requirements  
   3.4 Non-Functional Requirements  

4. [System Design](#chapter-4-system-design)  
   4.1 Architecture Design  
   4.2 Database Design  
   4.3 User Interface Design  

5. [Methodology](#chapter-5-methodology)  
   5.1 Data Collection and Preprocessing  
   5.2 Feature Extraction  
   5.3 Model Development  
   5.4 Evaluation Metrics  

6. [Implementation](#chapter-6-implementation)  
   6.1 Development Environment  
   6.2 Code Structure  
   6.3 Key Algorithms  

7. [Results and Impact](#chapter-7-results-and-impact)  
   7.1 Performance Metrics  
   7.2 Visualizations  
   7.3 Clinical Impact  

8. [Discussion](#chapter-8-discussion)  

9. [Conclusion and Future Scope](#chapter-9-conclusion-and-future-scope)  

[References](#references)  

[Appendices](#appendices)  

---

# List of Figures  

Figure 1.1: Multimodal Fusion Architecture  
Figure 4.1: System Architecture Diagram  
Figure 5.1: Feature Extraction Pipeline  
Figure 6.1: Web Application Interface  
Figure 7.1: ROC Curves  
Figure 7.2: Confusion Matrix  
Figure 7.3: Calibration Curve  

---

# List of Tables  

Table 3.1: Hardware Requirements  
Table 3.2: Software Requirements  
Table 5.1: Dataset Statistics  
Table 7.1: Performance Metrics  
Table 7.2: Confusion Matrix  

---

# Chapter 1: Introduction  

## 1.1 Overview  

Depression detection is critical in mental health care. Traditional methods rely on clinical interviews and questionnaires like PHQ-8. This project leverages multimodal AI to analyze patient interviews across text, audio, and visual channels for automated depression screening.  

The system uses machine learning models trained on the E-DAIC dataset, combining linguistic analysis, vocal biomarkers, and facial expressions. A web application provides real-time assessment integrated with PHQ-8 scoring.  

## 1.2 Objectives  

- Develop a multimodal depression detection model  
- Achieve high accuracy using fusion techniques  
- Create a user-friendly web interface  
- Ensure clinical relevance and privacy  

## 1.3 Problem Statement  

Manual depression diagnosis is time-consuming and subjective. Automated systems can provide objective, scalable screening. The challenge is integrating multiple modalities while preventing data leakage and maintaining clinical validity.  

## 1.4 Scope of the Project  

The project includes:  
- Multimodal feature extraction  
- Machine learning model development  
- Web application deployment  
- Evaluation on E-DAIC dataset  

---

# Chapter 2: Literature Review  

Recent studies show multimodal approaches outperform unimodal methods in depression detection. Research by Williamson et al. (2016) on E-DAIC dataset demonstrates the value of combining modalities. Advances in deep learning and fusion techniques have improved accuracy.  

Key papers:  
- "Multimodal Depression Detection" (IEEE, 2020)  
- "Audio-Visual Emotion Recognition" (ACM, 2019)  
- "Clinical Applications of AI in Mental Health" (Nature, 2021)  

---

# Chapter 3: System Requirements  

## 3.1 Hardware Requirements  

| Component | Specification |  
|-----------|---------------|  
| Processor | Intel i5 or equivalent |  
| RAM | 8 GB minimum |  
| Storage | 50 GB free space |  
| GPU | Optional for training |  

## 3.2 Software Requirements  

| Software | Version |  
|----------|---------|  
| Python | 3.8+ |  
| Flask | 2.0+ |  
| scikit-learn | 1.0+ |  
| TensorFlow | 2.0+ |  

## 3.3 Functional Requirements  

- Feature extraction from multimodal data  
- Model training and evaluation  
- Web-based prediction interface  
- PHQ-8 integration  

## 3.4 Non-Functional Requirements  

- Response time < 5 seconds  
- Accuracy > 60% AUC  
- Privacy compliance  
- Scalability  

---

# Chapter 4: System Design  

## 4.1 Architecture Design  

The system follows a modular architecture with separate components for feature extraction, model training, and web deployment.  

## 4.2 Database Design  

Uses CSV files for features and labels. No relational database required for this prototype.  

## 4.3 User Interface Design  

Web interface with interview upload, real-time analysis, and results display.  

---

# Chapter 5: Methodology  

## 5.1 Data Collection and Preprocessing  

Dataset: E-DAIC (219 samples, 65 depressed). Features: text (96), audio (1218), visual (214).  

## 5.2 Feature Extraction  

- Text: SBERT embeddings + TF-IDF  
- Audio: MFCCs + prosodic features  
- Visual: Facial expressions via face-api.js  

## 5.3 Model Development  

Ensemble models with SMOTE, PCA/SelectKBest, late fusion with AUC-based weights.  

## 5.4 Evaluation Metrics  

AUC, F1-score, confusion matrix, calibration.  

---

# Chapter 6: Implementation  

## 6.1 Development Environment  

Python with VS Code, virtual environment.  

## 6.2 Code Structure  

- main.py: Training pipeline  
- app.py: Web application  
- src/: Feature extraction modules  

## 6.3 Key Algorithms  

Voting classifiers, cross-validation, Youden's J threshold optimization.  

---

# Chapter 7: Results and Impact  

## 7.1 Performance Metrics  

Fusion AUC: 0.655, Accuracy: 62.6%, F1: 0.506  

## 7.2 Visualizations  

ROC curves, confusion matrices, calibration plots.  

## 7.3 Clinical Impact  

Provides objective screening tool for clinicians.  

---

# Chapter 8: Discussion  

The model shows promise but requires larger datasets. Multimodal fusion improves over single modalities.  

---

# Chapter 9: Conclusion and Future Scope  

Successfully developed multimodal depression detection system. Future work: larger datasets, real-time video analysis, clinical validation.  

---

# References  

1. Williamson, J. R., et al. (2016). Multimodal depression detection.  
2. IEEE Transactions on Affective Computing.  
3. ACM Multimedia Conference proceedings.  

---

# Appendices  

Appendix A: Code Snippets  
Appendix B: Dataset Details  
Appendix C: Screenshots