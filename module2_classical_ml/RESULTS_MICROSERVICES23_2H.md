# Module 2 Results — `microservices23-2h` 

## Results

| Rank | Model | Type | ROC-AUC | Precision | Recall | F1 | FPR | Predicted anomaly rate |
|---|---|---|---|---|---|---|---|---|
| 1 | LightGBM | Supervised | 0.9997 | 0.9978 | 0.9747 | 0.9861 | 0.0006 | 21.86% |
| 2 | Random Forest | Supervised | 0.9979 | 0.9897 | 0.9655 | 0.9775 | 0.0029 | 21.83% |
| 3 | GMM | Unsupervised | 0.9827 | 0.8525 | 0.9509 | 0.8990 | 0.0474 | 24.96% |
| 4 | Isolation Forest | Unsupervised | 0.9801 | 0.6981 | 0.9915 | 0.8193 | 0.1236 | 31.78% |
| 5 | Autoencoder | Unsupervised | 0.9491 | 0.9870 | 0.7809 | 0.8720 | 0.0030 | 17.70% |

(True anomaly rate: 22.37% for every row above — same held-out test split.)

### High-Degree Polynomial Linear Regression 
The High-Degree Polynomial Linear Regression model (`PolynomialRegressionAnomalyDetector`, degree=3, L2 Ridge alpha=1.0):

| Model | Type | ROC-AUC | Precision | Recall | F1 | FPR | Predicted anomaly rate |
|---|---|---|---|---|---|---|---|
| Polynomial Regression | Supervised | 0.9501 | 0.9986 | 0.8540 | 0.9207 | 0.0003 | 19.13% |

*(Evaluated at threshold 0.70 on the same held-out 621,040 test rows)*

## Randomly Shuffled 80/20 Train/Test Split Run


### Results Table (Random Shuffled 80/20 Split)

Evaluated on **621,040 test rows** at decision threshold **0.70**:

| Rank | Model | Paradigm | ROC-AUC | Precision | Recall | F1-Score | FPR | Actual Anomaly Rate | Predicted Anomaly Rate |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LightGBM** | Supervised | **0.99998** | 0.9951 | 0.9939 | **0.9945** | 0.0008 | **14.25%** | **14.24%** |
| 2 | **Random Forest** | Supervised | **0.99967** | 0.9921 | 0.9855 | **0.9888** | 0.0013 | **14.25%** | **14.16%** |
| 3 | **Polynomial Regression (Degree 3)** | Supervised | **0.99622** | **0.9983** | 0.8973 | **0.9451** | **0.0003** | **14.25%** | **12.81%** |
| 4 | **Unsupervised Ensemble (GMM+IF+AE)** | Unsupervised | **0.98683** | 0.8795 | 0.9115 | **0.8952** | 0.0208 | **14.25%** | **14.77%** |
| 5 | **Gaussian Mixture Model (GMM)** | Unsupervised | **0.98059** | 0.7943 | **0.9302** | **0.8569** | 0.0401 | **14.25%** | **16.69%** |
| 6 | **Isolation Forest** | Unsupervised | **0.97992** | 0.7587 | 0.9138 | **0.8290** | 0.0483 | **14.25%** | **17.17%** |
| 7 | **Deep Autoencoder** | Unsupervised | **0.97425** | 0.8908 | 0.8627 | **0.8765** | 0.0176 | **14.25%** | **13.80%** |

---

### Dataset 1 — 5% Anomaly Rate (Empirical 4.53%, 466,050 Held-Out Test Rows)

| Rank | Model | Paradigm | ROC-AUC | Precision | Recall | F1-Score | FPR | Actual Anomaly Rate | Predicted Anomaly Rate |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LightGBM** | Supervised | **0.99999** | 0.9901 | **0.9982** | **0.9942** | 0.0005 | **4.53%** | **4.57%** |
| 2 | **Random Forest** | Supervised | **0.99992** | **0.9969** | 0.9835 | **0.9902** | **0.0001** | **4.53%** | **4.47%** |
| 3 | **Polynomial Regression (Degree 3)** | Supervised | **0.99533** | 0.9968 | 0.9211 | **0.9575** | **0.0001** | **4.53%** | **4.19%** |
| 4 | **Unsupervised Ensemble (GMM+IF+AE)** | Unsupervised | **0.99282** | 0.6904 | 0.9385 | **0.7955** | 0.0200 | **4.53%** | **6.16%** |
| 5 | **Deep Autoencoder** | Unsupervised | **0.98136** | 0.7221 | 0.9172 | **0.8081** | 0.0167 | **4.53%** | **5.75%** |
| 6 | **Gaussian Mixture Model (GMM)** | Unsupervised | **0.98630** | 0.5755 | 0.9408 | **0.7141** | 0.0329 | **4.53%** | **7.41%** |
| 7 | **Isolation Forest** | Unsupervised | **0.99014** | 0.3881 | **0.9843** | **0.5567** | 0.0737 | **4.53%** | **11.49%** |

---

### Dataset 2 — 10% Anomaly Rate (Empirical 8.10%, 465,956 Held-Out Test Rows)

| Rank | Model | Paradigm | ROC-AUC | Precision | Recall | F1-Score | FPR | Actual Anomaly Rate | Predicted Anomaly Rate |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LightGBM** | Supervised | **0.99999** | **0.9933** | **0.9952** | **0.9942** | 0.0006 | **8.10%** | **8.12%** |
| 2 | **Random Forest** | Supervised | **0.99975** | 0.9918 | 0.9816 | **0.9866** | 0.0007 | **8.10%** | **8.02%** |
| 3 | **Polynomial Regression (Degree 3)** | Supervised | **0.99518** | **0.9977** | 0.8914 | **0.9416** | **0.0002** | **8.10%** | **7.24%** |
| 4 | **Unsupervised Ensemble (GMM+IF+AE)** | Unsupervised | **0.98479** | 0.7803 | 0.9083 | **0.8395** | 0.0225 | **8.10%** | **9.43%** |
| 5 | **Deep Autoencoder** | Unsupervised | **0.97370** | 0.7860 | 0.8878 | **0.8338** | 0.0213 | **8.10%** | **9.15%** |
| 6 | **Gaussian Mixture Model (GMM)** | Unsupervised | **0.97324** | 0.6707 | 0.9070 | **0.7711** | 0.0393 | **8.10%** | **10.95%** |
| 7 | **Isolation Forest** | Unsupervised | **0.98327** | 0.5834 | **0.9547** | **0.7242** | 0.0601 | **8.10%** | **13.26%** |

---



### Dataset 3 — 18% Target Rate (Empirical 15.96%, 465,955 Held-Out Test Rows)

| Rank | Model | Paradigm | ROC-AUC | Precision | Recall | F1-Score | FPR | Actual Anomaly Rate | Predicted Anomaly Rate |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LightGBM** | Supervised | **0.99998** | **0.9957** | **0.9949** | **0.9953** | 0.0008 | **15.96%** | **15.95%** |
| 2 | **Random Forest** | Supervised | **0.99975** | 0.9952 | 0.9860 | **0.9906** | 0.0009 | **15.96%** | **15.81%** |
| 3 | **Polynomial Regression (Degree 3)** | Supervised | **0.99487** | **0.9987** | 0.9044 | **0.9492** | **0.0002** | **15.96%** | **14.46%** |
| 4 | **Unsupervised Ensemble (GMM+IF+AE)** | Unsupervised | **0.98529** | 0.8727 | 0.9370 | **0.9037** | 0.0260 | **15.96%** | **17.14%** |
| 5 | **Deep Autoencoder** | Unsupervised | **0.97552** | 0.9049 | 0.8990 | **0.9020** | **0.0179** | **15.96%** | **15.86%** |
| 6 | **Isolation Forest** | Unsupervised | **0.98263** | 0.7707 | **0.9542** | **0.8527** | 0.0539 | **15.96%** | **19.76%** |
| 7 | **Gaussian Mixture Model (GMM)** | Unsupervised | **0.97625** | 0.7534 | 0.9474 | **0.8393** | 0.0589 | **15.96%** | **20.07%** |

---

### Dataset 4 — 25% Target Rate (Empirical 18.46%, 465,954 Held-Out Test Rows)

| Rank | Model | Paradigm | ROC-AUC | Precision | Recall | F1-Score | FPR | Actual Anomaly Rate | Predicted Anomaly Rate |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LightGBM** | Supervised | **0.99995** | **0.9960** | **0.9906** | **0.9933** | 0.0009 | **18.46%** | **18.37%** |
| 2 | **Random Forest** | Supervised | **0.99936** | 0.9892 | 0.9836 | **0.9864** | 0.0024 | **18.46%** | **18.36%** |
| 3 | **Polynomial Regression (Degree 3)** | Supervised | **0.99479** | **0.9992** | 0.8758 | **0.9334** | **0.0002** | **18.46%** | **16.18%** |
| 4 | **Deep Autoencoder** | Unsupervised | **0.97110** | 0.9218 | 0.8929 | **0.9071** | **0.0172** | **18.46%** | **17.89%** |
| 5 | **Unsupervised Ensemble (GMM+IF+AE)** | Unsupervised | **0.98337** | 0.8954 | 0.9163 | **0.9057** | 0.0242 | **18.46%** | **18.90%** |
| 6 | **Gaussian Mixture Model (GMM)** | Unsupervised | **0.97291** | 0.7984 | 0.9248 | **0.8570** | 0.0529 | **18.46%** | **21.39%** |
| 7 | **Isolation Forest** | Unsupervised | **0.98062** | 0.7783 | **0.9331** | **0.8487** | 0.0602 | **18.46%** | **22.14%** |

---

### Dataset 5 — 33% Target Rate (Empirical 25.55%, 466,052 Held-Out Test Rows)

| Rank | Model | Paradigm | ROC-AUC | Precision | Recall | F1-Score | FPR | Actual Anomaly Rate | Predicted Anomaly Rate |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LightGBM** | Supervised | **0.99994** | **0.9984** | **0.9907** | **0.9945** | 0.0005 | **25.55%** | **25.36%** |
| 2 | **Random Forest** | Supervised | **0.99955** | 0.9941 | 0.9853 | **0.9897** | 0.0020 | **25.55%** | **25.33%** |
| 3 | **Polynomial Regression (Degree 3)** | Supervised | **0.99678** | **0.9986** | 0.9106 | **0.9526** | **0.0005** | **25.55%** | **23.30%** |
| 4 | **Deep Autoencoder** | Unsupervised | **0.97528** | 0.9440 | 0.9121 | **0.9278** | **0.0186** | **25.55%** | **24.69%** |
| 5 | **Unsupervised Ensemble (GMM+IF+AE)** | Unsupervised | **0.98496** | 0.9268 | **0.9302** | **0.9285** | 0.0252 | **25.55%** | **25.65%** |
| 6 | **Isolation Forest** | Unsupervised | **0.97999** | 0.8519 | **0.9372** | **0.8925** | 0.0559 | **25.55%** | **28.11%** |
| 7 | **Gaussian Mixture Model (GMM)** | Unsupervised | **0.97752** | 0.8472 | 0.9335 | **0.8882** | 0.0578 | **25.55%** | **28.16%** |

---

### Results Table (52% Target, Empirical 42.74%, 466,048 Held-Out Test Rows)

| Rank | Model | Paradigm | ROC-AUC | Precision | Recall | F1-Score | FPR | Actual Anomaly Rate | Predicted Anomaly Rate |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LightGBM** | Supervised | **0.99998** | **0.9996** | **0.9954** | **0.9975** | **0.0003** | **42.74%** | **42.56%** |
| 2 | **Random Forest** | Supervised | **0.99974** | 0.9975 | 0.9920 | 0.9948 | 0.0018 | **42.74%** | **42.50%** |
| 3 | **Unsupervised Ensemble (GMM+IF+AE)** | Unsupervised | **0.98796** | 0.9731 | 0.9169 | 0.9442 | 0.0189 | **42.74%** | **40.27%** |
| 4 | **Polynomial Regression (Degree 3)** | Supervised | **0.99806** | 0.9993 | 0.9123 | 0.9538 | 0.0004 | **42.74%** | **39.01%** |
| 5 | **Autoencoder** | Unsupervised | **0.97819** | 0.9734 | 0.8838 | 0.9264 | 0.0180 | **42.74%** | **38.80%** |
| 6 | **Gaussian Mixture Model (GMM)** | Unsupervised | **0.98055** | 0.9722 | 0.9261 | 0.9486 | 0.0198 | **42.74%** | **40.71%** |
| 7 | **Isolation Forest** | Unsupervised | **0.97481** | 0.9098 | 0.9054 | 0.9076 | 0.0670 | **42.74%** | **42.53%** |

(Rank ordered by ROC-AUC; note F1 and precision orderings differ — see findings below.)

### Results Table (75% Target, Empirical 52.76%, 466,230 Held-Out Test Rows)

| Rank | Model | Paradigm | ROC-AUC | Precision | Recall | F1-Score | FPR | Actual Anomaly Rate | Predicted Anomaly Rate |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LightGBM** | Supervised | **0.99990** | **0.9994** | **0.9913** | **0.9953** | **0.0007** | **52.76%** | **52.33%** |
| 2 | **Random Forest** | Supervised | **0.99940** | 0.9985 | 0.9850 | 0.9917 | 0.0017 | **52.76%** | **52.05%** |
| 3 | **Polynomial Regression (Degree 3)** | Supervised | **0.99511** | 0.9990 | 0.9122 | 0.9536 | 0.0010 | **52.76%** | **48.18%** |
| 4 | **Unsupervised Ensemble (GMM+IF+AE)** | Unsupervised | **0.97523** | 0.9747 | 0.8896 | 0.9302 | 0.0258 | **52.76%** | **48.16%** |
| 5 | **Gaussian Mixture Model (GMM)** | Unsupervised | **0.97063** | 0.9459 | 0.9145 | 0.9299 | 0.0584 | **52.76%** | **51.01%** |
| 6 | **Autoencoder** | Unsupervised | **0.96844** | 0.9849 | 0.8725 | 0.9253 | 0.0149 | **52.76%** | **46.74%** |
| 7 | **Isolation Forest** | Unsupervised | **0.94357** | 0.9439 | 0.7116 | 0.8114 | 0.0473 | **52.76%** | **39.78%** |

(Rank ordered by ROC-AUC; note F1 and precision orderings differ — see findings below.)
