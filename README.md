# Heart Disease Prediction App

This is a machine learning-based web app that predicts the **risk of heart disease** based on clinical parameters using a trained ML model.


## 🚀 Live App
[🔗 Click here to try the app](https://heart-disease-prediction-nishu.streamlit.app/)  

![](/images/Demo_1.png)
![](/images/Demo_2.png)

---

## Project Overview

- Dataset: UCI Heart Disease (heart_cleveland.csv)
- ML Model: Logistic Regression
- Libraries: pandas, scikit-learn, matplotlib, seaborn, streamlit
- Deployed with: Streamlit

---

## Model Training

Two models were trained:

| Model               | Accuracy Score  |
|---------------------|-----------------|
| Decision Tree       | **0.77** (Best) |
| Logistic Regression | 0.73            |

The final deployed model is **Decision Tree** due to its balance of accuracy and interpretability.

---

## How to Run Locally

```bash
git clone https://github.com/NishuMehta/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
pip install -r requirements.txt
streamlit run app/main.py


## Author

- [Nishu Mehta](https://github.com/NishuMehta)

## Project Link

[GitHub Repository](https://github.com/NishuMehta/Heart-Disease-Prediction)