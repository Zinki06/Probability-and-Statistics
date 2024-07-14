import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 예제 데이터
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([0, 0, 0, 1, 1])

# 로지스틱 회귀 모델 학습
model = LogisticRegression()
model.fit(X, Y)

# 예측 확률 계산
X_test = np.linspace(0, 6, 300).reshape(-1, 1)
Y_prob = model.predict_proba(X_test)[:, 1]

# 시각화
plt.scatter(X, Y, color='red', label='Data')
plt.plot(X_test, Y_prob, color='blue', label='Logistic Regression')
plt.xlabel('Study Hours')
plt.ylabel('Pass Probability')
plt.title('Logistic Regression: Study Hours vs Pass Probability')
plt.legend()
plt.grid()
plt.show()
