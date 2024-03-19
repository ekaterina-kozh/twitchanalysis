from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('musae_RU_target.csv')
x = df.drop('partner', axis=1)
y = df['partner']

#Разделим данные на тестовые и тренировачные
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y,  test_size=0.2, random_state=42)

#Обучим модель
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x, y)

ypred = clf.predict(X_test)
accuracy = accuracy_score(y_test, ypred)
print(f'Оценка точности пассивно-агрессивного классификатора: {round(accuracy*100,2)}%')

#Построим матрицу ошибок
cf_matrix = confusion_matrix(y_test,ypred)
ax = sns.heatmap(cf_matrix/ cf_matrix.sum(axis=1, keepdims=True), annot=True, cmap='Blues', fmt='.4f', square=True)

ax.set_title('Матрица ошибок\n\n');
ax.set_xlabel('\nПредсказанные метки')
ax.set_ylabel('Истинные метки')
