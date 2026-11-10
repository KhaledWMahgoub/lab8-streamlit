import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('16P.csv', encoding='latin1')
df = df.drop('Response Id', axis=1).drop_duplicates().reset_index(drop=True)

y = df.pop("Personality")
questions = list(df.columns)

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

sv = SVC(C=1, gamma=0.01, kernel='rbf')

sv.fit(x_train,y_train)


sv.score(x_train,y_train)
sv.score(x_test,y_test)


joblib.dump(sv, 'svc_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(questions, 'questions.pkl')
