import re

import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class Solution:
    min_goodness = 0

    def __init__(self, path, min_goodness: int):
        self.min_goodness = min_goodness
        self.df = pandas.read_csv(path)

    def transform(self, a: str):
        a = re.sub(r"(</.*>)", "", a)
        a = re.sub(r"(<.*>)", "", a)
        return a

    def calculate(self):
        total_rows = self.df.shape[0]

        self.vectorizer = TfidfVectorizer()

        train_df = self.df[0:int(total_rows * 0.75)]
        test_df = self.df[int(total_rows * 0.75):]

        train_bodies = []
        train_goodness = []

        for index, row in train_df.iterrows():
            if row["Score"] > self.min_goodness:
                train_goodness.append(1)
            else:
                train_goodness.append(0)
            train_bodies.append(self.transform(row["Body"]))

        X_train = self.vectorizer.fit_transform(train_bodies)

        self.classifier = LogisticRegression()
        self.classifier.fit(X_train, train_goodness)

        return self.predict(test_df)

    def predict(self, test_df):
        test_bodies = []
        test_goodness = []
        for index, row in test_df.iterrows():
            test_bodies.append(self.transform(row["Body"]))
            if row["Score"] > self.min_goodness:
                test_goodness.append(1)
            else:
                test_goodness.append(0)
        X_test = self.vectorizer.transform(test_bodies)
        x_predict = self.classifier.predict(X_test)
        cnt = 0
        for i in range(0, len(x_predict)):
            if test_goodness[i] == x_predict[i]:
                cnt += 1
        print("Predicted values : ", cnt / len(test_goodness), sep=" ")
        return cnt / len(test_goodness)


lol = Solution("C:\\ai\\Answers.csv", 2)
print(lol.calculate())
