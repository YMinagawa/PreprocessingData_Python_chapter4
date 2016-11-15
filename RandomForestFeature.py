from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#ランダムフォレストモデルで特徴量の重要度にアクセスする(4章-6)

# wineデータセットを読み込む
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
# 列名の設定
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                   'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
# クラスラベルを表示
print('Class labels', np.unique(df_wine['Class label']))

# wineデータセットの先頭5行を表示する
print(df_wine.head())

# 特徴量(X)とクラスラベル(y)を別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values
# トレーニングデータとテストデータに分割
# 全体の30%をテストデータにする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 特徴量のスケーリング(特徴量の尺度を揃える)
# 決定木やランダムフォレストは特徴量のスケーリングに配慮する必要性のない数少ない機械学習アルゴリズムである
# だが2章で勾配降下法の最適化アルゴリズムを実装したときに示したように、
# 機械学習の最適化では特徴量の尺度が同じにした方がはるかにうまく動作する
# 正規化[0,1]と標準化(mean=0, sd=1)が存在する。標準化の方が実用的？

# 正規化 x_norm = (x(i)-xmin)/(xmax-xmin)
# min-maxスケーリングのインスタンスを生成
mms = MinMaxScaler()
# トレーニングデータをスケーリング
X_train_norm = mms.fit_transform(X_train)
# テストデータをスケーリング
X_test_norm = mms.fit_transform(X_test)

# 標準化 x_std = (x(i)-mean)/sd
# 標準化のインスタンスを生成(平均=0、標準偏差=1に変換)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

feat_labels = df_wine.columns[1:]

# ランダムフォレストオブジェクトの生成
# (木の個数 = 10000, 全てのコアを用いて並列計算を実行)
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
# モデルに適合
forest.fit(X_train, y_train)
# 特徴量の重要度を抽出
importances = forest.feature_importances_
# 重要度の降順で特徴量のインデックスを抽出
indices = np.argsort(importances)[::-1]
# 重要度の降順で特徴量の名称、重要度を表示
for f in range(X_train.shape[1]):
    print("%2d %-*s %f" %(f+1, 30, feat_labels[indices[f]],importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()