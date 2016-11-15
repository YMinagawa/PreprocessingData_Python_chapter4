from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import SequentialBackwardSelection
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# SBS(逐次後退選択)のテスト
# wineのデータセットの特徴量をトレーニングとテストに分割する

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

# k近傍分類機のインスタンスを生成(近傍点数=2)
knn = KNeighborsClassifier(n_neighbors=2)
# 逐次後退選択のインスタンスを生成(特徴量の個数が1になるまで特徴量を選択)
sbs = SequentialBackwardSelection.SBS(knn, k_features=1)
# 逐次後退選択を実行
sbs.fit(X_train_std, y_train)

# k近傍点の個数のリスト
k_feat = [len(k) for k in sbs.subsets_]
# 横軸を近傍点の個数、縦軸をスコアとした折れ線グラフのプロット
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

# 上のグラフからk={5,6,7,8,9,10,11}に対して分類器が100%の正解率を達成している
# 検証用のサブセットでこれだけの性能を達成した5つの特徴が何であるかを表示する
# subsets_は13要素あるsbsets_[0]は全ての特徴量、subsets_[1]は1つ特徴量が減った11個の特徴量、
# subsets[2]は2つ特徴量が減った10個の特徴量が格納されている
k5 = list(sbs.subsets_[8])
print('')
print(df_wine.columns[1:][k5])

# ここで13個すべての特徴量を使ったときと、5個使ったときの差を出す
# 13個すべての特徴量を用いてモデルに適合
knn.fit(X_train_std, y_train)
# トレーニングの正解率を出力
print('')
print('13個の特徴量を使った場合')
print('Training Accuracy', knn.score(X_train_std, y_train))
# テストの正解率を出力
print('Test Accuracy', knn.score(X_test_std, y_test))

# 5個の特徴量を用いてモデルに適合したとき
knn.fit(X_train_std[:,k5], y_train)
# トレーニングの正解率を出力
print('')
print('5個の特徴量を使った場合')
print('Training Accuracy', knn.score(X_train_std[:,k5], y_train))
# テストの正解率を出力
print('Test Accuracy', knn.score(X_test_std[:,k5], y_test))