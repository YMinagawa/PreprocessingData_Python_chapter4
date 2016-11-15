import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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

# Logistic回帰のpenaltyをl1とすることで、l1正規化が行われる
LogisticRegression(penalty='l1')
# L1正則化ロジスティック回帰のインスタンスを生成(逆正則化パラメーター)
lr = LogisticRegression(penalty='l1', C=0.1)
# トレーニングデータに適合
lr.fit(X_train_std, y_train)
# トレーニングデータに対する正解率の表示
print('Training accuracy : ', lr.score(X_train_std, y_train))
# テストデータに対する正解率の表示
print('Test Accuracy : ', lr.score(X_test_std, y_test))
# 切片の表示
print('切片 : ', lr.intercept_)
# 重み係数の表示
print('重み係数の表示 : ', lr.coef_)

# 各クラス毎に重み係数を見た時に0が多い
# 0ではないエントリはほんのわずかしかないので、重みベクトルは疎であるといえる

# 描画の準備
fig = plt.figure()
ax = plt.subplot(111)
# 各係数の色リスト
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
# 空のリストを生成(重み係数、逆正則化パラメーター
weights, params = [], []
# 逆正則化パラメーターの値ごとに処理
for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

# 重み係数をNumPy配列に変換
weights = np.array(weights)
# 各重み係数をプロット
for column, color in zip(range(weights.shape[1]), colors):
    # 横軸を逆正則化パラメーター、縦軸を重み係数とした折れ線グラフ
    plt.plot(params, weights[:,column], label=df_wine.columns[column+1], color=color)

# y=0に黒い破線を引く
plt.axhline(0, color='black', linestyle='--', linewidth=3)
# 横軸の範囲の設定
plt.xlim([10**(-5), 10**5])
# 軸のラベルの設定
plt.ylabel('weight coefficient')
plt.xlabel('C')
# 横軸を対数スケールに設定
plt.xscale('log')
plt.legend(loc='upper left')
plt.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()