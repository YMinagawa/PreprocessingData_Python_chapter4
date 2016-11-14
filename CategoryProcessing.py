import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# カテゴリーデーター処理(４章)
# サンプルデータを生成(Tシャツの色・サイズ・価格・クラスラベル)
# 名義特徴量(color)、順序特徴量(サイズ)、数値特徴量(price)が含まれている
# これらを数値的な特徴量に変換する方法を幾つか示す

df = pd.DataFrame([
['green', 'M', 10.1, 'class1'],
['red', 'L', 13.5, 'class2'],
['blue', 'XL', 15.3, 'class1']
])
# 列名を設定
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

# Tシャツのサイズと整数を対応させるディクショナリを生成
# 特徴量にXL=L+1=M+2のような差があることがわかっているとする。
size_mapping = {'XL':3, 'L':2, 'M':1}
# Tシャツのサイズを整数に変換
df['size'] = df['size'].map(size_mapping)
print('')
print('順序特徴量(サイズ)を整数に変換')
print(df)

# 次にクラスラベルを整数化（エンコーディング)する
# 多くの機械学習ライブラリではクラスラベルが整数値としてエンコードされていることを要求するため。
# クラスラベルと整数を対応させるディクショナリを生成
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
print('')
print('クラスラベルのエンコーディング用マップ')
print(class_mapping)
print('')

print('クラスラベルのエンコード')
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
print('')

# 整数とクラスラベルを対応させるディクショナリを生成
inv_class_mapping = {v:k for k, v in class_mapping.items()}
# 整数からクラスラベルに変換
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print('クラスラベルの逆エンコード')
print(df)

# scikit-learnで直接実装されているラベルエンコーダを使用してみる
# ラベルエンコーダのインスタンスを生成
class_le = LabelEncoder()
# クラスラベルから整数に変換
y = class_le.fit_transform(df['classlabel'].values)
print(y)

# クラスラベルを文字に戻す
class_le.inverse_transform(y)
print(y)
print('')

# 名義特徴量でのone-hotエンコーディング

# color列をLabelEncoderで同様に変換する
# そうすると、blue, red, greenの色で数値の大きさが決まり、大小関係ができる
# そのため、最終的に得られる結果は最適ではない。
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:,0])
print('Color列をLabelEncoderでエンコードした結果')
print(X)

# 上記のようなエンコードは最適でないので、
# 最適化をはかるため、ダミー特徴量を新たに生成する
# blue = (blue, green, red) = (1, 0, 0)となる
# one-hotエンコーダの生成
ohe = OneHotEncoder(categorical_features=[0])
# one-hotエンコーディングを実行
Xt = ohe.fit_transform(X).toarray()
print('Color列をOneHotEncodingでエンコードした結果')
print(Xt)

# dummy特徴量を作る一番簡単な方法はpandasのget_dummies
dum = pd.get_dummies(df[['price', 'color', 'size']])
print(dum)