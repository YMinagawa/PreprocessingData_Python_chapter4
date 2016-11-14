import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer

# 欠測データへの対応一覧(4章)

# サンプルデータを作成
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0, 11.0, 12.0,'''

# python2.7を使用している場合は文字列をunicodeに変換する必要性がある
# csv_data = unicode(cxv_data)
# サンプルデータの読み込み

df = pd.read_csv(StringIO(csv_data))
print("data set", df)

# 各特徴量の欠測値をカウント
print("欠測値")
print(df.isnull().sum())
print('')

# 欠測値を含む行を削除
print("data set(欠測値が含まれる行を削除)")
print(df.dropna())
print('')

# 欠測値を含む列を削除
print("data set(欠測値を含む列を削除)")
print(df.dropna(axis=1))
print('')

# すべての列がNaNである行だけを削除
print("data set(全ての列がNaNである行を削除)")
print(df.dropna(how='all'))
print('')

# 非NaN値が4つ未満の行を削除
print("data set(非NaN値が4つ未満の行を削除)")
print(df.dropna(thresh=4))
print('')

# 特定の列(この場合は'C')にNaNが含まれている行だけを削除
print("data set(特定の列にNaNが含まれていたら、その行を削除)")
print(df.dropna(subset=['C']))
print('')

# Imputerを用いた欠測補完
# 欠測値保管のインスタンスを生成(平均値補完)
# strategyには'median' 'most_frequent'もある
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
# 補完の実行
imputed_data = imr.transform(df.values)
print(imputed_data)