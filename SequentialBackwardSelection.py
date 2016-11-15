from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# 逐次後退選択(Sequential Backward Selection :SBS)
# 新しい特徴部分空間に目的の個数の特徴量が含まれるまで、特徴量全体から特徴量を逐次的に削除していく
# 各段階で削除する特徴量を決定するには、最小化したい評価関数Jを定義する
# 評価関数の計算結果として得られうる評価は、単に、ある特徴量を削除する前後の分類器の性能の差として定義される。
# もう少し直感的にいえば、各段階で性能の低下が最も少ない特徴量を削除する。このSBSの定義に基づき4つの単純なステップにまとめる
# 1. アルゴリズムをk=dで初期化する。dは全体の特徴空間Xdの次元数を表す
# 2. Jの評価を最大化する特徴量x-を決定する。xはx∈Xkである(Xk-x)。(Xk-x)はある特徴量xを除いた特徴量のこと
#           x- = argmax(J(Xk-x))
# 3. 特徴量の集合から特徴量x-を削除する
#           Xk-1 := Xk - x-; k := k-1
# 4. kが目的とする特徴量の個数に等しくなれば終了する。そうでなければステップ2に戻る
# scikit-learnに実装されていないので、一から実装する

class SBS():
    """
    逐次後退選択(sequential backward selection)を実行するクラス
    """

    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring;                # 特徴量を評価する指標
        self.estimator = clone(estimator);     # 推定器
        self.k_features = k_features;          # 選択する特徴量の個数
        self.test_size = test_size;            # テストデータの割合
        self.random_state = random_state       # 乱数種を固定するrandom_state

    def fit(self, X, y):
        # トレーニングデータとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)

        # すべての特徴量の個数、列インデックス(indicesはindexの複数)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        # すべての特徴量を用いてスコアを算出
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        # スコアを格納
        self.scores_ = [score]
        # 指定した特徴量の個数になるまで処理を反復
        while dim > self.k_features:
            # 空のリストの生成(スコア、列インデックス)
            scores = []
            subsets = []

            # combinations('ABCD', 2) = AB, AC, AD, BC, BD, CD
            # combinations('range(4)', 3) = 012, 013, 023, 123
            # 特徴量の部分数合を表す列インデックスの組み合わせごとに処理を反復
            for p in combinations(self.indices_, r=dim-1):
                # スコアを算出して格納
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                # 特徴量の部分集合を表す列インデックスのリストを格納
                subsets.append(p)

            # 最良のスコアのインデックスを抽出
            best = np.argmax(scores)
            # 最良のスコアとなる列インデックスを抽出して格納
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            # 特徴量の個数を１つだけ減らして次のステップへ
            dim -= 1

            # スコアを格納
            self.scores_.append(scores[best])

        # 最後に格納したスコア
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        # 抽出した特徴量を返す
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # 指定された列番号indicesの特徴量を抽出してモデルに適合
        self.estimator.fit(X_train[:,indices], y_train)
        # テストデータを用いてクラスラベルを予測
        y_pred = self.estimator.predict(X_test[:, indices])
        # 真のクラスラベルと予測値を用いてスコアを算出
        score = self.scoring(y_test, y_pred)
        return score
