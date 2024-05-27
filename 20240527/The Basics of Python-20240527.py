#!/usr/bin/env python
# coding: utf-8

# # 今日使うファイルは「The Basics of Python-20240527.ipynb」

# <pre>
# 今日のポイント：
# ①   宿題：課題2と3の参考コードを説明。
# ②　復習と今日の達成目標：　　
#     （一）制御構文（if, else, elif; while, else; for, in; break; continue; pass）　　
#     （二）Python Libraries (NumPy, Pandas, Matplotlib, Seaborn, Plotly, Scikit-Learn)
#     （三）Classでオブジェクトの内容(変数と関数を定義)を記述しておく
#     （四）パッケージとモジュール
#     （五）補足：logistic regression application example: Titanic
# ③　すぐ使えるPython！
#     （1）BeautifulSoupでweb scrape
# 
# 構築環境問題　　
#     1）ダウンロードした.ipynb/.pyファイルの開き方をMISC.xlsxで纏めたので、分からない時はそれを参照ください。
#     2）授業中Jupyter notebookを使うので、.ipynbファイルを開いて使ってください。
# <pre>

# In[ ]:


###################cell.No.1###################
import random

def generate_random_pairs(numbers):
#     random.seed(100)
    random.shuffle(numbers)
    return [(numbers[i], numbers[i+1]) for i in range(0, len(numbers), 2)]

aa=['淺野 こころ', '神之薗 星凪', '清水 大暉','井戸 陸斗','鈴木 健介','竹内 智也','石田 暖人','岩村 胡春','宇治野 正惇','関根 弘樹','田邉 圭佑','松澤 舜平','三輪 涼太','神戸 翔太','竹内 ひかり','先生']
pairs=generate_random_pairs(aa)
print(pairs)


# # 復習・練習

# # if else elif 

# In[ ]:


###################cell.No.2###################
# '''
# if 条件1:
#     処理1を実行...
# [elif 条件2:
#     処理2を実行...]
# [else: 　　　　　　　　#条件1と条件2以外の場合
#     処理3を実行...]
# '''


# In[ ]:


###################cell.No.3###################
# example 3: 80=< x <=100 "優秀"　60=< x <80 "良好"  x<60 "頑張りましょう！"
# example 4: 80<= x <=100 "優秀"　60<= x <80 "良好"  x<60 "頑張りましょう！"

x=35
if 80 <= x <=100:
    print("優秀")
elif 60 <= x <80 :
    print("良好")
else:
    print("頑張りましょう！")


# # while else

# In[ ]:


###################cell.No.4###################
# '''
# while 条件1:
#     処理1を実行...
# [else:
#     処理2を実行...]
# '''


# In[ ]:


###################cell.No.5###################
n = 0
while n < 5:
    print(n)
    n += 1
    
## もしif文を使うと、どんな結果がでる？
# n = 0
# if n < 5:
#     print(n)
#     n += 1


# # for in

# In[ ]:


###################cell.No.6###################
# '''
# for var in 条件1:
#     処理1を実行...
# [else:
#     処理2を実行...]
# '''


# In[ ]:


###################cell.No.7###################
for i in range(5):
    print(i)


# # break, continue, pass

# In[ ]:


###################cell.No.8###################
# break: 繰り返し処理を抜け出す
for n in range(10):
    if n == 5:
        break
    print(n)                 # 0, 1, 2, 3, 4


# In[ ]:


###################cell.No.9###################
# continue: 繰り返し処理をスキップ
for n in range(10):
    if n == 5:
        continue
    print(n)                 # 0, 1, 2, 3, 4, 6, 7, 8, 9


# In[ ]:


###################cell.No.10###################
# pass: 何も処理を行わない、複雑のプロジェクトのコードを考える時、枠の中の細かい部分はまだ考えついてない時、とりあえず席取りのような役割
# 構文法的には文が必要だが、コードとしては何も実行したくない場合のプレースホルダとして有用です。

for n in range(10):
    if n == 5:
        pass
    print(n)   


# # 下の練習をやってみてください！
# 
# # 練習１
# 宅急便のサイズによって、料金が変わるシステム、まずinput()を使ってサイズを確認しましょう。 \
# そのあと、100サイズ以下(<=100)の場合：「郵送料は1000円です」をプリントアウト、 \
#             100-140サイズ(100<x<=140]の場合：「郵送料は1200円です」をプリントアウト、 \
#             140-160サイズ(140<x<=160]の場合：「郵送料は1500円です」をプリントアウト 
# 

# In[ ]:


###################cell.No.11###################
x=int(input('荷物のサイズ：'))

if x in range(0,100):
    print("郵送料は1000円です")
elif x in range(100,140) :
    print("郵送料は1200円です")
elif x in range(140,160) :
    print("郵送料は1500円です")


# # 練習2
# 空の辞書を作る。
# input()を使ってIDとパスワードを聞いてください。
# これを三回繰り返して異なるIDとパスワードのペアを辞書に追加。
# 辞書をプリントアウト。
# 
# そして、下記のコードを使って、IDとパスワードを入力してください。
# import ipywidgets as wg
# from IPython.display import display
# myName = wg.Text(description='ID:')
# Lucky_number = wg.Text(description='Lucky number:')
# display(myName, Lucky_number)
# 
# ログインのIDとパスワードを両方とも一致する場合、「Welcome」をプリントアウト、 \
# 一つ違ったら「もう一回正しいIDとパスワードを入力してください」をプリントアウト  \

# # 練習3
# https://www.nagoya-u.ac.jp/academics/curriculum/grading_gpa/index.html　 \
# 1-3．100点満点による評価を記号による評価に換算する場合の標準的方法 \
# を参照して、それぞれの点数の評価記号をプリントアウト

# 2回目の宿題提出期限：5月27日　\
# 内容：練習２と３

# # 練習4
# 航空会社の搭乗手続きを行う時、荷物を預ける場面で、 \
# もし23KGを超える時、１KGあたり440円を徴収する、100KGまで預けるが、もし100KGを超えると、「お預けできない。」 \
# ヒントHint: input(), if-elif条件
# 
# (ここの数字は例を作るためです、本当は距離によって料金は違う、例えば：東京→マニラ（片道）の重量超過（23kg～32kg以下）：6,000円、重量超過（32kg超～45kg以下）：30,000円)

# In[ ]:


###################cell.No.12###################
x=int(input('荷物の重さ(単位はKG)：'))

if x in range(1,23):
    print("23KG以内なので、超過手荷物料金は課さない")
elif x in range(23,100) :
    excess_baggage_charge = (x-23)*440
    print(f"23KGを超えたので、超過手荷物料金は{excess_baggage_charge}円です。")
elif x > 100 :
    print("100KGを超えたので、お預けできない。")


# # 練習5
# 一週間の給与を計算する場面で、 \
# もし一週間で40時間未満（40含む）の場合、一時間1,027円で、40時間以上の場合、40時間超過した分を給与1.5倍で計算、一週間のトータル給与を計算する。 \
# input(), if-elif条件

# In[ ]:


###################cell.No.13###################
x = float(input('一週間の労働時間：'))

if 0 <= x <= 40:
    wage = x * 1027
    print(f"一週間のトータル給与は{wage}円です。")
elif x > 40 :
    wage = 40*1027 + (x-40)*1027 * 1.5
    print(f"一週間のトータル給与は{wage}円です。")
else:
    print("労働時間数を正しく入力してください！")


# In[ ]:


###################cell.No.14###################
#素数かどうかを判断する、（素数：正の約数が1とその数自身であり、1でない自然数のこと）

num = int(input("任意の1より大きいな整数を入力してください"))

#考え方はnumの全ての約数を取得し、その約数は1と自身以外の場合も取得できるなら素数ではないと判断

i = 2
flag = True

while i < num:
    if num % i == 0:
        flag = False
    i += 1

if flag == True:
    print(num,"は素数")
else:
    print(num,"は素数ではない")


# In[ ]:


###################cell.No.15###################
#100以下の素数をすべてプリントアウト、（素数：正の約数が1とその数自身である約数で、1でない自然数のこと）
#考え方はnumの全ての約数を取得し、その約数は1と自身以外の場合も取得できるなら素数ではないと判断
from time import *

begin = time()

num = 2
list_prime_number = []
while num < 10000:
    i = 2
    flag = True

    while i < num:
        if num % i == 0:
            flag = False
        i += 1

    if flag == True:
#         print(num,"は素数")
        list_prime_number.append(num)
    num += 1

end = time()

# print(list_prime_number)
print(f"必要の時間は{end-begin}秒です。")

# 10000のかかった時間は：3.28275990486145秒


# In[ ]:


###################cell.No.16###################
# 素数の改善バージョン1
#1000以下の素数をすべてプリントアウト、（素数：正の約数が1とその数自身である約数で、1でない自然数のこと）
#考え方はnumの全ての約数を取得し、その約数は1と自身以外の場合も取得できるなら素数ではないと判断
from time import *

begin = time()

num = 2
list_prime_number = []
while num < 10000:
    i = 2
    flag = True

    while i < num:
        if num % i == 0:
            flag = False
            break
        i += 1

    if flag == True:
#         print(num,"は素数")
        list_prime_number.append(num)
    num += 1

end = time()

# print(list_prime_number)
print(f"必要の時間は{end-begin}秒です。")
# 改善バージョン1：10000のかかった時間は：0.40697598457336426秒


# In[ ]:


###################cell.No.17###################
# 素数の改善バージョン2
#1000以下の素数をすべてプリントアウト、（素数：正の約数が1とその数自身である約数で、1でない自然数のこと）
#考え方はnumの全ての約数を取得し、その約数は1と自身以外の場合も取得できるなら素数ではないと判断
from time import *

begin = time()

num = 2
list_prime_number = []
while num < 10000:
    i = 2
    flag = True

    while i <= (num**0.5):
        if num % i == 0:
            flag = False
            break
        i += 1

    if flag == True:
#         print(num,"は素数")
        list_prime_number.append(num)
    num += 1

end = time()

# print(list_prime_number)
print(f"必要の時間は{end-begin}秒です。")
# 改善バージョン2：10000のかかった時間は：0.034707069396972656秒


# In[ ]:


###################cell.No.18###################
# 1000以内のナルシシスト数を全てプリントアウト。
# （ナルシシスト数(Narcissistic number)：n 桁の自然数であって、その各桁の数の n 乗の和が、元の自然数に等しくなるような数）
# 1000以内なら、3桁の自然数であり、各桁の数の３乗の和がその自然数と等しいなら、ナルシシスト数

i = 100

while i < 1000:
#     まず各桁を取り出す、一の位をa、10の位をb、100の位をcとする
    a = i//100
    b = i//10%10  #b=(i-100*a)//10
    c = i%10
    
    if a**3+b**3+c**3 == i:
        print(i)
    i +=1


# In[ ]:





# # （二）Python Libraries (NumPy, Pandas, Matplotlib, Seaborn, Plotly, Scikit-Learn)

# # NumPy (多次元配列（multidimensional array）、数値演算)

# In[ ]:


###################cell.No.19###################
# The standard way to import NumPy:
import numpy as np

# Create a 2-D array, 配列とは、プログラムで扱うデータ構造において、同じ長さのデータを並べたものである。
# set every second element in some rows and find max per row:

x = np.arange(15, dtype=np.int64).reshape(3, 5) #reshape(3, 5)は三行5列の配列に形を変えるという意味
x


# In[ ]:


###################cell.No.20###################
x[1:] #Output結果を観察してください # x[1:,:]もチェックしてみて！


# In[ ]:


###################cell.No.21###################
x[:, ::2]  #Output結果を観察してください


# In[ ]:


###################cell.No.22###################
x.shape # (3, 5) 3は行の数、5は列の数


# In[ ]:


###################cell.No.23###################
x.ndim # 2は次元数、Rank


# In[ ]:


###################cell.No.24###################
x.max(axis=1) #x.max(axis=0)もチェックしてみて！


# In[ ]:


###################cell.No.25###################
# NumPy arrayを作る (配列とは、プログラムで扱うデータ構造において、同じ長さのデータを並べたものである。)
arr = np.array([1, 2, 3, 4, 5])

# 平均と標準偏差を計算
mean_value = np.mean(arr)
std_dev = np.std(arr)

print(f"Mean: {mean_value}, Standard Deviation: {std_dev}")


# In[ ]:


###################cell.No.26###################
x = np.arange(30, dtype=np.int64).reshape(3, 5, 2) #reshape(3, 5)は三行5列の配列に形を変えるという意味
x


# In[ ]:


###################cell.No.27###################
print('Shape:', x.shape)
print('Rank:', x.ndim)


# In[ ]:


###################cell.No.28###################
# 23を抽出
x[2,1,1]


# In[ ]:





# # Pandas (データフレーム(DataFrame)、データ前処理)

# In[ ]:


###################cell.No.29###################
import pandas as pd

# Creating a Pandas DataFrame
data = {'Name': ['John', 'Alice', 'Bob'],
        'Age': [25, 28, 22],
       'Gender': [0, 1, 0]}
df_DataFrame = pd.DataFrame(data)

print(df_DataFrame)
print(df_DataFrame['Gender'])
print(df_DataFrame.index)


# In[ ]:


###################cell.No.30###################
data = {'Name': ['John', 'Alice', 'Bob'],
        'Age': [25, 28, 22],
       'Gender': [0, 1, 0]}
df_Series = pd.Series(data)
print(df_Series)
print(df_Series['Gender'])
print(df_Series.index)


# In[ ]:


###################cell.No.31###################
df_Series['Age'][-1]


# In[ ]:


###################cell.No.32###################
df = pd.DataFrame([[10, 20,30, 30], [25, 50,65, 80]], index=["row1", "row2"], columns=["A", "B", "C", "D"])
print(df)
print('-'*100)
print(df.query('A >= 5 and C < 50')) #dataframeはquery関数で抽出できるが、seriesはquery関数使えない


# # NumPy vs. Pandas

# In[ ]:


###################cell.No.33###################
import numpy as np

arr = np.array([5, 32, 17, 29])
print(arr)
print(type(arr))
print(arr.shape)
print(len(arr))
print(arr[0])


# In[ ]:


###################cell.No.34###################
# arr.values
# arr.index


# In[ ]:


###################cell.No.35###################
import pandas as pd

series1 = pd.Series([5, 32, 17, 29])
print(series1)
print(type(series1))
print(series1.shape)
print(len(series1))
print(series1[0])


# In[ ]:


###################cell.No.36###################
print(series1.values)
print(series1.index)


# In[ ]:


###################cell.No.37###################
import pandas as pd

data1 = pd.Series([25, 28, 22], index = ['John', 'Alice', 'Bob'])
data2 = pd.DataFrame([25, 28, 22], index = ['John', 'Alice', 'Bob'])

print(data1)
print(data1.index)
print('data1_John:', data1['John'])
print(data1.loc[['John',"Bob"]])
print(data1.iloc[0])
print(data1.shape)

print(data2)
print(data2.index)
# print('data2_John:', data2['John'])
print(data2.loc[['John',"Bob"]])
print(data2.iloc[0])
print(data2.shape)


# In[ ]:


###################cell.No.38###################
list1=[25, 28, 22]
print(list1)
print(list1[0])
print(list1.index)


# In[ ]:


###################cell.No.39###################
data = {'Name': ['John', 'Alice', 'Bob'],
        'Age': [25, 28, 22],
       'Gender': [0, 1, 0]}
df = pd.Series(data)
print(df)
print(df['Gender'][1])


# In[ ]:


###################cell.No.40###################
data = {'Name': ['John', 'Alice', 'Bob'],
        'Age': [25, 28, 22],
       'Gender': [0, 1, 0]}
df = pd.DataFrame(data)
df = df.set_index("Age", inplace = False)
print(df)
# print(df['Gender'][1]) #df['Gender'][1]ここの1はインデックスではなくなります、Ageの数字をインデックスとして設定したので
print(df['Gender'][25]) 

print('-'*100)
df = df.reset_index(drop=False)
print(df)
df = df.set_index("Gender", inplace = False)
print(df)
print('-'*100)
print(df['Age'][0])  #ここの0はインデックス、Genderが０のAgeの値をすべて取り出す
print(df['Name'][0]) #ここの0はインデックス、Genderが０のNameの値をすべて取り出す


# In[ ]:





# # Scikit-learn (オープンソース機械学習ライブラリ)

# ## ロジスティック回帰を例として、機械学習の流れを説明する
# 
# ロジスティック回帰は、対象となるあるものが特定のクラスに属する確率を予測するために、機械学習で使用される分類アルゴリズムです。
# 複数の入力変数と、出力変数(例えば：Yes/no、True/False、1/0 など) との関係
# 
# 下記の分類の例では、複数の特徴からアヤメを分類する問題、Iris-setosaとIris-versicolorとIris-virginicaは全部アヤメの子種類
# 入力変数：がく片(sepal)の長さ、がく片の幅、花びら(petal)の長さ、花びらの幅
# 出力変数：Iris-setosa（０）、Iris-versicolor（１）、Iris-virginica（２）

# ![sepals-vs-petals.jpg](attachment:sepals-vs-petals.jpg)

# ![iris-machinelearning.png](attachment:iris-machinelearning.png)

# In[ ]:


###################cell.No.41###################
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load iris dataset as an example #例としてirisデータセットを読み込む
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets #データをトレーニングセットとテストセットに分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler #StandardScalerを使用した特徴の標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train a Logistic Regression model #ロジスティック回帰モデルの作成と訓練
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set #テストセットで予測を立てる
predictions = model.predict(X_test_scaled)

# Calculate accuracy #予測精度を計算する（計算式：Accuracy = matches / samples）
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy}")


# In[ ]:


###################cell.No.42###################
y_test


# In[ ]:


###################cell.No.43###################
predictions


# In[ ]:


###################cell.No.44###################
y_pred111 = [0, 1, 1, 3]
y_true111 = [0, 1, 2, 3]
accuracy_score(y_pred111, y_true111)


# In[ ]:





# # Matplotlib (データの視覚化)

# In[ ]:


###################cell.No.45###################
import pandas as pd 
import matplotlib.pyplot as plt 
iris = pd.read_csv("iris.csv") 
  
plt.plot(iris.Id, iris["SepalLength"], "r--") 
plt.show 


# In[ ]:


###################cell.No.46###################
iris.plot(kind ="scatter", 
          x ='SepalLength', 
          y ='PetalLength') 
plt.grid() 


# In[ ]:


###################cell.No.47###################
import matplotlib.pyplot as plt

# Creating a simple line plot
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
xy_df = pd.DataFrame({'x' : x, 'y' : y})

plt.plot(xy_df['x'], xy_df['y'])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')
plt.show()


# In[ ]:


###################cell.No.48###################
iris


# In[ ]:





# # Seaborn  (データの視覚化)

# In[ ]:


###################cell.No.49###################
import seaborn as sns

# Using Seaborn to create a scatter plot
tips = sns.load_dataset('tips')
sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.title('Scatter Plot using Seaborn')
plt.show()


# In[ ]:


###################cell.No.50###################
import seaborn as sns 
  
iris = sns.load_dataset('iris') 
  
# style used as a theme of graph  
# for example if we want black  
# graph with grid then write "darkgrid" 
sns.set_style("whitegrid") 
  
# sepal_length, petal_length are iris 
# feature data height used to define 
# Height of graph whereas hue store the 
# class of iris dataset. 
sns.FacetGrid(iris, hue ="species",  
              height = 6).map(plt.scatter,  
                              'sepal_length',  
                              'petal_length').add_legend() 


# In[ ]:


###################cell.No.51###################
iris = sns.load_dataset("iris")
plt.figure(figsize=(8,6))
sns.violinplot(x="species", y="petal_length", data=iris)
plt.title("Petal Length Distribution by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()


# In[ ]:





# # Plotly (データの視覚化)

# In[ ]:


###################cell.No.52###################
import plotly.express as px

# Creating a Plotly scatter plot　#scatter: 散布図
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", title="Iris Dataset")
fig.show()


# In[ ]:


###################cell.No.53###################
import plotly.express as px 
 
df = px.data.iris() 
 
fig = px.scatter_3d(df, x = 'sepal_width', 
                    y = 'sepal_length', 
                    z = 'petal_width', 
                    color = 'species') 
fig.show()


# In[ ]:


###################cell.No.54###################
# matplotlibで３Dプロット
from sklearn import datasets
iris = datasets.load_iris()

from mpl_toolkits.mplot3d import Axes3D
y = iris.target

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d') # ※1 三次元プロットを指定

markers = ['o', '^', 's']
colors = ['blue', 'red', 'green']

for i, label in enumerate(iris.target_names):
  ax.scatter(xs=iris.data[iris.target==i, 0], # x軸データ
             ys=iris.data[iris.target==i, 1], # y軸データ
             zs=iris.data[iris.target==i, 2], # z軸データ
             label=iris.target_names[i],
             marker=markers[i],
             s=100,
             c=colors[i])

ax.legend(loc='best', fontsize=14)

ax.set_title('Iris', size=16)
ax.set_xlabel(iris.feature_names[0], size=14)
ax.set_ylabel(iris.feature_names[1], size=14)
ax.set_zlabel(iris.feature_names[2], size=14)

plt.show()


# In[ ]:


###################cell.No.55###################
# Plotlyは、ドロップダウン、ボタン、スライダーなどの追加など、プロットを操作するためのさまざまなツールを提供します。

import plotly.graph_objects as px
import numpy as np

np.random.seed(42)
 
random_x = np.random.randint(1, 101, 100)
random_y = np.random.randint(1, 101, 100)
 
plot = px.Figure(data=[px.Scatter(
    x=random_x,
    y=random_y,
    mode='markers',)
])
 
# Add dropdown
plot.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=["type", "scatter"],
                    label="Scatter Plot",
                    method="restyle"
                ),
                dict(
                    args=["type", "bar"],
                    label="Bar Chart",
                    method="restyle"
                )
            ]),
            direction="down",
        ),
    ]
)
 
plot.show()


# In[ ]:





# # （三）Classでオブジェクトの内容(変数と関数を定義)を記述しておく

# In[ ]:


###################cell.No.56###################
class Person:
    name = "ssdf"
    
    def say_hello(a):
        print("おはよう！")
#

p1 = Person()
p2 = Person()


# In[ ]:


###################cell.No.57###################
print(p1.name)


# In[ ]:


###################cell.No.58###################
print(id(p1))
print(id(p2))


# In[ ]:


###################cell.No.59###################
p1.say_hello()

# TypeError: Person.say_hello() takes 0 positional arguments but 1 was given
# Pythonのクラス内で定義した関数を外から呼び出す場合、
# 自動的に第一引数にレシーバーが渡される。
# それを引き取るための引数を定義してあげる必要がある。
# 慣習的にはselfを使う。
# https://qiita.com/mizuno_jin/items/7c1f5a90dfb89bd64bff


# In[ ]:


###################cell.No.60###################
print(p2.name)


# In[ ]:


###################cell.No.61###################
class Test:
    def test_method(self):
        print(self)

t = Test()
print("tのメモリー上の保存id:")
print(t)

print('-' * 100)
print("tのtest_methodという方法を呼び出す時、selfのid:")
t.test_method()


# In[ ]:


###################cell.No.62###################
class Person:
    def say_hello(self):
        print("おはよう！私の名前は：%s"%self.name)
#

p1 = Person()
p1.name = "ルフィ"
p1.say_hello()


# In[ ]:


###################cell.No.63###################
list_employee = ["Yamada", "Yamashita", "Yamagami"]
class Person_list:
    def __init__(self):
        print("あなたのお名前は？")
    def add_person(self):
        print("おはよう！私の名前は：%s"%self.name)
        list_employee.append(self.name)

p1 = Person_list()
p1.name = "Luffy"
p1.add_person()
print(list_employee)

p2 = Person_list()
p2.name = "Zoro"
p2.add_person()
print(list_employee)


# In[ ]:


###################cell.No.64###################
class Person_list:
    def __init__(self):
        self.value = ["Yamada", "Yamashita", "Yamagami"]
        print("あなたのお名前は？")
    def add_person(self,new_value):
        print("おはよう！私の名前は：%s"%new_value)
        self.value.append(new_value)
        print(self.value)

list_0411 = Person_list()
list_0411.add_person("Luffy")

list_0412 = Person_list()
list_0412.add_person("Zoro")


# In[ ]:


###################cell.No.65###################
class Cat:
    def set_name(self, name):
        self.name=name
    def get_name(self):
#         print(self.name)
        return self.name  #関数の実行結果を呼び出し元に通知するのが「 return文」,値を返して関数を終了する
        print(self.name)
    def say_hello(self):
        print("meow, meow, meow, hello, my name is %s"%self.name)
    def eat(self):
        if self.button == 1:
            print("ご飯を食べたい")
        elif self.button == 2:
            print("水を飲みたい")
        else:
            print("正しい数字を入力してください")


# In[ ]:


###################cell.No.66###################
cat1 = Cat()
cat1.set_name("kitty")
cat1.get_name()
# print(cat1.get_name())
cat1.say_hello()
cat1.button = 1
cat1.eat()


# In[ ]:


###################cell.No.67###################
print(cat1.name)


# In[ ]:


###################cell.No.68###################
cat1_button = int(input('Number_button:'))


# In[ ]:


###################cell.No.69###################
cat1.button=cat1_button
cat1.eat()


# # Class作成の練習（dog, Rectangle長方形面積計算）

# dogのclassを作って、catのコードを参考しながら、set_name、get_name、say_hello、playの関数を定義して、playの関数に関して、button == 1の場合は"「だるまさんがころんだ」のゲームをやりたい", button == 2の場合は"ボールゲームをやりたい"をプリントアウト。dog1の名前を「ハチ公」にし、 dog2の名前を「白いおとうさん」にしてください。

# In[ ]:


###################cell.No.70###################
class Dog:
    def set_name(self, name):
        self.name=name
    def get_name(self):
        print(self.name)
        return self.name
    def say_hello(self):
        print("woof, woof, woof, hello, my name is %s"%self.name)
    def play(self):
        if self.button == 1:
            print("「だるまさんがころんだ」のゲームをやりたい")
        elif self.button == 2:
            print("ボールゲームをやりたい")
        else:
            print("正しい数字を入力してください")


# In[ ]:


###################cell.No.71###################
dog1 = Dog()
dog1.set_name("ハチ公")
dog1.get_name()
dog1.say_hello()
dog1.button = 1
dog1.play()


# In[ ]:


###################cell.No.72###################
dog2 = Dog()
dog2.set_name("白いおとうさん")
dog2.get_name()
dog2.say_hello()
dog2.button = 2
dog2.play()


# In[ ]:





# In[ ]:


###################cell.No.73###################
class Rectangle:
    def __init__(self, width, height):
        self.hidden_width = width
        self.hidden_height = height
    def calculate_area(self):
        print(self.hidden_width*self.hidden_height)
        return (self.hidden_width*self.hidden_height)


# In[ ]:


###################cell.No.74###################
cal_1=Rectangle(3,4)
cal_1.calculate_area()


# In[ ]:


###################cell.No.75###################
cal_1.hidden_height


# In[ ]:


###################cell.No.76###################
# class内の変数(attribute)やメソッド(method)にプライベートな意味合いを持たせたい場合、
# 該当変数やメソッドの前に二つのアンダースコア（下線）をつけます。

class Rectangle_2:
    # __二つのアンダースコア（下線）で始まる変数の名前は非公開
    def __init__(self, width, height):
        self.__width = width
        self.__height = height
    def calculate_area(self):
        print(self.__width*self.__height)
        return (self.__width*self.__height)


# In[ ]:


###################cell.No.77###################
cal_2=Rectangle_2(4,5)
cal_2.calculate_area()


# In[ ]:


###################cell.No.78###################
cal_2.__height #AttributeErrorが出たのは__heightという変数はclassの中で隠されたから


# In[ ]:


###################cell.No.79###################
cal_2._Rectangle_2__height #非公開の変数を呼び出すために、．の後ろに{「_」+class_name}+「__height」,波括弧の部分を追加必要


# In[ ]:





# # （四）パッケージとモジュール

# In[ ]:


###################cell.No.80###################
import My_package1.My_package2.my_module1
My_package1.My_package2.my_module1.myfunc()


# In[ ]:


###################cell.No.81###################
from My_package1.My_package2 import my_module1
my_module1.myfunc()


# In[ ]:


###################cell.No.82###################
from My_package1.My_package2.my_module1 import myfunc
myfunc()


# In[ ]:


###################cell.No.83###################
from My_package1.My_package2.my_module1 import *
myfunc()


# In[ ]:


###################cell.No.84###################
from My_package1.My_package2.my_module1 import myfunc as myfunc111
myfunc111()


# In[ ]:





# # （五）補足：logistic regression application example: Titanic
# https://www.kaggle.com/c/titanic/data

# In[ ]:


###################cell.No.85###################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[ ]:


###################cell.No.86###################
titanic = pd.read_csv('titanic.csv')
titanic.head()


# In[ ]:


###################cell.No.87###################
titanic.info()


# In[ ]:


###################cell.No.88###################
# Remove missing values
titanic.dropna(inplace=True)


# In[ ]:


###################cell.No.89###################
# Convert categorical variables into numerical ones
titanic['Sex'] = titanic['Sex'].map({'female': 0, 'male': 1})
titanic['Embarked'] = titanic['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})


# In[ ]:


###################cell.No.90###################
# Select the features and target variable
X = titanic[['Age', 'Sex', 'Pclass','SibSp', 'Parch', 'Embarked']]
y = titanic['Survived']


# In[ ]:


###################cell.No.91###################
X


# In[ ]:


###################cell.No.92###################
# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


###################cell.No.93###################
# Build the logistic regression model
model = LogisticRegression(random_state=42)


# In[ ]:


###################cell.No.94###################
# Fit the model to the training data
model.fit(X_train, y_train)


# In[ ]:


###################cell.No.95###################
# Make predictions on the testing data
y_pred = model.predict(X_test)


# In[ ]:


###################cell.No.96###################
# Calculate the accuracy of the model
accuracy = accuracy_score(y_pred, y_test)


# In[ ]:


###################cell.No.97###################
print('Accuracy:', accuracy)


# In[ ]:


###################cell.No.98###################
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred, y_test)


# In[ ]:


###################cell.No.99###################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


###################cell.No.100###################
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix=confusion_matrix(y_test,y_pred)
print(matrix)
score=accuracy_score(y_test,y_pred)
print(score)
report=classification_report(y_test,y_pred)
print(report)
plot_confusion_matrix(matrix, classes=['Not-Survived', 'Survived'])


# In[ ]:


###################cell.No.101###################
# 上のConfusion matrixを分かりやすく理解するために、本当の値と予測値の最初の五つの数値だけで観察しましょう。

y_pred111 = np.array(y_test[:5])
print('予測の値：', y_pred111)
y_true111 = y_pred[:5]
print('本当の値：', y_true111)

print('-'*100)
print(accuracy_score(y_true111, y_pred111))
matrix=confusion_matrix(y_true111,y_pred111)
print(matrix)
plot_confusion_matrix(matrix, classes=['0', '1'])


# In[ ]:





# In[ ]:





# # すぐ使えるPython！
# # （1）BeautifulSoupでweb scrape

# In[ ]:


###################cell.No.102###################
import requests
from bs4 import BeautifulSoup

my_url="https://www.chembio.nagoya-u.ac.jp/labhp/organic1/publication/pub2023.html"
page_soup = BeautifulSoup(requests.get(my_url).content ,'html.parser') # parser:解析器、分析器


# In[ ]:


###################cell.No.103###################
page_soup


# In[ ]:


###################cell.No.104###################
list_title = []
for title in page_soup.select('h2[class="publication"]'):
    print(title.get_text(strip=True, separator=' '))
    print('-' * 100)
    list_title.append(title.get_text(strip=True, separator=' '))


# In[ ]:


###################cell.No.105###################
import pandas as pd

# dictionary of lists
dict1 = {'titles': list_title}

df = pd.DataFrame(dict1)

# saving the dataframe
df.to_csv('2023_publications_titles_1.csv')


# In[ ]:


###################cell.No.106###################
import pandas as pd

df = pd.DataFrame(list_title,columns=['titles']) #columns:列

# saving the dataframe
df.to_csv('2023_publications_titles_2.csv')


# In[ ]:





# In[ ]:


###################cell.No.107###################
import requests
from bs4 import BeautifulSoup

url = 'https://web.iitm.ac.in/bioinfo2/cpad2/peptides/?page=1'
url_soup = BeautifulSoup(requests.get(url).content ,'html.parser')

for tr in url_soup.select('tr[data-toggle="modal"]'):
    print(tr.get_text(strip=True, separator=' '))
    print('-' * 120)


# In[ ]:


###################cell.No.108###################
iitm_list_title = []

for tr in url_soup.select('tr[data-toggle="modal"]'):
    print(tr.get_text(strip=True, separator=' '))
    print('-' * 100)
    iitm_list_title.append(tr.get_text(strip=True, separator=' '))


# In[ ]:


###################cell.No.109###################
import pandas as pd

df_iitm_list = pd.DataFrame(iitm_list_title,columns=['titles']) #columns:列

# saving the dataframe
df_iitm_list.to_csv('iitm_titles.csv')


# In[ ]:


###################cell.No.110###################
# !pip3 install wordcloud
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# In[ ]:


###################cell.No.111###################
df = pd.read_csv('iitm_titles.csv')
df.head()


# In[ ]:


###################cell.No.112###################
#Create a word cloud
stopwords1 = stopwords.words()
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords1,
                          max_words=500,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df['titles']))
fig = plt.figure(figsize=(100,100))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:


###################cell.No.113###################
df = pd.read_csv('2023_publications_titles_1.csv')
df.head()


# In[ ]:


###################cell.No.114###################
#Create a word cloud
stopwords1 = stopwords.words()
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords1,
                          max_words=500,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df['titles']))
fig = plt.figure(figsize=(100,100))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:


###################cell.No.115###################
import requests
from bs4 import BeautifulSoup

url = 'https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=bisimide&btnG='
url_soup = BeautifulSoup(requests.get(url).content ,'html.parser')


# In[ ]:


###################cell.No.116###################
articles = url_soup.find_all("div", {"class": "gs_ri"})


# In[ ]:


###################cell.No.117###################
def parse_data_from_article(article):
    title_elem = article.find("h3", {"class": "gs_rt"})
    title = title_elem.get_text()
    title_anchor_elem = article.select("a")[0]
    url = title_anchor_elem["href"]
    article_id = title_anchor_elem["id"]
    authors = article.find("div", {"class": "gs_a"}).get_text()
    return {
        "title": title,
        "authors": authors,
        "url": url,
    }


# In[ ]:


###################cell.No.118###################
data = [parse_data_from_article(article) for article in articles]


# In[ ]:


###################cell.No.119###################
data


# In[ ]:


###################cell.No.120###################
data[1]["url"]


# In[ ]:


###################cell.No.121###################
import pandas as pd

titles_list=[]
authors_list=[]
urls_list=[]
for i in range(len(data)):
    titles_list.append((data[i]["title"]))
    authors_list.append((data[i]["authors"]))
    urls_list.append((data[i]["url"]))


# In[ ]:


###################cell.No.122###################
import pandas as pd

dict1 = {'titles': titles_list, 'authors': authors_list, 'urls': urls_list }

df = pd.DataFrame(dict1)

# saving the dataframe
df.to_csv('scrape_google_scholar_with_python.csv')


# In[ ]:





# In[ ]:




