#!/usr/bin/env python
# coding: utf-8

# <pre>
# 実習の時、実行終わったらペア先生に見せて、お互い問題なかったらOK、問題あれば助け合ってください。二人とも解決できない場合、私を呼んでください。
# 
# 自由闊達の学風のもとで、みんな自由に議論・意見・要望を出してください。
# 多種多様性を重視・尊重します。
# <pre>

# <pre>
# 今日のポイント：
# ①　構築環境問題　　
#     1）前回授業やったデフォルトdirectoryを探す目的：ダウンロードした.ipynb/.pyファイルを自分パソコンのどのホルダーの中にコピーするかを分かるために。授業中はそのまま開いて使ってください。　　
# ②　復習と今日の達成目標：　　
#     １）五つのオブジェクト型（list, tuple, set, dict, function）　　
#     ２）制御構文（if, else, elif; while, else; for, in;          break; continue; pass ）
# ③　すぐ使えるPythonファイルを実行してみてください！
# <pre>

# In[1]:


# ランダムに二人ペアを組むことにより、実習の時、お互いの先生をやって、お互いの実習結果をチェックする。
# 目標：授業時間の効率アップを目指す

import random

def generate_random_pairs(numbers):
#     random.seed(100)
    random.shuffle(numbers)
    return [(numbers[i], numbers[i+1]) for i in range(0, len(numbers), 2)]

aa=['淺野 こころ', '神之薗 星凪', '清水 大暉','井戸 陸斗','鈴木 健介','竹内 智也','石田 暖人','岩村 胡春','宇治野 正惇','関根 弘樹','田邉 圭佑','松澤 舜平','三輪 涼太','神戸 翔太','竹内 ひかり','先生']
pairs=generate_random_pairs(aa)
print(pairs)


# # 20240430
# 
# ### Pythonで非常によく使われるデータ型 int float str

# ### 比較演算子(==, !=, <, >, <=, >=, is, is not, in, not in)
# 
# ==　等しい　\
# !=　異なる　\
# is　等しい　\
# is not　異なる　\
# in　含まれる　\
# not in　含まれない

# # リスト list

# In[ ]:


a = [1, 3, 5, 7, 9]
for n in a:
    print(n)


# In[ ]:


a = [1, 3, 5, 7, 9]
for n in range(5):
    print(n, a[n])


# In[ ]:


print(a[2])


# In[ ]:


print(a[-1])


# In[ ]:


a1 = a[0:2] #リストの0番から1番の数字を取る
a1


# In[ ]:


a2 = a[2:] #リストの2番から最後の数字を取る
a2


# In[ ]:


a3 = a[:3] #リストの最初から2番までの数字を取る
a3


# In[ ]:


print([1, 2, 3] + [4, 5, 6]) # +演算子を用いてリストを結合


# In[ ]:


# append:追加
a = [1, 3, 5, 7, 9]
print(a)
a.append(100)
a


# In[ ]:


a.append('abc')
a


# In[ ]:


# extend:拡張
a.extend([10, 11, 12])
a


# In[ ]:


#　insert：挿入
a.insert(0, 300)
print(a)


# # 復習・練習：
# リストを作る。（四択一でいい）
# 
# ①['勉強・研究', '単位', '友達', '英語', '健康', '安全', '就職', '留学', '進学']　\
# ②趣味　\
# ③好きなお菓子　\
# ④受けている授業名　
# 
# そのあと、　\
# リストの0番目と2番目をプリントアウト　\
# リストの0番から2番（含め）までをプリントアウト　\
# リストに('家族')を追加してプリントアウト　\
# リストの0番に'試験'を追加してプリントアウト　

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


aaa=['勉強・研究','単位','友達','英語','健康','安全','就職','留学','進学']


# In[ ]:


aaa[2]


# In[ ]:


aaa[0:3]


# In[ ]:


aaa.append('家族')
print(aaa)


# In[ ]:


aaa.insert(0, '試験')
print(aaa)


# In[ ]:


bbb=list()
bbb.append('勉強・研究')
bbb.append(['単位','友達','英語','健康','安全','就職','留学','進学'])
print(bbb)


# In[ ]:


bbb[0]


# In[ ]:


bbb[1][2]


# # タプル tuple

# In[ ]:


a1 = [10, 20, 30, 40]    #リストは変更可能
a2 = (110, 120, 130, 140)    #タプルは変更不可能


# In[ ]:


a1[2] = 60      # 要素変更できる
a1


# In[ ]:


a2[2] = 60      # 要素変更できない
a2


# In[ ]:


a2 = (110, 120, 130, 140)    #タプルは変更不可能
a3=list(a2)   #変更したい場合リストにタイプ変更
a3


# In[ ]:


# a2 = (110, 120, 130, 140) 
a2 = (110, 120, )
print(a2)
print(type(a2))


# In[ ]:


a2 = (110)
print(a2)
print(type(a2))


# In[ ]:


a4=(110, 120, 130, 140) +(10, 20, 30, 40) 
a4


# In[ ]:


a4=(110, 120, 130, 140) + tuple([10]) 
a4


# In[ ]:


a4=(110, 120, 130, 140) + [10]
a4


# In[ ]:


tuple1 = (0, 1, 2)

list1 = list(tuple1)
list1.remove(0)
tuple1_remove = tuple(list1)

print(tuple1_remove)


# # 復習・練習：　
# 先ほど作ったリストをタプルにデータ型変更してください。　\
# そのあと、　\
# タプルの0番から3番（含め）までをプリントアウト

# In[ ]:


tuple_aaa=tuple(aaa)
print(tuple_aaa)


# In[ ]:


tuple_aaa[0:4]


# # セット set

# In[ ]:


set1={1,2,3}


# In[ ]:


set1[0]        #セットは順番がない


# In[ ]:


type(set1)


# In[ ]:


s = set([1,2,3])     #これはリストをセットに変換した作り方でできたセット
s


# In[ ]:


s=set()
s.add(1)
s.add(2)
s.add(3)
print(s)
print(type(s))


# In[ ]:


s.remove(2)
s


# In[ ]:


s1 = set([1, 2, 3, 4, 5])
s2 = set([1, 3, 5, 7, 9])
s3 = set([1, 2, 3])
 
print(s1 | s2) #　和集合を求める set([1, 2, 3, 4, 5, 7, 9])
print(s1 & s2) #　積集合を求める set([1, 3, 5])
print(s1 | s3) #　和集合を求める set([1, 2, 3, 4, 5])
print(s1 & s3) #　積集合を求める set([1, 2, 3])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 辞書 dict
# # from here 20240513

# In[ ]:


# IDと年齢の辞書
d = {'Yamada': 30, 'Uemura': 40, 'Tanaka': 70}


# In[ ]:


d1 = d['Yamada']


# In[ ]:


d1


# In[ ]:


d['Tanaka']=80
d


# In[ ]:


d[1]   #この形の辞書は順番がない


# In[ ]:


d2 = {1: 30, 2: 40, 3: 70}


# In[ ]:


d2[1] #こちらの１は何番目の順番の数字ではない、indexインデックス（索引）のこと


# In[ ]:


d['Honda']=90
d


# In[ ]:


new_dict = [('Honda', 50), ('Ueda', 20)]
d.update(new_dict)
d


# In[ ]:


new_dict1 = {'Honda': 90, 'Ueda': 20}
d.update(new_dict1)
d


# # フローチャートとは

# <pre>
# フローチャート（flowchart）とは、プログラムの流れを設計するための図解術です。
# プログラムの概要/詳細ロジックの設計、可視化
# 
# 記号：楕円ー＞開始/終了記号
#          長方形ー＞処理記号
#          菱形ー＞判断記号
#          平行四辺形ー＞データ/入出力記号
# 例：
# <pre>

# In[ ]:


name = input("Enter your name: ")
print(name)


# In[ ]:


Lunch = input("今日の昼ごはんは何を食べましたか？ ")
print(Lunch)


# In[ ]:





# In[ ]:


question1 = input("こんにちは、ご用件はなんでしょうか？ ")


# In[ ]:


if question1=="今日の昼ごはんは何を食べようかな？決めてくれる？":
    import random

    # List of items
    items = ['唐揚げ', '家系ラーメン', '二郎系ラーメン', '韓国焼肉', 'スパゲッティ', '豚生姜焼き']

    # Select a random item
    selected_item = random.choice(items)

    print(selected_item)
    print("今日の昼ごはんは"+selected_item+"食べましょうか？いかがでしょうか?")


# In[ ]:


# 上記二つセルのコードを統一して実行してみると

question1 = input("こんにちは、ご用件はなんでしょうか？\n") #\nは改行コード
# print(question1)
print("*********************************************")
if question1=="今日の昼ごはんは何を食べようかな？決めてくれる？":
    import random

    # List of items
    items = ['唐揚げ', '家系ラーメン', '二郎系ラーメン', '韓国焼肉', 'スパゲッティ', '豚生姜焼き']

    # Select a random item
    selected_item = random.choice(items)

    # print(selected_item)
    print("はい、かしこまりました。今日の昼ごはんは"+selected_item+"食べましょうか？いかがでしょうか?")


# In[ ]:





# In[ ]:


# バージョン情報を調べる

# 方法１
# !pip3 list 

# 方法２
# import pandas
# print(pandas.__version__)


# In[ ]:


import ipywidgets as wg
from IPython.display import display
myName = wg.Text(description='ID:')
Lucky_number = wg.Text(description='Lucky number:')
display(myName, Lucky_number)


# In[ ]:


### new cell
print(myName.value + "'s lucky number is " + str(Lucky_number.value) )


# In[ ]:





# # if else elif 

# In[ ]:


# '''
# if 条件1:
#     処理1を実行...
# [elif 条件2:
#     処理2を実行...]
# [else:
#     処理3を実行...]
# '''
# 
# example 1: 80=<x<100 "優秀"
# example 2: 80=<x<100 "優秀"　60=<x<80 "良好"
# example 3: 80=<x<100 "優秀"　60=<x<80 "良好"  x<60 "頑張りましょう！"
# example 4: x>=100 "満点"　80=<x<100 "優秀"　60=<x<80 "良好"  x<60 "頑張りましょう！"


# In[ ]:


if 80=<x<100:
    print("優秀")


# In[ ]:


if x in range(80,100):
    print("優秀")


# In[ ]:


x=85
if x in range(80,100):
    print("優秀")


# In[ ]:


# example 2: 80=<x<=100 "優秀"　60=<x<80 "良好"
x=65
if x in range(80,101):
    print("優秀")
elif x in range(60,80) :
    print("良好")

# ここの「elif」行のインデントがないことを気を付けましょう！インデントがあるとエラーが出る。


# In[ ]:


# example 3: 80=<x<=100 "優秀"　60=<x<80 "良好"  x<60 "頑張りましょう！"

x=35
if x in range(80,101):
    print("優秀")
elif x in range(60,80) :
    print("良好")
else:
    print("頑張りましょう！")


# In[ ]:


# example 4: x>=100 "満点"　80=<x<100 "優秀"　60=<x<80 "良好"  x<60 "頑張りましょう！"

x=85
if x in range(80,100):
    print("優秀")
elif x in range(60,80) :
    print("良好")
elif x in range(0,60) :
    print("頑張りましょう！")
elif x >=100 :
    print("満点")


# In[ ]:





# # while else

# In[ ]:


# '''
# while 条件1:
#     処理1を実行...
# [else:
#     処理2を実行...]
# '''


# In[ ]:


n = 0
while n < 5:
    print(n)
    n += 1


# In[ ]:


n = 0
while n < 5:
    print(n)
    n += 1
else:
    print('END')


# In[ ]:





# # for in

# In[ ]:


# '''
# for var in 条件1:
#     処理1を実行...
# [else:
#     処理2を実行...]
# '''


# In[ ]:


for i in range(5):
    print(i)


# In[ ]:


for i in range(5):
    print("Hello World!")


# In[ ]:


for n in [1, 2, 3]:
    print(n) 


# In[ ]:


for a in "ABC":
    print(a) 


# In[ ]:


for k in {'one': 1, 'two': 2, 'three': 3}:
    print(k) 


# In[ ]:


aaa={'one': 3, 'two': 5, 'three': 7}
for k in aaa:
    print(k)
    print(aaa[k]) 


# In[ ]:


aaa={'one': 3, 'two': 5, 'three': 7}
for k in aaa:
    print(k+':'+str(aaa[k])) 
else:
    print('END')


# In[ ]:





# In[ ]:


# print([x * 2 for x in a])  #=> [2, 4, 6] : 内包表記
# をfor loopの形で書き換えると、

list=[]
for x in a:
    print(x*2)
    list.append(x*2)
    print(list)


# In[ ]:





# In[ ]:


#　map()はリストの各要素をすべて同じ処理を行い、処理した結果を返す

a = [1, 2, 3]

def double(x): return x * 2

print(list(map(double, a)))                  #=> [2, 4, 6] : 関数方式
print(list(map(lambda x: x * 2, a)))         #=> [2, 4, 6] : lambda方式
print([x * 2 for x in a])                    #=> [2, 4, 6] : 内包表記


# In[ ]:





# # break, continue

# In[ ]:


# break: 繰り返し処理を抜け出す
for n in range(10):
    if n == 5:
        break
    print(n)                 # 0, 1, 2, 3, 4


# In[ ]:


# continue: 繰り返し処理をスキップ
for n in range(10):
    if n == 5:
        continue
    print(n)                 # 0, 1, 2, 3, 4, 6, 7, 8, 9


# In[ ]:


# pass: 何も処理を行わない
for n in range(10):
    if n == 5:
        pass
    print(n)   


# In[ ]:





# In[ ]:


#一階微分
from sympy import *
import numpy as np
x = Symbol('x')
y = x**2 + 1
yprime = y.diff(x)
print(yprime)


# In[ ]:


#さらに二階微分
yprimeprime = yprime.diff(x)
print(yprimeprime)


# In[ ]:





# In[ ]:


# !pip3 install gtts


# In[ ]:


from gtts import gTTS
gTTS('Welcome To InterviewBit').save('interviewbit.mp3')


# In[ ]:


from gtts import gTTS
import os

file=open("Python.txt", "r", encoding='utf-8').read()
speech = gTTS(text=file, lang="ja", slow=False) #"en"
speech.save("voice.mp3")
# os.system("voice.mp3")


# In[ ]:





# In[4]:


import random
print("数字を当ててポイントを稼ぎるゲームです")
print("あたると100ポイント獲得、外れると50ポイントなくなる")
a=random.randint(1,3)
score = 100 
while True:
    b=int(input("1から3までの整数を当ててください"))
    if (a==b):
        score+=100
        print("おめでとうございます！今のポイントは", score)
        break
    elif(b<a):
        score=score-50
        print("もっと大きく当ててみて！今のポイントは", score)
    else:
        score=score-50
        print("もっと小さく当ててみて！今のポイントは", score)
       


# # 下の練習をやってみてください！
# 
# # 練習１
# 宅急便のサイズによって、料金が変わるシステム、まずinput()を使ってサイズを確認しましょう。 \
# そのあと、100サイズ以下(<=100)の場合：「郵送料は1000円です」をプリントアウト、 \
#             100-140サイズ(100<x<=140]の場合：「郵送料は1200円です」をプリントアウト、 \
#             140-160サイズ(140<x<=160]の場合：「郵送料は1500円です」をプリントアウト 
# 
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
# 一つ違ったら「もう一回正しいIDとパスワードを入力してください」をプリントアウト

# In[ ]:




