#!/usr/bin/env python
# coding: utf-8

# In[29]:


import ipywidgets as wg 
from IPython.display import display

ID_PW_dict = {} 

# i=1
# while i in range(4): 
for i in range(1,4): 
    ID = input(f"{i}番目のID：") #IDを取得
    PW = input(f"{i}番目のPassword：")#Passwordを取得
    
    ID_PW_dict[ID]=PW #取得したIDとパスワードを辞書に追加
#     dict_temp = {ID: PW} #取得したIDとパスワードのペアを一時的な辞書dict_tempを作成
#     ID_PW_dict.update(dict_temp) #一時的な辞書dict_tempをID_PW_dict辞書に追加

#     i += 1
print(ID_PW_dict)


# In[30]:


ID_guest = wg.Text(description='ID:') #IDを入力
PW_guest = wg.Text(description='pw:') #パスワードを入力
display(ID_guest, PW_guest)


# In[31]:


if ID_guest.value in list(ID_PW_dict.keys()):
    if ID_PW_dict[ID_guest.value] == PW_guest.value: 
        print("Welcome") 
    else: 
        print("もう一回正しいIDとパスワードを入力してください") 
else:
    print('You have not yet registered!') #まだ登録されていない


# In[33]:


# # #　上のセルを実行する時、エラーが出た方はipywidgetsのバージョン問題があり、下記の予備コードをコメント解除してから使ってください。
# ID_guest = input('ID:') #IDを入力
# PW_guest = input('pw:') #パスワードを入力
# print(ID_guest, PW_guest)

# if ID_guest in list(ID_PW_dict.keys()):
#     if ID_PW_dict[ID_guest] == PW_guest: 
#         print("Welcome") 
#     else: 
#         print("もう一回正しいIDとパスワードを入力してください") 
# else:
#     print('You have not yet registered!') #まだ登録されていない


# In[ ]:




