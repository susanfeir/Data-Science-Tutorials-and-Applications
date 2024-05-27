#!/usr/bin/env python
# coding: utf-8

# ![screenshot.png](attachment:screenshot.png)

#method1

score = int(input('点数: '))

if score in range(0,101):
        if score <60:
            print("当該点数の評価記号はFです")
        elif score <65:
            print("当該点数の評価記号はC-です")
        elif score <70 :
            print("当該点数の評価記号はCです")
        elif score <80 :
            print("当該点数の評価記号はBです")
        elif score <95 :
            print("当該点数の評価記号はAです")
        else:
            print("当該点数の評価記号はA+です")
else:
    print("0から100までの整数を入力してください")


# In[25]:


# #method2
#
# score = int(input('点数: '))
#
# point = [95,  80,  70,  65,  60,  0]
# grade = ["A+", "A", "B", "C","C-", "F"]
#
#
# if 0 <= score <= 100:
#     i=0
#     while i < len(point):
#         if score >= point[i]:
#             print(f"当該点数:{score}点の評価記号は{grade[i]}です。")
#             break
#         else:
#             i += 1
# else:
#     print("0から100までの整数を入力してください")


# #method3
#
# score = int(input('点数: '))
#
# point = [60, 65, 70, 80, 95, 100]
# grade = ["F", "C-", "C", "B", "A", "A+"]
#
#
# if 0 <= score <= 100:
#     i=0
#     while i < len(point):
#         if score < point[i]:
#             print(f"当該点数:{score}点の評価記号は{grade[i]}です。")
#             break
#         else:
#             i += 1
# else:
#     print("0から100までの整数を入力してください")
#





