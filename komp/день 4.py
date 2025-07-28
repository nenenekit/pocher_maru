import random
угадайка=random.randint (1,10)
попытки=3
for i in range (попытки):
    огурец = int(input("угадай"))
    if огурец==угадайка:
        print("победа")
        exit()
    elif огурец>угадайка:
        print("меньше")
    elif огурец<угадайка:
        print("больше")
print("вы проиграли")