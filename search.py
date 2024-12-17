import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import time

def linear_search(x, arr): #линейный поиск
    for i in range(0,len(arr)):
        if arr[i]==x:
            return i
    return -1

def binary_search(x, arr): #бинарный поиск
    start = 0
    end = len(arr) - 1
    while start <= end:
        mid = (start + end) // 2
        if arr[mid] == x:
            return mid
        elif x > arr[mid]:
            start = mid + 1
        else:
            end = mid - 1
    return -1

def jump_search(x, arr): #поиск прыжками
    arr_len = len(arr)
    step = int(arr_len**0.5)
    prev = 0
    while arr[min(step, arr_len) - 1] < x:
        prev = step
        step += int(arr_len**0.5)
        if prev >= arr_len:
            return -1
    for i in range(prev, min(step, arr_len)):
        if arr[i] == x:
            return i
    return -1

def interpolationSearch(arr, n, x):
    low = 0
    high = (n - 1)  
    while low <= high and x >= arr[low] and x <= arr[high]: 
        if low == high: 
            if arr[low] == x: 
                return low
            return -1
        pos = int(low + (((float(high - low)/( arr[high] - arr[low])) * (x - arr[low])))) 
        if arr[pos] == x: 
            return pos 
        if arr[pos] < x: 
            low = pos + 1
        else: 
            high = pos - 1
    return -1

def exponential_search(x, arr): #экспоненционный поиск
    if len(arr) == 0:
        return -1
    if arr[0] == x:
        return 0
    index = 1
    while index < len(arr) and arr[index] < x:
        index *= 2
    start = index//2
    end = min(index, len(arr))
    while start <= end:
        mid = (start + end) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            start = mid + 1
        else:
            end = mid - 1
    return -1


#main
a=[]
b=[]
#--------------------------линейный поиск
elems = [i for i in range(50000,10050000,500000)]
results = []
for elem in elems:
    arr = [i for i in range(1,elem+1)]
    start = time.perf_counter()
    _ = linear_search(elem,arr)
    end = time.perf_counter()
    results.append(end-start)
    print(f"{elem} - {end-start:.4f}")
x = np.array(elems).reshape(-1,1)
y = np.array(results)
model = LinearRegression()
model.fit(x, y)
x_pred = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_pred = model.predict(x_pred)
plt.scatter(x, y, color='blue', label='Экспериментальные точки')
plt.plot(x_pred, y_pred, color='red', label='Регрессионная кривая')
plt.title('Линейный поиск')
plt.xlabel('Количество элементов')
plt.ylabel('Время поиска')
plt.legend()
plt.grid()
a.append(x_pred)
b.append(y_pred)
plt.show()
# #------------------------бинарный поиск
elems1 = [i for i in range(500000,10500000,500000)]
results = []
for elem in elems1:
    arr_normal = [i for i in range(1, elem+1)]
    arr = [i for i in arr_normal for _ in range(10)]
    start = time.perf_counter()
    _ = binary_search(elem//2,arr)
    end = time.perf_counter()
    results.append(end-start)
    print(f"{elem} - {end-start:.7f}")
x = np.array(elems1).reshape(-1,1)
y = np.array(results)
x_log = np.log(x)
model = LinearRegression()
model.fit(x_log, y)
org = model.coef_
model.coef_ = abs(model.coef_)
x_pred = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_pred = model.predict(np.log(x_pred))
a.append(x_pred)
b.append(y_pred)
model.coef_ = org
x_pred = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_pred = model.predict(np.log(x_pred))
plt.scatter(x, y, color='blue', label='Экспериментальные точки')
plt.plot(x_pred, y_pred, color='red', label='Регрессионная кривая')
plt.title('Бинарный поиск')
plt.xlabel('Количество элементов')
plt.ylabel('Время поиска')
plt.legend()
plt.grid()
plt.show()
#-------------------------поиск прыжком
results = []
for elem in elems:
    arr = [i for i in range(1,elem+1)]
    start = time.perf_counter()
    _ = jump_search(elem,arr)
    end = time.perf_counter()
    results.append(end-start)
    print(f"{elem} - {end-start:.5f}")
x = np.array(elems).reshape(-1,1)
y = np.array(results)
x_sqrt = np.sqrt(x)
model = LinearRegression()
model.fit(x_sqrt, y)
org = model.coef_
model.coef_ = abs(model.coef_)
x_pred = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_pred = model.predict(np.sqrt(x_pred))
a.append(x_pred)
b.append(y_pred)
model.coef_ = org
x_pred = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_pred = model.predict(np.sqrt(x_pred))
plt.scatter(x, y, color='blue', label='Экспериментальные точки')
plt.plot(x_pred, y_pred, color='red', label='Регрессионная кривая')
plt.title('Прыжковый поиск')
plt.xlabel('Количество элементов')
plt.ylabel('Время поиска')
plt.legend()
plt.grid()
plt.show()
#-------------------------интерполяционный поиск(худший случай)
results = []
for elem in elems1:
    arr_normal = [i for i in range(1, elem+1)]
    arr = [i for i in arr_normal for _ in range(10)]
    start = time.perf_counter()
    _ = interpolationSearch(arr,len(arr),elem-1)
    end = time.perf_counter()
    results.append(end-start)
    print(f"{elem} - {end-start:.7f}")
x = np.array(elems1).reshape(-1,1)
y = np.array(results)
x_log2 = np.log2(x)
x_log2_log2 = np.log2(x_log2)
model = LinearRegression()
model.fit(x, y)
x_pred = np.linspace(min(elems1), max(elems1), 100).reshape(-1, 1)
y_pred = model.predict(x_pred)
plt.scatter(x, y, color='blue', label='Экспериментальные точки')
plt.plot(x_pred, y_pred, color='red', label='Регрессионная кривая')
plt.title('Интерполяционный поиск в худшем случае')
plt.xlabel('Количество элементов')
plt.ylabel('Время поиска')
plt.legend()
plt.grid()
plt.show()
#-------------------------интерполяционный поиск(средний случай)
results = []
for elem in elems1:
    arr_normal = [i for i in range(1, elem+1)]
    arr = [i for i in arr_normal for _ in range(10)]
    start = time.perf_counter()
    _ = interpolationSearch(arr,len(arr),elem//2)
    end = time.perf_counter()
    results.append(end-start)
    print(f"{elem} - {end-start:.7f}")
x = np.array(elems1).reshape(-1,1)
y = np.array(results)
x_log2 = np.log2(x)
x_log2_log2 = np.log2(x_log2)
model = LinearRegression()
model.fit(x_log2_log2, y)
org = model.coef_
model.coef_ = abs(model.coef_)
x_pred = np.linspace(min(elems1), max(elems1), 100).reshape(-1, 1)
x_pred_log2 = np.log2(x_pred)
x_pred_log2_log2 = np.log2(x_pred_log2)
y_pred = model.predict(x_pred_log2_log2)
a.append(x_pred)
b.append(y_pred)
model.coef_ = org
x_pred = np.linspace(min(elems1), max(elems1), 100).reshape(-1, 1)
x_pred_log2 = np.log2(x_pred)
x_pred_log2_log2 = np.log2(x_pred_log2)
y_pred = model.predict(x_pred_log2_log2)
plt.scatter(x, y, color='blue', label='Экспериментальные точки')
plt.plot(x_pred, y_pred, color='red', label='Регрессионная кривая')
plt.title('Интерполяционный поиск в среднем случае')
plt.xlabel('Количество элементов')
plt.ylabel('Время поиска')
plt.legend()
plt.grid()
plt.show()
#------------------экспоненциальный поиск
results = []
for elem in elems1:
    arr_normal = [i for i in range(1, elem+1)]
    arr = [i for i in arr_normal for _ in range(10)]
    start = time.perf_counter()
    _ = exponential_search(elem//2,arr)
    end = time.perf_counter()
    results.append(end-start)
    print(f"{elem} - {end-start:.7f}")
x = np.array(elems1).reshape(-1,1)
y = np.array(results)
x_log = np.log(x)
model = LinearRegression()
model.fit(x_log, y)
org = model.coef_
model.coef_ = abs(model.coef_)
x_pred = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_pred = model.predict(np.log(x_pred))
a.append(x_pred)
b.append(y_pred)
model.coef_ = org
x_pred = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_pred = model.predict(np.log(x_pred))
plt.scatter(x, y, color='blue', label='Экспериментальные точки')
plt.plot(x_pred, y_pred, color='red', label='Регрессионная кривая')
plt.title('Экспоненциальный поиск')
plt.xlabel('Количество элементов')
plt.ylabel('Время поиска')
plt.legend()
plt.grid()
plt.show()
#итоговый ответ
colors = ["red","green","orange","maroon","blue"]
strs = ["Линейный", "Бинарный", "Прыжковый", "Интерполяционный", "Экспоненциальный"] 
for i in range(0,5):
    #выравниваем графики для правильного отображания на одной картинке
    x_off1 = a[i] - a[i][0]
    y_off1 = b[i] - b[i][0]
    plt.plot(x_off1, y_off1, color=colors[i], label=f"{strs[i]} поиск")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Итоговый результат")
plt.xlabel("Количество элементов")
plt.ylabel("Время поиска")
plt.legend()
plt.grid()
plt.show()
