def square(x) :            # 计算平方数
    print("square")
    return x ** 2

map_results = map(square, [1,2,3,4,5])
list_results = map(list, zip(map_results))
tuple_results = tuple(list_results)
print("result", tuple_results)


