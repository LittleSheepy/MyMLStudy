x, y = 0, 0
for item in input().split(";"):
    if not 2 <= len(item) <= 3:
        continue
    try:
        step = int(item[1:])
    except ValueError:
        continue
    direction = item[0]
    if direction == "A":
        x -= step
    elif direction == "D":
        x += step
    elif direction == "S":
        y -= step
    elif direction == "W":
        y += step
print(f"{x},{y}")
