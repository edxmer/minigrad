import random

# Light to dark
ascii_array = [' ', '.', ',', '-', '~', ':', ';', '=', '!', '•', '*', 'o', 'O', '#', '$', '@', 'M']

with open("optdigits.tra", "r", encoding="utf-8") as f:
    lines = f.readlines()
    values = [[int(x) for x in l.strip().split(',')] for l in lines]

id = random.randint(0, len(ascii_array)-1)

print(len(values))

print('Number: ', values[id][64])
xs = values[id][0:64]
for y in range(8):
    for x in range(8):
        print(ascii_array[xs[y*8+x]]*3, end='')
    print()