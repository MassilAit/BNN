from itertools import combinations_with_replacement

max_total=8
for total in range(4, max_total + 1):
        for h1 in range(2, total - 1):
            h2 = total - h1
            if h2 <= h1:
                  print([h1,h2])