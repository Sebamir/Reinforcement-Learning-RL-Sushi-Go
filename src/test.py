nigiri_list = [
    [1,1,1,1],
    [2]
]
max_maki= max(sum(x) for x in nigiri_list)

print(max_maki)

winners = []

for i, count in enumerate(nigiri_list):
    if sum(count) == max_maki:
        sum_count= sum(count)
        print(sum_count)
        winners.append(i)

          
print (winners)