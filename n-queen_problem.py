from itertools import permutations
N=int(input("Enter the value for n:")) 
sol=0
cols = range(N)
for combo in permutations(cols):
    if N==len(set(combo[i]+i for i in cols))==len(set(combo[i]-i 
for i in cols)):
        sol += 1
        print('Solution '+str(sol)+': '+str(combo)+'\n')
        print("\n".join(' x ' * i + ' Q ' + ' x ' * (N-i-1) for i in combo) + "\n\n\n\n")