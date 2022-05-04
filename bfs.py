from typing import List
from collections import deque
 

def printpath_012(path: List[int]) -> None:
     
    size = len(path)
    for i in range(size):
        print(path[i], end = " ")
    print()
 
def isNotVisited_012(x: int, path: List[int]) -> int:
 
    size = len(path)
    for i in range(size):
        if (path[i] == x):
            return 0
    return 1
 
def findpaths_012(g: List[List[int]], src: int,
              dst: int, v: int) -> None:
                   
    q = deque()
    path = []
    path.append(src)
    q.append(path.copy())
     
    while q:
        path = q.popleft()
        last = path[len(path) - 1]
        if (last == dst):
            printpath_012(path)
        for i in range(len(g[last])):
            if (isNotVisited_012(g[last][i], path)):
                newpath = path.copy()
                newpath.append(g[last][i])
                q.append(newpath)
 
if __name__ == "__main__":
     
    v = 6
    g = [[] for _ in range(6)]
    g[0].append(3)
    g[0].append(1)
    g[0].append(2)
    g[1].append(3)
    g[2].append(0)
    g[2].append(1)
    g[2].append(5)
    g[3].append(4)
    g[3].append(5)
 
    src = 2
    dst = 5
    print("path from src {} to dest {} are".format(
        src, dst))
    findpaths_012(g, src, dst, v)