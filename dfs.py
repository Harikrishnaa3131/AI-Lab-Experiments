from collections import defaultdict

class Graph:

	def __init__(self, vertices):
		self.V = vertices
		self.graph = defaultdict(list)
	def addEdge(self, u, v):
		self.graph[u].append(v)

	def printAllPath_012(self, u, d, visited, path):
		visited[u]= True
		path.append(u)
		if u == d:
			print (path)
		else:
			for i in self.graph[u]:
				if visited[i]== False:
					self.printAllPath_012(i, d, visited, path)
		path.pop()
		visited[u]= False

	def printAllPaths_012(self, s, d):
	    
		visited =[False]*(self.V)
		path = []
		self.printAllPath_012(s, d, visited, path)



g = Graph(6)
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(0, 3)
g.addEdge(2, 0)
g.addEdge(2, 1)
g.addEdge(1, 3)
g.addEdge(2, 5)
g.addEdge(3, 5)
g.addEdge(3, 4)

s = 2 ; d = 5
print ("Following are all different paths from % d to % d :" %(s, d))
g.printAllPaths_012(s, d)