# -*- coding: utf-8 -*-
"""
Created on Fri May 12 01:02:47 2017

@author: sonam
"""

'''
Question 1:
Given two strings s and t, determine whether some anagram of t is a substring of s.
 For example: if s = "udacity" and t = "ad", then the function returns
True. Your function definition should look like: question1(s, t) and return a boolean True or False.

input-output:
Input: String s, and string t
Output: True or False based on the validity of substring t with regard to String s

Test possible cases:
input s ="udacity", t = "ad" (valid case) should ouput True
input t or s as None
t does not contain characters found in s
s = "udacity", t = "a" - Single character input in t
t or s has special characters (numbers or symbols)
s = "udacity", t = "uc" - Invalid anagram. A word not in dictionary

Brainstorming:
Compare characters in t and s if there are any overlaps
Search each character in t with regard to s

References:

http://stackoverflow.com/questions/3788870/how-to-check-if-a-word-is-an-english-word-with-python
https://pymotw.com/2/collections/counter.html
http://stackoverflow.com/questions/8270092/python-remove-all-whitespace-in-a-string
http://pythex.org/
http://www.velvetcache.org/2010/03/01/looking-up-words-in-a-dictionary-using-python

'''

from nltk.corpus import wordnet
from collections import Counter
import re

def question1(s, t):
    
    # Verify for None input
    if s == None or t == None:
        return False
    # Verity for single character in input t
    if len(t) <= 1:
        return False
    
    # Verify for special characters (numbers, symbols).
    # Regex searches for any character except whitespace (\s), comma (\,) , period (\.) and alpha ([a-zA-Z]).
    if re.search('[^a-zA-Z*\s*]', s) or re.search('[^a-zA-Z*\s*]', t):
        return False
    
    # Anagram checker
    # Remove all whitespaces in strings and set all characters to lowercase
    s = s.replace(" ", "").lower()
    t = t.replace(" ", "").lower()

    m = Counter(s)
    n = Counter(t)
    
    # Finds intersection of Counter c and d (taking positive minimums) and compares if it still appears in d.
    # Checks for a valid anagram against WordNet from NLTK
    if m & n == n and wordnet.synsets(t):
        return True
    else:
        return False
        
print "Let's check the validity of question1(s, t) outputs for test cases or different inputs for s and t:"

print " **************** Question 1 ****************"
# input s ='udacity', t = 'ad' (valid case)
# Should be True
s = "udacity"; t = "ad"
print "Test Case 1 -", question1(s, t)

# t does not contain characters found in s
# Should be False
s = "udacity"; t = "an"
print "Test Case 2 -", question1(s, t)

# s = 'udacity', t = 'uy' - Invalid anagram. Not in dictionary
# Should be False
s = "udacity"; t = "ac"
print "Test Case 3 -", question1(s, t)

# characters of t should only be used once in s
# Should be False
s = "udacity"; t = "uu"
print "Test Case 4 -", question1(s, t)

# s = 'udacity', t = 'a' - Single character input in t
# Should be False
s = "udacity"; t = "y"
print "Test Case 5 -", question1(s, t)

# input t or s as None
# Should be False
s = "udacity"; t = None
print "Test Case 6 -", question1(s, t)

# t or s has special characters (numbers or symbols)
# Should be False
print "Test Case 7 -", question1("uda%city7", "c2")


'''
Question 2
Given a string a, find the longest palindromic substring contained in a.
Your function definition should look like question2(a), and return a string.

Input: String a
Output: String - Longest palindromic substring of a
Test cases:
input is None
input has no palindromes
input has a palindrome
input has two or more palindromes, but of equal length
input has an even palindrome
input has a mix of symbols, numbers and whitespaces, with a valid palindrome
input is a sentence palindrome, where punctuation, capitalization, and spaces are usually ignored

Brainstorming:
Iterate over string a
For each character in string a, search one character to its adjacent left and right
If palindrome is found, store in memory and search two characters adjacent, three, etc until no more palindrome

Reference:
http://stackoverflow.com/questions/17331290/how-to-check-for-palindrome-using-python-logic
http://pythex.org/
http://stackoverflow.com/questions/16343849/python-returning-longest-strings-from-list

'''

def question2(a):
    
    # validity for None input r=l, k=m, i=n
    if a == None:
        return False
    
    # Remove punctuations [,.!':;?-], whitespaces (\s) and sets all characters to lowercase.
    a = re.sub("\s*[,.!':;?-]*", "", a).lower()
    
    # Check for even and/or odd palindromes
    def palinCheck(window, l, m, n, a, min_valid, odd_ind):
        #print window, r, k, i, a
        # While the indices are within bounds of the string
        while (n - m + 1) >= 0 and (n + m - 1) <= len(a):  
            # Checks the reverse of the subset of the string with [::-1]
            # Pass in min_valid = 2 for even palindromes and min_valid = 3 for odd
            if str(window) == str(window[::-1]) and len(window) >= min_valid:
                l.append(window)
                m += 1 # Increments index to search for adjacent characters to the right and left
                # Increase the window spread. odd_ind = 1 for odd palindromes
                window = a[n - m: n + m + odd_ind]
                palinCheck(window, l, m, n, a, min_valid, odd_ind) # Recursion until all possible palindromes are found
            return l

    l = [] # Stores any found palindrome
    m = 1 # Counter for adjacent character search
    
    # Starts with index 1, Ends with index -1
    # Even palindrome checker
    for n in range(1, len(a) - 1):
        # Min palindrome to search for is 2 characters in length
        window = a[n - m: n + m]
        palinCheck(window, l, m, n, a, min_valid=2, odd_ind=0)
        
    # Odd palindrome checker
    for n in range(1, len(a) - 1):
        # Min palindrome to search for is 3 characters in length
        window = a[n - m: n + m + 1]
        palinCheck(window, l, m, n, a, min_valid=3, odd_ind=1)
        
    # If there are no palindromes found, return False
    # Else, return the longest palindrome(s)
    if len(l) == 0:
        return False
    else:
        lp = max(len(y) for y in l)
        longest_palin = [x for x in l if len(x) == lp]
        return longest_palin
        
print "call question2(a) with different a values:"

print " **************** Question 2 ****************"
# input is None
# Should be False
print "Test Case A -", question2(None)

# input has no palindromes
# Should be False
print "Test Case 2 -", question2("robotics")

# input has a palindrome
# Should be "civic"
print "Test Case 3 -", question2("civic")

# input has a mix of symbols, numbers and whitespaces, with a valid palindrome
# Should be "level"
print "Test Case 4 -", question2("s3f nt!@ofhleveltglr%hn,n6s")

# input is a sentence palindrome, where punctuation, capitalization, and spaces are usually ignored
# Should be "wasitacaroracatisaw"
print "Test Case 5 -", question2("Was it a car or a cat I saw?")

# input has an even palindrome
# Should be "liveontimeemitnoevil"
print "Test Case 6 -", question2("Live on time, emit no evil")

# input has two or more palindromes, but of equal max length
# Should be "rotor" and "kayak"
print "Test Case 7 -", question2("rotorkayak")


'''
Question 3
Given an undirected graph G, find the minimum spanning tree within G. A minimum 
spanning tree connects all vertices in a graph with the smallest possible total weight of edges.
Your function should take in and return an adjacency list structured like this:
{'A': [('B', 2)], 'B': [('A', 2), ('C', 5)], 'C': [('B', 5)]}
Vertices are represented as unique strings. The function definition should be question3(G)

Input: Adjacency list of graph G
Output: Adjacency list - min spanning tree
Test cases:
input is None
input has a min spanning tree
graph is disconnected
Brainstorming:
Choose an arbitary vertex, v 
Then, choose an edge that has smallest weight and grow the tree
Repeat until minimum spanning tree is obtained

Reference:

http://stackoverflow.com/questions/3282823/get-key-with-the-least-value-from-a-dictionary

http://www.stoimen.com/blog/2012/11/19/computer-algorithms-prims-minimum-spanning-tree/

The above adjacency list returned by the function means that:
The weight of edge between:
A and B is 2
B and A is 2
B and C is 5
C and B is 5
Therefore the edge weight between A and B is less than that of B and C in this particular graph 

'''
import numpy as np

# Undirected graph
G = {'A': [('B', 4), ('E', 3)],
     'B': [('A', 4), ('C', 8), ('D', 2), ('E', 2)], 
     'C': [('B', 8), ('D', 1), ('E', 6)],
     'D': [('B', 2), ('C', 1)],
     'E': [('A', 3), ('B', 2), ('C', 6)]}

'''
G Graph Visualization

A--4--B--8--C       
|   / |    /|       
3  2  2   1 |       
| /   |  /  |        
 E     D    |       
 |          |       
 +----6-----+  
 
 '''

# Disconnected graph
DG = {'A': [('B', 1)],
     'B': [('A', 1)], 
     'C': [('D', 5)],
     'D': [('C', 5)]}


def question3(gr):
    # Reject None input
    if gr == None:
        return False
    
    # Initialize
    PQ = {} # Priority queue
    the_P = {} # Parent

    # Select a vertex arbitrarily as the root
    root = np.random.choice(gr.keys())
    ##root = 'DG'
    ##print "Root vertex:", root

    # Set priority of each member in PQ to approx infinity
    for v in gr:
        PQ[v] = 1e9
    # Set priority of starting vertex to 0
    PQ[root] = 0
    
    # Set parent of starting vertex to null
    the_P[root] = None
    
    ##print PQ
    ##print the_P
    
    # Prim's algorithm
    
    while PQ:
        # Get minimum from Q. u=(key, priority value)
        u = min(PQ.items(), key=lambda x: x[1])
        ##print "u: ", u
        # Initialize list to store neighboring vertices
        temp = []
        # Check all neighbor vertices to u
        for v in gr[u[0]]:
            ##print "v: ", v
            # If the vertex is found in PQ and its weight is less than the priority...
            # v=(key, weight value)
            if v[0] in PQ and v[1] < PQ[v[0]]:
                temp.append(v)
                # Add u as the parent vertex of v
                ## the_P[u[0]] = [v]
                the_P[u[0]] = temp
                # And add the weight value as the new priority
                PQ[v[0]] = v[1]
                ##print "the_P: ", the_P
        # Remove u from PQ
        PQ.pop(u[0])
        ##print "PQ new: ", PQ
    return the_P
        
# print question3(DG) # graph disconnected
# print question3(G) # undirected G

print "call question3(G) with different vertices and different edge weights"

print " **************** Question 3 ****************"
# input is None
# Should be False
print "Test Case 1 -", question3(None)

# input has a min spanning tree
print "Test Case 2 -", question3(G)

# input graph is disconnected
print "Test Case 3 -", question3(DG)

'''
Question 4:
Find the least common ancestor between two nodes on a binary search tree.
The least common ancestor is the farthest node from the root that is an ancestor
of both nodes. For example, the root is a common ancestor of all nodes on the tree,
but if both nodes are descendents of the root's left child, then that left child
might be the lowest common ancestor. You can assume that both nodes are in the tree,
and the tree itself adheres to all BST properties. The function definition should
 look like question4(T, r, n1, n2), where T is the tree represented as a matrix, 
 where the index of the list is equal to the integer stored in that node and a 1 
 represents a child node, r is a non-negative integer representing the root, and 
 n1 and n2 are non-negative integers representing the two nodes in no particular
 order. For example, one test case might be:
 
question4([[0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [1, 0, 0, 0, 1],
     [0, 0, 0, 0, 0]],
    3,
    1,
    4)
and the answer would be 3.

'''

### T[i][j] = 1, where i is an ancestor to j

'''
Tree representation
    3
   / \
  0   4
 /
1

Input: Matrix of BST, root node, node 1, node 2
Output: (Integer) least common ancestor of both nodes
Test cases:
input is None
input is not a valid matrix
input has no LCA
input has LCA
Brainstorming:
Create BST from matrix, via insertion
Search for LCA using single traversal

Reference:

http://blog.rdtr.net/post/algorithm/algorithm_tree_lowest_common_ancestor_of_a_binary_tree/
http://www.geeksforgeeks.org/lowest-common-ancestor-binary-tree-set-1/
http://yucoding.blogspot.my/2016/04/leetcode-question-lowest-common.html
(http://www.ritambhara.in/build-binary-tree-from-ancestor-matrics/

'''

# Balanced tree
K = [[0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0]]

'''
K tree visualization
     3
   /   \
  2     5
 / \   / \
0   1 4   6
'''
# Invalid tree with values other than 0 or 1
J = [[0, 1, 0, 0, -1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 2]]

# Lowest Common Ancestor
def question4(T, r, n1, n2):
    # Handle None input
    if T == None or r == None or n1 == None or n2 == None:
        return "None input entered."
    
    # Construct dict of {node: parent}
    node_parent_map = {}
    for i in range(len(T)):
        for j in range(len(T)):
            # Handle invalid matrix
            if T[i][j] < 0 or T[i][j] > 1:
                return "Invalid matrix entered."
            # Add key:value pair if a node-parent relationship exists
            # T[i][j] == 1, where i is an ancestor to j
            elif T[i][j] == 1:
                node_parent_map[j] = i
    
    # Helper to find child nodes in a BST
    def find_child_node(node_parent_map, current):
        temp = []
        for node, parent in node_parent_map.iteritems():
            if parent == current:
                temp.append((node, parent))
        return temp

    # Traverse tree from root node
    # Initialize current node as the root node 
    # and two null variables (nt1, nt2) as checks for the respective target nodes (n1, n2)
    current = r
    nt1 = None
    nt2 = None
    # While the checks are not equal to the target nodes
    while nt1 != n1 and nt2 != n2:
        # Returns LCA when the target nodes are in both left and right subtrees
        if n1 < current and n2 > current:
            nt1 = n1
            nt2 = n2
            return current
        # If the target nodes are in the left subtree...
        elif n1 < current and n2 < current:
            # Finds all child nodes of the current node
            temp = find_child_node(node_parent_map, current)
            # Check if LCA is found
            if len(temp) == 0:
                return "No LCA found within specified root."
            elif temp[0][0] == n1 and temp[1][0] == n2:
                nt1 = n1
                nt2 = n2
                return current
            else:
                # Assigns current node to left child node
                current = temp[0][0]
        # If the target nodes are in the right subtree...
        elif n1 > current and n2 > current:
            # Finds all child nodes of the current node
            temp = find_child_node(node_parent_map, current)
            # Check if LCA is found
            if len(temp) == 0:
                return "No LCA found within specified root."
            elif temp[0][0] == n1 and temp[1][0] == n2:
                nt1 = n1
                nt2 = n2
                return current
            else:
                # Assigns current node to right child node
                current = temp[1][0]
        else:
            return None

print "call question4(T, r, n1, n2) with different input cases:"

print " **************** Question 4 ****************"
# input is None
# Should be "None input entered."
print "Test Case 1 -", question4(None, 3, 1, 4)

# input is an invalid matrix
# Should be "Invalid matrix entered."
print "Test Case 2 -", question4(J, 3, 1, 4)

# input has no LCA
# Should be "No LCA found with specified root."
print "Test Case 3 -", question4(K, 6, 1, 4)

# input has a LCA
# Should be "2"
print "Test Case 4 -", question4(K, 3, 0, 1)

'''
Question 5
Find the element in a singly linked list that's m elements from the end. For example, 
if a linked list has 5 elements, the 3rd element from the end is the
3rd element. The function definition should look like question5(ll, m), where ll is 
the first node of a linked list and m is the "m'th number from the end".
You should copy/paste the Node class below to use as a representation of a node in 
the linked list. Return the value of the node at that position.

class Node(object):
def init(self, data):
self.data = data
self.next = None

Input: Singly linked list and integer m (m items from the end of the list)
Output: (Integer/Float/Char) Data element m

Test cases:
input is None
input is a valid singly linked list
input m is not within list

Brainstorming:
Create linked list class
Add elements to linked list
Search elements in linked list forward
Calculate length of list
Use this calculation to find the reverse position from the end of the list

Reference:

https://www.codefellows.org/blog/implementing-a-singly-linked-list-in-python/

'''

# Indivisual node class for linked list
class Node(object):
    def __init__(self, data):
        self.data = data
        self.next = None

# Linked list class
class LinkedList(object):
    def __init__(self, head=None):
        self.head = head
    
    def get_size(self):
        # Initialize head as starting node
        # and length = 0
        current = self.head
        length = 0
        # Traverse through next nodes until the end and add 1 to the length each time
        while current:
            length += 1
            current = current.next
        return length
   
    def append(self, new_node):
        current = self.head
        # If head node is present...
        if self.head:
            while current.next:
                # Cycle through the next nodes
                current = current.next
            # And append new node to end of list
            current.next = new_node
        else:
            # Add node as the head
            self.head = new_node
    
    def get_position(self, position):
        counter = 1
        current = self.head
        # Handle invalid position
        if position < 1:
            return None
        while current and counter <= position:
            # Returns node if position matches with counter
            if counter == position:
                return current
            # Cycle through next nodes
            current = current.next
            counter += 1
        # If position of node is not found return None
        return None
    
    def get_position_reverse(self, position, size):
        counter = 1
        current = self.head
        # Calculates the position integer from total length of list
        # Then, search through list forward until position is found
        position = size - position
        if position < 1:
            return None
        while current and counter <= position:
            if counter == position:
                return current
            current = current.next
            counter += 1
        return None

# Set up nodes
n1 = Node(1)
n2 = Node(2)
n3 = Node(3)
n4 = Node(4)
n5 = Node(5)

# Set up Linked List
# 1 -> 2 -> 3 -> 4 -> 5
linkl = LinkedList(n1)
linkl.append(n2)
linkl.append(n3)
linkl.append(n4)
linkl.append(n5)

# Test out linked list
# Should be 3
print "Test Position:", linkl.get_position(3).data
# Should be 1
print "Test Position:", linkl.head.data

# Initialize empty list for Test Case 4
link2 = LinkedList()

def question5(ll, m):
    # Handle None input
    if ll == None or m == None:
        return "None input entered."
    # Handle empty list
    if ll.get_size() == 0:
        return "Linked list is empty."
    # Get length of linked list
    size = ll.get_size()
    # Handle None output
    if ll.get_position_reverse(m, size) == None:
        return "No position found."
    else:
        # Get node position from end of list
        node_val = ll.get_position_reverse(m, size).data
        return node_val

print "call question5(ll, m) with test cases:"

print " **************** Question 5 ****************"
# input is None
# Should be "None input entered."
print "Test Case 1 -", question5(None, None)

# input is a valid singly linked list
# Should be 4
# linkl: 1 -> 2 -> 3 -> 4 -> 5
# 1 node from the end = node 4
print "Test Case 2 -", question5(linkl, 1)

# input m is not within list
# Should be "No position found."
print "Test Case 3 -", question5(linkl, 8)

# input ll is an empty list
# Should be "Linked list is empty."
print "Test Case 4 -", question5(link2, 1)
