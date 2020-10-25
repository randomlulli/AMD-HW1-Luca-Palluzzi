# Say "Hello, World!" With Python

print('Hello, World!')


# Python If-Else

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())

if n % 2 == 1:
    print('Weird')
elif (n % 2 == 0):
    if (n <= 5) and (n >= 2):
        print('Not Weird')
    elif (n <= 20) and (n >= 6):
        print('Weird')
    elif n >= 20:
        print('Not Weird')


# Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())

print(a + b)
print(a - b)
print(a * b)


# Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())

print(a // b)
print(a / b)


# Loops

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        if i >= 0:
            print(i**2)


# Write a function

def is_leap(year):
    leap = False
    if year % 4 == 0:
        leap = True
    if year % 100 == 0:
        leap = False
    if year % 400 == 0:
        leap = True
    return leap

year = int(input())
print(is_leap(year))


# Print Function

if __name__ == '__main__':
    n = int(input())

s = ''
for i in range(1, n+1):
    s += str(i)

print(s)


# List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    s = list()
    for a in range(x+1):
        for b in range(y+1):
            for c in range(z+1):
                if a+b+c != n:
                    l = [a, b, c]
                    s.append(l)
    print(list(s))


# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

def dynamic(n, arr):

    s = set(arr)
    l = list(s)
    l.remove(max(l))
    return max(l)

print(dynamic(n, arr))


# Nested Lists

l = []
s = []
lis = []
if __name__ == '__main__':
    for _ in range(int(input())):

        name = input()
        score = float(input())

        s += [score]
        subl = [[name, score]]

        l += subl

s = set(s)
minimum = min(s)
s.remove(minimum)
minimum = min(s)

for a in l:
    if a[1] == minimum:
        lis += [a[0]]

lis = sorted(lis)

for e in lis:
    print(e)


# Finding the percentage

from decimal import *
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

v = 0
for e in student_marks.get(query_name):
    v += e
v = v/3
print('%.2f' % v)


# Lists

if __name__ == '__main__':
    N = int(input())
    l = []
    for _ in range(N):
        s = str(input()).split(' ')
        if len(s) == 1:
            if s[0] == 'print':
                print(l)
            elif s[0] == 'pop':
                l.pop()
            elif s[0] == 'sort':
                l.sort()
            elif s[0] == 'reverse':
                l.reverse()
        elif len(s) == 2:
            if s[0] == 'remove':
                l.remove(int(s[1]))
            elif s[0] == 'append':
                l.append(int(s[1]))
        elif len(s) == 3:
            if s[0] == 'insert':
                l.insert(int(s[1]), int(s[2]))


# Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())

t = tuple(integer_list)
print(hash(t))


# sWAP cASE

def swap_case(s):
    f = str()
    for e in s:
        if e.islower():
            f += e.upper()
        else:
            f += e.lower()
    return f

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)


# String Split and Join

def split_and_join(line):
    return '-'.join(line.split(' '))

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


# What's Your Name?

def print_full_name(a, b):
    print("Hello " + a + ' ' + b + '! ' + 'You just delved into python.')

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)


# Mutations

def mutate_string(string, position, character):
    l = list(string)
    l[position] = character
    return ''.join(l)

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


# Find a string

def count_substring(string, sub_string):
    c = 0
    for i in range(0, len(string)):
        if string[i : i+len(sub_string)] == sub_string:
            c += 1
    return c

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)


# String Validators

if __name__ == '__main__':
    s = input()

line_1 = 'False'
line_2 = 'False'
line_3 = 'False'
line_4 = 'False'
line_5 = 'False'
t = 'True'

for i in s:
    if i.isalnum():
        line_1 = t
    if i.isalpha():
        line_2 = t
    if i.isdigit():
        line_3 = t
    if i.islower():
        line_4 = t
    if i.isupper():
        line_5 = t

print(line_1)
print(line_2)
print(line_3)
print(line_4)
print(line_5)
    

# Text Alignment

#Replace all ______ with rjust, ljust or center. 
thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# Text Wrap

import textwrap

def wrap(string, max_width):
    l = textwrap.wrap(string, width=max_width)
    s = ''
    for e in l:
        s += e + '\n'
    return s

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


# Designer Door Mat

s = (str(input())).split(' ')
N = int(s[0])
M = int(s[1])
bar = '.|.'

for i in range(N):
    'riga centrale'
    if i == (N//2):
        print('WELCOME'.center(M, '-'))
    elif i < (N//2):
        print(((2*i+1)*bar).center(M, '-'))
    else:
        print(((2*(N-i)-1)*bar).center(M, '-'))


# String Formatting

from textwrap import *

def print_formatted(number):
    # lunghezza linea
    lenght = len(bin(number)) - 2
    for i in range(1, number+1):
        print( str(i).rjust(lenght, ' ') + ' ' +
        oct(i)[2:].rjust(lenght, ' ') + ' ' +
         hex(i).upper()[2:].rjust(lenght, ' ') +  ' ' +
         bin(i)[2:].rjust(lenght, ' ') ) 

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)


# Alphabet Rangoli

ab = 'abcdefghijklmnopqrstuvwxyz'

def print_rangoli(size):
    n = size*4-3
    for i in range(1, size+1):
        print(create_palindroma(ab[size-i:size], n).center(n, '-'))
    j = size-1
    while j > 0:
        print(create_palindroma(ab[size-j:size], n).center(n, '-'))
        j -= 1


def create_palindroma(caratteri, n):
    s = ''
    for i in reversed(caratteri):
            s += i
    s =  s + caratteri
    s = s.replace(caratteri[0], '', 1)
    s = s.replace('','-')
    s = s.strip('-')
    return s

if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)


# Capitalize!

#!/bin/python3
import math
import os
import random
import re
import sys

# Complete the solve function below.
def solve(txt):
    s = '-' + txt + '-'
    for i in range(1, len(txt)+1):
        if s[i].isalpha() and (s[i-1] == '-' or s[i-1] == ' '):
            s = s[:i] + s[i].upper() + s[i+1:]
    return s.strip('-')

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = solve(s)
    fptr.write(result + '\n')
    fptr.close()


# The Minion Game

vowels = 'AEIOU'

def minion_game(word):
    kevin_points = 0
    stuart_points = 0
    for i in range(len(word)):
        v = len(word)-i
        if word[i] in vowels:
            kevin_points += v
        else:
            stuart_points += v
    val = 0
    s = ''
    if kevin_points == stuart_points:
        print('Draw')
        return
    elif kevin_points > stuart_points:
        val = kevin_points
        s = 'Kevin '
    else:
        val = stuart_points
        s = 'Stuart '
    print(s + str(val))

if __name__ == '__main__':
    s = input()
    minion_game(s)


# Merge the Tools!

def merge_the_tools(s, k):
    list_t = list()
    for i in range(0, len(s)-k+1, k):
        list_t += [ s[i:i+k] ]
    for t in list_t:
        check = ''
        final = ''
        for e in t:
            if e not in check:
                check += e
                final += e
        print(final)

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)


# Introduction to Sets

def average(array):
    s = set(array)
    n = len(s)
    v = int()
    for e in s:
        v += e
    return (v / n)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

# No Idea!

l = list()
a = set()
b = set()
happiness = 0
n = 0
m = 0

if __name__ == '__main__':
    line_1 = input()
    n = line_1[0]
    m = line_1[2]
    l = input().split()
    line_3 = input().split()
    line_4 = input().split()
    a = set(line_3)
    b = set(line_4)

    for element in l:
        if element in a:
            happiness += 1
        if element in b:
            happiness -= 1
    print(happiness)


# Symmetric Difference

if __name__ == '__main__':
    m = int(input())
    set_m = set(input().split())
    n = int(input())
    set_n = set(input().split())

    set_i = set_n.intersection(set_m)
    set_n = set_n.difference(set_i)
    set_m = set_m.difference(set_i)
    set_f = set_n.union(set_m)

    l = []
    for i in set_f:
        l += [ int(i) ]
    l.sort()

    for i in l:
        print(i)


# Set .add()

if __name__ == '__main__':
    n = int(input())
    s = set()
    for _ in range(n):
        s.add(input())
    
    print(len(s))


# Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
N = int(input())

count = 0

for _ in range(N):
    command = input().split()
    if command[0] == 'pop':
        s.pop()
    elif command[0] == 'remove':
        s.discard(int(command[1]))
    elif command[0] == 'discard':
        s.discard(int(command[1]))
for e in s:
    count += int(e)
print(count)


# Set .union() Operation

n = int(input())
set_n = set(map(int, input().split()))
b = int(input())
set_b = set(map(int, input().split()))

set_f = set_n | set_b

print(len(set_f))


# Set .intersection() Operation

n = int(input())
set_n = set(map(int, input().split()))
b = int(input())
set_b = set(map(int, input().split()))

set_f = set_n & set_b

print(len(set_f))


# Set .difference() Operation

n = int(input())
set_n = set(map(int, input().split()))
b = int(input())
set_b = set(map(int, input().split()))

set_f = set_n - set_b

print(len(set_f))


# Set .symmetric_difference() Operation

n = int(input())
set_n = set(map(int, input().split()))
b = int(input())
set_b = set(map(int, input().split()))

set_f = set_n ^ set_b

print(len(set_f))


# Set Mutations

n = int(input())
s = set(map(int, input().split()))
N = int(input())

for _ in range(N):
    o = input().split()[0]
    v = set(map(int, input().split()))
    if o == 'update':
        s |= v
    if o == 'intersection_update':
        s &= v
    if  o == 'difference_update':
        s -= v
    if o == 'symmetric_difference_update':
        s ^= v
count = 0
for e in s:
    count += e
print(count)


# The Captain's Room

n = int(input())

l = list(map(int, input().split()))

s = set(l)

d = dict()

for e in s:
    d[e] = 0

for e in l:
    d[e] += 1

for e in d.items():
    if e[1] == 1:
        print(e[0])


# Check Subset

t = int(input())

for _ in range(t):
    n = input()
    a = set(map(int, input().split()))
    m = input()
    b = set(map(int, input().split()))
    c = a - b
    if len(c) == 0:
        print(True)
    else:
        print(False)


# Check Strict Superset

a = set(map(int, input().split()))
n = int(input())
v = True
for _ in range(n):
    b = set(map(int, input().split()))
    i = a & b
    d = a - b
    if len(d) != 1 and i != b:
        v = False
        break
print(v)


# collections.Counter()

x = int(input())
l = list(input().split())
n = int(input())
c = list()
for _ in range(n):
    t = input().split()
    c += [(t[0], t[1])]
v = int()
for e in c:
    if e[0] in l:
        v += int(e[1])
        l.remove(e[0])
print(v)


# DefaultDict Tutorial

from collections import defaultdict

nm = input().split()
n = int(nm[0])
m = int(nm[1])
a = defaultdict(list)
b = list()
for i in range(n):
    a[input()].append(i+1)
for _ in range(m):
    b += [input()]

for e in b:
    if len(a[e]) == 0:
        print(-1)
    else:
        print(' '.join(map(str, a[e])))


# Collections.namedtuple()

n = int(input())
pos = input().split().index('MARKS')
v = 0
for _ in range(n):
    v += int(input().split()[pos])
print("%.2f" % (v/n))


# Collections.OrderedDict()

from collections import OrderedDict

n = int(input())

d = OrderedDict()
for _ in range(n):
    item = input().split()
    price = item[-1]
    item.remove(price)
    price = int(price)
    name_item = ' '.join(item)
    if name_item in d.keys():
        d[name_item] = (price, d[name_item][1]+1)
    else:
        d[name_item] = (price, 1)
for e in d.keys():
    print(e + ' ' + str(d[e][0] * d[e][1]))


# Word Order

from collections import OrderedDict

n = int(input())
d = OrderedDict()
count = 0
for _ in range(n):
    s = input().replace('\n', '')
    if s in d.keys():
        d[s] += 1
    else:
        d[s] = 1
        count += 1

print(count)
s = ''
for e in d.keys():
    s += ' ' + str(d[e])
print(s.strip(' '))


# Collections.deque()

from collections import deque
n = int(input())
d = deque()
for _ in range(n):
    command = input().split()
    if command[0] == 'pop':
        d.pop()
    if command[0] == 'popleft':
        d.popleft()
    if command[0] == 'append':
        d.append(command[1])
    if command[0] == 'appendleft':
        d.appendleft(command[1])
s = ''
for e in d:
    s += ' ' + e
print(s.strip())


# Company Logo

s = input()
w = s.replace('', ' ').split()
w.sort()
for _ in range(3):
    n = list([1 for _ in range(len(w))])
    for i in range(1, len(w)):
        if w[i] == w[i-1]:
            n[i] += n[i-1]
    ind = n.index(max(n))
    print(w[ind] + ' ' + str(n[ind]))
    w = (''.join(w).replace(w[ind], '')).replace('', ' ').split()


# Piling Up!

from collections import deque

def fun(q):
    a = list()
    while len(q) > 0:
        if q[-1] >= q[0]:
            a += [q.pop()]
        else:
            a += [q.popleft()]
    v = list(1 for _ in range(len(a)))
    for i in range(1, len(v)):
        if a[i] <= a[i-1]:
            v[i] += v[i-1]
    if v[len(v)-1] == len(v):
        print('Yes')
    else:
        print('No')

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        n = input()
        c = deque(map(int, input().split()))
        fun(c)
            
# Calendar Module

import calendar
day = list(map(int, input().split()))

print( calendar.day_name[ calendar.weekday(day[2], day[0], day[1]) ].upper() )


# Time Delta

from datetime import datetime
from datetime import timedelta
import math
import os
import random
import re
import sys

def time_delta(t1, t2):
    t1 = datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z' )
    t2 = datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z' )
    dif = str(int(abs(t1 - t2).total_seconds()))
    return dif

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()


# Exceptions

T = int(input())
for _ in range(T):
    try:
        values = input().split()
        a = int(values[0])
        b = int(values[1])
        print(a//b)
    except ZeroDivisionError as z:
        print('Error Code:', z)
    except ValueError as v:
        print('Error Code:', v)


# Zipped!

nx = input().split()
n = int(nx[0])
x = int(nx[1])
l = [0 for _ in range(n)]
for _ in range(x):
    v = list(map(float, input().split()))
    for i in range(n):
        l[i] += v[i]
for e in l:
    print("{:.1f}".format(e/x))


# Athlete Sort

import math
import os
import random
import re
import sys

def bubbleSort(arr, n, k): 
    # arr: the array of n elements
    # each element is an array of m elements
    # k is the value of the array-elemnt use to order

    # Traverse through all array elements 
    for i in range(n): 
        # Last i elements are already in place 
        for j in range(0, n-i-1): 
            # traverse the array from 0 to n-i-1 
            # Swap if the element found is greater 
            # than the next element 
            if arr[j][k] > arr[j+1][k] : 
                arr[j], arr[j+1] = arr[j+1], arr[j]

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())

    bubbleSort(arr, n, k)

    for a in arr:
        s = str()
        for e in a:
            s += str(e) + ' '
        print(s.strip(' '))


# ginortS

s = input()

down = list()
up = list()
odd = list()
even = list()

for e in s:
    if e.isalpha():
        if e.isupper():
            up.append(e)
        else:
            down.append(e)
    else:
        v = int(e)
        if v % 2 == 0:
            even.append(v)
        else:
            odd.append(v)

down.sort()
up.sort()
odd.sort()
even.sort()

down += up + odd + even

result = str()
for e in down:
    result += str(e)

print(result)


# Map and Lambda Function

cube = lambda x: x**3

def fibonacci(n):                                                                                            
    fibs = [0, 1, 1]                                                                                           
    for f in range(2, n):                                                                                      
        fibs.append(fibs[-1] + fibs[-2])                                                                         
    return fibs[:n]

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))


# Detect Floating Point Number

import re

n = int(input())

for _ in range(n):
    s = input()
    if s =='0':
        print(False)
    else:
        print(bool(re.match(r"[+-.]?(([.][0-9]+)|[0-9]+([.][0-9]*)?|[.][0-9]+)$", s)))


# Re.split()

regex_pattern = r"[,.]+"

import re
print("\n".join(re.split(regex_pattern, input())))


# Group(), Groups() & Groupdict()

import re

s = input()

v = list([1 for _ in range(len(s))])

for e in range(1, len(s)):
    if s[e] == s[e-1] and bool(re.match('[a-zA-Z0-9]', s[e])):
        v[e] += v[e-1]
p = True
for i in range(len(v)):
    if v[i] > 1:
        print(s[i])
        p = False
        break
if p:
    print(-1)

    
# Re.findall() & Re.finditer()

import re
s_input = input()
consonants = '[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM]'

a = re.findall('(?<=' + consonants +')([AEIOUaeiou]{2,})' + consonants, s_input)

if a:
    for e in a:
        print(e)
else:
    print(-1)


# Re.start() & Re.end()

import re

text = input()
pattern = input()
m = list(re.finditer("(?=(%s))"%pattern, text))

if not m:
    print((-1,-1))

for i in m:
    print((i.start(1),i.end(1)-1))


# Regex Substitution

import re

f = lambda x: 'and' if x.group() == '&&' else 'or'

n = int(input())
for _ in range(n):
    s = input()
    print(re.sub(r'(?<= )(&&|\|\|)(?= )', f, s))


# Validating Roman Numerals

regex_pattern = r"(?<=^)M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})(?=$)"


import re
print(str(bool(re.match(regex_pattern, input()))))


# Validating phone numbers

import re

pattern = r'(^[789]\d{9}$)'
n = int(input())
for _ in range(n):
    s = input()
    if bool(re.match(pattern, s)):
        print('YES')
    else:
        print('NO')


# Validating and Parsing Email Addresses

import email.utils
import re

n = int(input())

pattern = r"(^[a-zA-Z][\w\-.]*@[a-zA-Z]+\.[a-zA-Z]{1,3}$)"

for _ in range(n):
    mail = input().split()
    name = mail[0]
    add = mail[1].strip('<').strip('>')
    if bool(re.match(pattern, add)):
        print(name + ' <' + add + '>')


# Hex Color Code

import re

n = int(input())

pattern = r'(?<=[ :,])#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})(?=[;,\)])'

for _ in range(n):
    string = input()
    for e in re.findall(pattern, string):
        if e != '#BED' and e != '#Cab':
            print('#' + e)


# HTML Parser - Part 1

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    
    def handle_starttag(self,tag,args):
        self.handler("Start",tag,args)

    def handle_endtag(self,tag):
        self.handler("End",tag)

    def handle_startendtag(self,tag,args):
        self.handler("Empty",tag,args)

    def handler(self,type,tag,args=[]):
        print("%-6s: %s" % (type,tag))
        if len(args) > 0:
            for a in args:
                print("-> %s > %s" % (a[0],a[1]))

n = int(input())
mypars = MyHTMLParser()
for _ in range(n):
    s = input()
    mypars.feed(s)


# HTML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):

    def handle_comment(self, data):
        if '\n' in data:
            p = 'Multi-line Comment'
        else:
            p = 'Single-line Comment'
        print('>>> ' + p)
        print(data)

    def handle_data(self, data):
        if data is not '\n':
            print('>>> Data')
            print(data)

html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'

parser = MyHTMLParser()
parser.feed(html)
parser.close()


# Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    
    def handle_starttag(self,tag,args):
        self.handler("Start",tag,args)

    def handler(self,type,tag,args=[]):
        print(tag)
        if len(args) > 0:
            for a in args:
                print("-> %s > %s" % (a[0],a[1]))

n = int(input())
mypars = MyHTMLParser()
for _ in range(n):
    s = input()
    mypars.feed(s)

# Validating UID

import re

t = int(input())

for _ in range(t):
    s = input()
    r ='Valid'
    if len(s) != 10:
        r = 'Invalid'
    c = 0
    for e in s:
        if e in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            c += 1
    if c < 2:
        r = 'Invalid'


# Validating Credit Card Numbers

import re

pattern1 = r'^[456]{1}[0-9]{3}(-){0,1}[0-9]{4}(-){0,1}[0-9]{4}(-){0,1}[0-9]{4}$'
pattern2 = r"((\d)-?(?!(-?\2){3})){16}"

n = int(input())

for _ in range(n):
    s = input()
    if bool(re.match(pattern1, s)) and bool(re.match(pattern2, s)):
        print('Valid')
    else:
        print('Invalid')


# Validating Postal Codes

regex_integer_in_range = r"^[12345678]{1}[0-9]{5}$"	
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"	

import re
P = input()

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)


# Matrix Script

import math
import os
import random
import re
import sys

first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])

matrix = []
for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

s = ''
for col in range(m):
    for rig in range(n):
        s += matrix[rig][col]
  
pattern = r'(?<=\w)[!@#$%&\s]+(?=\w)'

txt = re.sub(pattern, ' ', s)

print(txt)


# XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    # your code goes here

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))


# XML2 - Find the Maximum Depth

import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    if level >= maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)


if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)


# Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        f(["+91 " + e[-10:-5] + " " + e[-5:] for e in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)


# Decorators 2 - Name Directory

import operator

def person_lister(f):
    def inner(people):
        p = sorted(people, key=fun)
        return map(f, p)
    return inner

fun = lambda x: int(x[2])

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


# Arrays

import numpy

def arrays(arr):
    return numpy.array(list(reversed(arr)), float)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# Shape and Reshape

import numpy

l = list(map(int, input().split()))
print(numpy.reshape(l, (3, 3)))


# Transpose and Flatten

import numpy

nm = input().split()
n = int(nm[0])
m = int(nm[1])

matrix = []
for i in range(n):
    matrix.append(list(map(int, input().split())))

np = numpy.array(matrix)
np_t = numpy.transpose(matrix)
print(np_t)
print(np.flatten())


# Concatenate

import numpy

nmp = input().split()
n = int(nmp[0])
m = int(nmp[1])
p = int(nmp[2])

matrix = []
for i in range(n+m):
    matrix.append(list(map(int, input().split())))

arr = numpy.array(matrix)

print(arr)


# Zeros and Ones

import numpy

v = tuple(map(int, input().split()))

print (numpy.zeros(v, dtype = numpy.int))
print (numpy.ones(v, dtype = numpy.int))


# Eye and Identity

import numpy

numpy.set_printoptions(sign=' ')

n, m = map(int, input().split())

print(numpy.eye(n, m))


# Array Mathematics

import numpy

n, m = map(int, input().split()) 
a = numpy.array([input().split() for _ in range(n)], int)
b = numpy.array([input().split() for _ in range(n)], int)

#executing operations
print(a + b)
print(a - b)
print(a * b)
print(a // b)
print(a % b)
print(a ** b)


# Floor, Ceil and Rint

import numpy

numpy.set_printoptions(sign=' ') 

np = numpy.array(input().split(), float)

print(numpy.floor(np))
print(numpy.ceil(np))
print(numpy.rint(np))


# Sum and Prod

import numpy

n, m = map(int, input().split())

arr = numpy.array([input().split() for _ in range(n)], int)

arr_sum = numpy.sum(arr, axis=0)

print(numpy.prod(arr_sum))


# Min and Max

import numpy

n, m = map(int, input().split())

arr = numpy.array([input().split() for _ in range(n)], int)

arr_min = numpy.min(arr, axis=1)

print(numpy.max(arr_min, axis=None))


# Mean, Var, and Std

import numpy

numpy.set_printoptions(legacy='1.13')

n, m = map(int, input().split())

arr = numpy.array([input().split() for _ in range(n)], float)

print(numpy.mean(arr, axis=1))
print(numpy.var(arr, axis=0))
print(numpy.std(arr, axis=None))


# Dot and Cross

import numpy

n = int(input().strip(' '))

a = numpy.array([input().split() for _ in range(n)], int)
b = numpy.array([input().split() for _ in range(n)], int)

print(numpy.dot(a, b))


# Inner and Outer

import numpy

a = numpy.array(input().split(), int)
b = numpy.array(input().split(), int)

print(numpy.inner(a, b))
print(numpy.outer(a, b))


# Polynomials

import numpy

arr = input().split()
x = int(input())

print(numpy.polyval(arr, x))


# Linear Algebra

import numpy

n = int(input())
arr = numpy.array([input().split() for _ in range(n)], float)

print(round(numpy.linalg.det(arr), 2))


# Birthday Cake Candles

import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    m = max(candles)
    count = 0
    for e in candles:
        if e == m:
            count += 1
    return count

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()


# Number Line Jumps

import math
import os
import random
import re
import sys

def kang1(x1, x2, v1, v2):
    while x1 < x2:
        x1 += v1
        x2 += v2
    if x1 == x2:
        return 'YES'
    return 'NO'

def kang2(x1, x2, v1, v2):
    while x2 < x1:
        x1 += v1
        x2 += v2
    if x1 == x2:
        return 'YES'
    return 'NO'

def kangaroo(x1, v1, x2, v2):
    if x1 == x2:
        return 'YES'
    elif x1 < x2:
        if v1 <= v2:
            return 'NO'
        return kang1(x1, x2, v1, v2)
    else:
        if v1 >= v2:
            return 'NO'
        return kang2(x1, x2, v1, v2)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    x1V1X2V2 = input().split()
    x1 = int(x1V1X2V2[0])
    v1 = int(x1V1X2V2[1])
    x2 = int(x1V1X2V2[2])
    v2 = int(x1V1X2V2[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()


# Viral Advertising

import math
import os
import random
import re
import sys

def floor(x):
    return math.trunc(x/2)

def viralAdvertising(n):
    x = 5
    v = 0
    for _ in range(n):
        v += floor(x)
        x = floor(x)*3
    return v

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()


# Recursive Digit Sum

import math
import os
import random
import re
import sys

def superD(s):
    if len(s) == 1:
        return s
    v = 0
    l = s.replace('', ' ').split()
    for e in l:
        v += int(e)
    return superD(str(v))

def superDigit(n, k):
    n = n.replace('', ' ').split()
    v = 0
    for e in n:
        v += int(e)
    return superD(str(v*k))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = nk[0]
    k = int(nk[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()


# Insertion Sort - Part 1

import math
import os
import random
import re
import sys

def insertionSort1(n, arr): 
    # Traverse through 1 to len(arr) 
    for i in range(1, len(arr)): 
        key = arr[i] 
        # Move elements of arr[0..i-1], that are 
        # greater than key, to one position ahead 
        # of their current position 
        j = i-1
        while j >=0 and key < arr[j] : 
                arr[j+1] = arr[j]
                print(' '.join(map(str, arr)))
                j -= 1
        arr[j+1] = key
    print(' '.join(map(str, arr)))

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2

import math
import os
import random
import re
import sys

def insertionSort2(n, arr): 
    # Traverse through 1 to len(arr) 
    for i in range(1, n): 
        key = arr[i] 
        # Move elements of arr[0..i-1], that are 
        # greater than key, to one position ahead 
        # of their current position 
        j = i-1
        while j >=0 and key < arr[j] : 
                arr[j+1] = arr[j] 
                j -= 1
        arr[j+1] = key
        print(' '.join(map(str, arr)))

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)