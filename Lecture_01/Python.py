## String and Variables
my_var = 1
print type(my_var)

my_var = 'String'
print type(my_var)

str1 = 'a'
str2 = '+b'
print str1+str2

x = 'a'
y = x*8
print y


init_int = 3
print init_int*3
print init_int**3

s = 'This IS StrING'
s.lower()
s.upper()

ix = s.split('S')
print ix
print type(ix)


##List
ix.append(16)
print ix
ix.append([1,2,3])
print ix
ix.insert(0,{'1':2,3:4,1:'11'})
print ix

print "-----------------"
## Slicing of lists
print len(ix)
print ix[0]
print ix[len(ix)-1]
print ix[-1]
print "----------------------"
###ix[<start>:<end+1>]


print "-----------------"
ix = range(1,10)
print ix

print ix[::2]
print ix[1::2]

## Tuples
ix = (1,2,3)
print ix[0]

my_list = [(1,2), ('3',4)]
my_list.append(1)
print my_list
my_list[-1] = 4
my_list[0] = (1,2,3)
print my_list

my_tuple = (1,[2,3])
print my_tuple
my_tuple[1].append(1)
print my_tuple


## Dictionaries
x = dict()
x.setdefault('my_key','my_value')
x[2] = '5'
print x
print x.keys()
print x.values()

## Set
x = set([1,2,3,4,1,4])
print x
for ele in x:
    print ele


def sum(v1,v2,v3, *args,**kwargs):
    print v1+v2+v3
    print kwargs
    print args
    
# def sumgeneric(*args):
#     print 
#     for ix in args:
    
sum(1,2,3)
print "------------------"
sum(1,2,3,4,5)
print "-------------"
sum(1,2,3,4,5,a=1,b=2,6)


class MyClass:
    my_list = [] ## Shared among all the objects
    def __init__(self,x=1,y=2):
        print "Welcome to the constructor"
        self.x = x
        self.y = y
        self.my_private_list = []
    def add(self):
        return self.x+self.y
    def multiply(self):
        return self.x*self.y
    def square(self):
        return self.x**2
    def addit(self,value):
        self.my_list.append(value)
    def add_private(self, value):
        self.my_private_list.append(value)

 obj1 = MyClass(4)
print obj1.x
print obj1.y
obj1.addit('obj1 value')
obj1.add_private('v1')
print obj1.my_private_list
print "~~~~~~~~~~~~~~~~~~~~~~~~"
obj2 = MyClass(3,9)
print obj2.x
print obj2.y
obj2.addit('obj2 value')
obj2.add_private('v2')

print obj1.my_private_list


import numpy as np
print np.unique([1,2,3,4,1])


temp_list = [1,2,3,4]
print type(temp_list)
np_arr = np.asarray(temp_list)
temp_list = np.array(temp_list)
print np_arr.shape
print type(temp_list)


test_mat = np.zeros((2,2))
print test_mat.dtype
print type(test_mat)       