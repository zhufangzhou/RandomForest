RandomForest
============

C++ RandomForest

##data format

###text type

one sample per line
label and features are divided by space
all label and features are treated as double float (64bit) type

**example**

**1** 1.2 3.5 2.1 2.1

**0** 0.2 20 3.4 2.1

###binary type

####all-in-one file

all label and features are treated as double float (64bit) type

you need to specify the num of feature in the function.

####separate files

label and feature are stored in two file ,64bit per value.

you need to specify the num of feature in the function too.
