"""*******************************************************
This code summarizes the lecture on COMMENTS and STRINGS
******************************************************"""
print(__doc__)
# comments
# this is a comment
""" This is comment too """
''' This is comment too '''

""" I like David's dog"""
''' Her name is Michaella '''
# strings
print('Hello wold')
print("Hello wold")

print('Yossi\"s dog is beautiful')

my_message = 'hello world'
print(my_message)

my_message2 = "my name is \'Husne\'"
print(my_message2)

my_messages5 = """ I am writing a string on multible lines
"""
#help(print)

my_message = 'hello world'
print(len(my_message))
print(my_message[0])
print(my_message[-1])
print(my_message[4])
print(my_message[6])
print(len(my_message))
my_message = 'hello world'
print(len(my_message))




print(my_message[0:5])

print(my_message[:5])
print(my_message[2:7])

print('the length of my message is', len(my_message))

print(my_message.lower())
print(my_message.upper())
print(my_message.count('hello'))
print(my_message.count('l'))
print(my_message.find('world'))
print(my_message.find('universe'))
print(my_message.replace('world', 'universe'))
print(my_message)

print(dir(my_message))
help(str.upper)
help(my_message.upper)

my_message2 = "hello universe"

print(my_message+my_message2)
print('Hello '+', '+' Michaelle')

print('Hello', '+', 'Michaelle')

message = 'Hello {0}, welcome to the {1}'

print(message.format('Mathieu', 'world'))
print(message.format('David', 'Jerusalem'))
print(message.format('Husne', 'Tel Aviv'))

name = 'Adelina'
place = 'Tel-Aviv'

message = f'Hello {name}, welcome to {place}'
print(message)

message = f'Hello {name.upper()}, welcome to {place}'
print(message)

print((my_message + ',') * 10)
