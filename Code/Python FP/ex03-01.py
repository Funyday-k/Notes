
a = []
condition = True
result = 0

while condition:
    
    num = int(input("Please enter a number:"))


    if num > 1:
        for i in range(2,num):
            if (num % i) == 0 :
                condition =  False
                break
        else:
            a.append(num)
    else:
        condition = False
        break

        
if a != []:
    for i in a:
        result += i


print('Result='+str(result))