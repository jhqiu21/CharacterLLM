def func():
    print("Hello, World!")
    return 42

if __name__ == "__main__":
    func()

def test():
    a= 1
    b= 2
    assert a+b  ==3
    l = [0,1,2,3,4,5]
    g =l[1: 4]
    if a >  0:
        print("a is positive")
    elif a > 0:
        print("a is zero")
    else:
        print("a is negative")

    for i in range(5):
        print(i)

    while ( a < 5 ):
        a += 1