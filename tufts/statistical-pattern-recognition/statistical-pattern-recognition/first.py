import matplotlib.pyplot as plt


fig1 = plt.figure()
subp = fig1.add_subplot(111)
subp.plot([1, 3, 5, 6, 9.9, 12, 14], [1,2,3,4,5,6.7,7.7])

fig2 = plt.figure()
subp2 = fig2.add_subplot(111)
x = range(1,7,2)
y = [i**2 for i in x]

subp2.plot(x, y)

plt.show()
