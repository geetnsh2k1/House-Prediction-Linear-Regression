import numpy as np
import matplotlib.pyplot as plt

class Regression:
    
    def __init__(self):
        self.size, self.bedrooms, self.price = Regression.read_file()
        self.m = len(self.size)
        self.q0 = 0
        self.q1 = 0
        self.q2 = 0
    
    #h(q) = q0 + q1.x
    #(1/2m)*(sum((h(q)-y)^2))
    def cost_function(self, q0=0, q1=0, q2=0):
        result = 0
        for i in range(self.m):
            h = q0+(q1*self.size[i])+(q2*self.bedrooms[i])
            y = self.price[i]
            result += (h-y)**2
        result = (1/(2*self.m))*result
        return result
    
    #qj := qj - learning_rate*(1/m)*(sum(h(q)-y)).xj
    def gradien_descent(self, learning_rate, no_iterations):
        cf = []
        for i in range(no_iterations):
            s0 = 0
            s1 = 0
            s2 = 0
            for i in range(self.m):
                temp_size = (self.size[i] - np.mean(self.size))/(np.max(self.size)-np.min(self.size))
                temp_price = (self.price[i] - np.mean(self.price))/(np.max(self.price)-np.min(self.price))
                temp_bed = (self.bedrooms[i] - np.mean(self.bedrooms))/(np.max(self.bedrooms)-np.min(self.bedrooms))
                s0 += ((self.q0+(self.q1*temp_size)+(self.q2*temp_bed)) - temp_price)*1
                s1 += ((self.q0+(self.q1*temp_size)+(self.q2*temp_bed)) - temp_price)*temp_size
                s2 += ((self.q0+(self.q1*temp_size)+(self.q2*temp_bed)) - temp_price)*temp_bed
            t0 = self.q0 - learning_rate*((1/self.m)*s0)
            t1 = self.q1 - learning_rate*((1/self.m)*s1)
            t2 = self.q2 - learning_rate*((1/self.m)*s2)
            cf.append(self.cost_function(t0, t1, t2))
            self.q0 = t0
            self.q1 = t1
            self.q2 = t2
        return cf
    
    @staticmethod
    def plot_cost_function_to_iterations(cf, total_iterations=200):
        ni = np.arange(total_iterations)
        plt.plot(ni, cf)
        plt.ylabel("Cost Function")
        plt.xlabel("Number Of Iterations")
        plt.show()
    
    @staticmethod
    def read_file():
        house_size = []
        house_bedrooms = []
        house_price = []
        with open("./data2.csv", 'r') as file:
            while True:
                data = file.readline().split(',')
                if len(data) == 3:
                    house_size.append(eval(data[0]))
                    house_bedrooms.append(eval(data[1]))
                    house_price.append(eval(data[2]))
                else:
                    if len(data) == 1:
                        break
                    else:
                        pass
        return np.array(house_size), np.array(house_bedrooms), np.array(house_price)

r=Regression()
cf = r.gradien_descent(0.05, 200)
hs = eval(input("Enter house size : "))
bedrooms = eval(input("Enter number of bedrooms : "))
print(round(r.q0 + r.q1*hs + r.q2*bedrooms, 2), "Lakhs")