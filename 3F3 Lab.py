import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def ksdensity(data, width=0.3):
    """Returns kernel smoothing function from data points in data"""
    def ksd(x_axis):
        # x_axis is a vector of x values to evaluate the density at
        def n_pdf(x, mu=0., sigma=1.): # normal pdf
            u = (x - mu) / abs(sigma)
            y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
            y *= np.exp(-u * u / 2)
            return y

        prob = [n_pdf(x_i, data, width) for x_i in x_axis] # each row is one x value
        pdf = [np.average(pr) for pr in prob] # each row is one x value
        return np.array(pdf) # this returns a vector of pdf values
    return ksd

"""Plot normal distribution"""
fig, ax = plt.subplots(2) # 2 subplots
x = np.random.randn(1000) # 1000 random numbers from normal distribution
ax[0].hist(x, bins=30) # number of bins
x_n = np.linspace(-3, 3, 100)
ax[0].plot(x_n, 225 * stats.norm.pdf(x_n)) # plot a scaled normal distribution


ks_density = ksdensity(x, width=0.4) # width is the bandwidth
#np.linspace(start, stop, number of steps)
x_values = np.linspace(-5., 5., 100)
ax[1].plot(x_values, ks_density(x_values))

"""Plot uniform distribution"""
fig2, ax2 = plt.subplots(2)
x = np.random.rand(1000) # 1000 random numbers from uniform distribution
ax2[0].hist(x, bins=20)
ks_density = ksdensity(x, width=0.06) # width is the bandwidth
x_values = np.linspace(-1., 2., 100) # x values to evaluate the density at
ax2[1].plot(x_values, ks_density(x_values))
ax2[1].plot(x_values, stats.uniform.pdf(x_values)) # plot a scaled uniform distribution
plt.show()

"""For N = 100, 1000 and 10000, plot the histogram """
fig3, ax3 = plt.subplots(3)
x_n = np.linspace(0, 1, 100)

for i in range(3):
    x = np.random.rand(10**(i+2))
    line = np.ones(100)
    mean = line * ((10**(i+2))/30)
    three_std = [3*np.sqrt(i) for i in mean*(1-(1/30))]
    ax3[i].hist(x, bins=30)
    #plot the uniform distributions 3 standard deviations away from the mean
    ax3[i].plot(x_n , mean) #mean
    ax3[i].plot(x_n, mean - three_std) #mean - 3*std
    ax3[i].plot(x_n, mean + three_std) #mean + 3*std
    ax3[i].set_title('N = 10' + '0'*(i+2))
    
plt.show()

""" Now using the Jacobian change of variable formula, plot the histogram of the normally distributed data with y=ax+b"""
fig4, ax4 = plt.subplots(1)
x = np.random.randn(1000)
y = 5*x + 5 # a = 5, b = 5
ax4.hist(y, bins=30)
#Overlay the normal distribution calculated using the Jacobian change of variable formula
x_n = np.linspace(-20, 25, 1000)
ax4.plot(x_n, 1000* stats.norm.pdf((x_n - 5)/5)*0.2)
ax4.set_title('y = 5x + 5 Using Jacobian change of variable formula')
plt.show()

""" Now do the same for y = x^2 """
fig5, ax5 = plt.subplots(1)
x = np.random.randn(1000)
y = x**2
ax5.hist(y, bins=30)
ax5.plot(x_n, 400* stats.norm.pdf(np.sqrt(x_n))/np.sqrt(x_n))
ax5.set_title('y = x^2 Using Jacobian change of variable formula')
plt.show()

""" Inverse CDF method of generating exponential distribution """
#Generate x(i) using the uniform distribution
x = np.random.rand(1000)
#Generate y(i) using the inverse CDF method
y = -np.log(1-x)
#Plot the histogram of y(i)
fig6, ax6 = plt.subplots(2)
ax6[0].hist(y, bins=30)
#Plot the exponential distribution
x_n = np.linspace(0, 10, 1000)
ax6[0].plot(x_n, 400* stats.expon.pdf(x_n))
ax6[0].set_title('Exponential distribution using inverse CDF method - Histogram')
#Plot kernel density estimates
ks_density = ksdensity(y, width=0.4) # width is the bandwidth
ax6[1].set_title('Exponential distribution using inverse CDF method - KS density')
ax6[1].plot(x_n, ks_density(x_n))
ax6[1].plot(x_n, stats.expon.pdf(x_n)) # plot a scaled exponential distribution
plt.show()

""" Simulation from a difficult density"""
#Choose parameters alpha and beta.
alpha = [0.5, 1.5] #shape parameter
beta = [-1, 0, 1]
for i in alpha:
    for j in beta:
        b = (1/i) * np.arctan(j*np.tan(np.pi*i/2)) #location parameter
        s = (1 + (j**2)*(np.tan(np.pi*i/2))**2)**(1/(2*i)) #scale parameter
        #Generate a uniform random variable U from the interval [-pi,pi]
        u = np.random.uniform(-np.pi, np.pi, 1000)
        #Generate an exponential random variable with mean 1
        v = np.random.exponential(1, 1000)
        #Use U and V to calculate X
        x = b + s*np.sin(i*(u+b))/((np.cos(u))**(1/i))*(np.cos(u-i*(u+b))/v)**((1-i)/i)
        #Plot the histogram of X
        fig7, ax7 = plt.subplots(1)
        ax7.hist(x, bins=50)
        ax7.set_title('alpha = ' + str(i) + ', beta = ' + str(j))
        plt.show()
