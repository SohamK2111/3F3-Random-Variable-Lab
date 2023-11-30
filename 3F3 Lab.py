import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.widgets import Slider
import matplotlib.cm as cm
import math


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

# """Plot normal distribution"""
# fig, ax = plt.subplots(2) # 2 subplots
# x = np.random.randn(1000) # 1000 random numbers from normal distribution
# ax[0].hist(x, bins=30) # number of bins
x_n = np.linspace(-3, 10, 1000)
# ax[0].plot(x_n, 220 * stats.norm.pdf(x_n)) # plot a scaled normal distribution
# ax[1].plot(x_n, stats.norm.pdf(x_n)) # plot a scaled normal distribution
# ax[0].legend(['Normal distribution', 'Sample Histogram'])

# # ax[0].text(0.05, 0.7, "$N$ = 1000 \nbins = 30", bbox=dict(facecolor='white', alpha=0.5), transform=ax[0].transAxes)

# # ks_density = ksdensity(x, width=0.4) # width is the bandwidth
# # #np.linspace(start, stop, number of steps)
# # x_values = np.linspace(-5., 5., 100)
# # ax[1].plot(x_values, ks_density(x_values))
# # ax[1].legend(['KS density', 'Normal distribution'])
# # ax[1].text(0.05, 0.7, "$N$ = 1000 \nwidth = 0.4", bbox=dict(facecolor='white', alpha=0.5), transform=ax[1].transAxes)


# # """Plot uniform distribution"""
# # fig2, ax2 = plt.subplots(2)
# # x = np.random.rand(1000) # 1000 random numbers from uniform distribution
# # ax2[0].hist(x, bins=30)
# # ks_density = ksdensity(x, width=0.06) # width is the bandwidth
# # x_values = np.linspace(-1., 2., 100) # x values to evaluate the density at
# # ax2[1].plot(x_values, ks_density(x_values))
# # ax2[1].text(0.05, 0.7, "$N$ = 1000 \nwidth = 0.06", bbox=dict(facecolor='white', alpha=0.5), transform=ax2[1].transAxes)


# # ax2[0].plot(x_values, 33*stats.uniform.pdf(x_values)) # plot a scaled uniform distribution
# # ax2[1].plot(x_values, stats.uniform.pdf(x_values)) # plot a scaled uniform distribution
# # ax2[1].legend(['KS Density', 'Uniform distribution'])
# # ax2[0].legend(['Uniform distribution', 'Sample Histogram'])
# # ax2[0].text(0.05, 0.7, "$N$ = 1000 \nbins = 30", bbox=dict(facecolor='white', alpha=0.5), transform=ax2[0].transAxes)



# # plt.show()

# """For N = 100, 1000 and 10000, plot the histogram """
# fig3, ax3 = plt.subplots(3)
# x_n = np.linspace(0, 1, 100)

# for i in range(3):
#     x = np.random.rand(10**(i+2))
#     line = np.ones(100)
#     mean = line * ((10**(i+2))/30)
#     three_std = [3*np.sqrt(i) for i in mean*(1-(1/30))]
#     ax3[i].hist(x, bins=30)
#     #plot the uniform distributions 3 standard deviations away from the mean
#     ax3[i].plot(x_n , mean) #mean
#     ax3[i].plot(x_n, mean - three_std) #mean - 3*std
#     ax3[i].plot(x_n, mean + three_std) #mean + 3*std
#     ax3[i].set_title('N = 1' + '0'*(i+2))
#     if i == 0:
#         ax3[i].legend(['$\mu$', '$\mu - 3\sigma$', '$\mu + 3\sigma$'])
        
    
# plt.show()

# """ Repeat the above for the normal distribution """
# #Calculate the probability of falling in each bin
# fig3, ax3 = plt.subplots(3)
# x_n = np.linspace(-3, 3, 100)

# for i in range(3):
#     x = np.random.randn(10**(i+2)) # this is an array of 1000 random numbers from normal distribution
#     x = [x[i] for i in range(len(x)) if x[i] >= -3 and x[i] <= 3] # remove outliers - cruicial for estimates to fit the data because otherwise the outliers will skew the mean and standard deviation
#     h, xedges, pathes = ax3[i].hist(x, bins=30) 
#     #Obtain bin centres manually by dividing the range(-3,3) into 30 bins and taking the midpoints
#     binlist = []
#     for j in range(30):
#         binlist.append(-3 + (j+1)*6/30)
#     binlist = np.array(binlist)
#     bincentre = binlist - 0.1 #subtract 0.1 to get the bin centres
#     #Obtain probabilities of falling into these bins
#     binprobs = []
#     for j in range(30):
#         binprobs.append(stats.norm.cdf(bincentre[j]+0.1) - stats.norm.cdf(bincentre[j]-0.1))
#     binprobs = np.array(binprobs)
#     print(binprobs)
#     variance = binprobs* 10**(i+2) * (1 - binprobs)
#     std = np.sqrt(variance)
#     ax3[i].set_title('N = 1' + '0'*(i+2))
#     ax3[i].set_xlim(-3, 3)
#     #plot mean and 3 standard deviations
#     ax3[i].plot(bincentre, binprobs * 10**(i+2), 'x') #mean
#     ax3[i].plot(bincentre, binprobs * 10**(i+2) - 3*std, 'x') #mean - 3*std
#     ax3[i].plot(bincentre, binprobs * 10**(i+2) + 3*std, 'x') #mean + 3*std
#     if i == 0:
#         ax3[i].legend(['$\mu$', '$\mu - 3\sigma$', '$\mu + 3\sigma$'])
# plt.show()

# ax8 = plt.subplot(111)
# #plot the variation of bin variance with probability of falling into the bin
# binprobs = np.linspace(0, 1, 30)
# variance = binprobs*(1-binprobs)
# ax8.plot(binprobs, variance, 'x')
# ax8.set_xlabel('Probability of falling into bin')
# ax8.set_ylabel('Variance of bin')
# plt.show()

# """ Now using the Jacobian change of variable formula, plot the histogram of the normally distributed data with y=ax+b"""
# fig4, ax4 = plt.subplots(1)
# x = np.random.randn(1000)
# y = 5*x + 10 # a = 3, b = 5
# ax4.hist(y, bins=40)
# #Overlay the normal distribution calculated using the Jacobian change of variable formula
# x_n = np.linspace(-20, 40, 1000)
# ax4.plot(x_n, 800* stats.norm.pdf((x_n - 10)/5)*0.2)
# ax4.set_title('$y = 5x + 10$ using Jacobian change of variable formula')
# plt.show()


# """ Now do the same for y = x^2 """
# fig5, ax5 = plt.subplots(1)
# x = np.random.randn(1000)
# y = x**2
# ax5.hist(y, bins=30)
# ax5.plot(x_n, 410* stats.norm.pdf(np.sqrt(x_n))/np.sqrt(x_n))
# ax5.set_title('$y = x^2$ using Jacobian change of variable formula')
# plt.show()


# """ Now taking a uniform distribution between 0 and 2pi, plot the histogram of y = sin(x) """
# fig5, ax5 = plt.subplots(1)
# x_n = np.linspace(-1, 1, 1000)
# x = np.random.uniform(0, 2*np.pi, 5000)
# y = np.sin(x)
# ax5.hist(y, bins=30)
# #now using jacobian change of variable formula and counting two solutions for each x plot the distribution of y
# ax5.plot(x_n, 350/(np.pi*np.sqrt(1-x_n**2)))
# ax5.set_title('y = sin(x) Using Jacobian change of variable formula')

# plt.show()

# """ Now y = min(sin(x), 0.7)"""
# x_n = np.linspace(-1, 0.7, 1000)
# fig5, ax5 = plt.subplots(1)
# x = np.random.uniform(0, 2*np.pi, 5000)
# y = np.minimum(np.sin(x), 0.7)
# ax5.hist(y, bins=40)
# ax5.plot(x_n, 220/(np.pi*np.sqrt(1-x_n**2)))
# ax5.set_title('y = min(sin(x), 0.7) Using Jacobian change of variable formula')

# plt.show()

# """ Inverse CDF method of generating exponential distribution """
# #Generate x(i) using the uniform distribution
# x = np.random.rand(1000)
# #Generate y(i) using the inverse CDF method
# y = -np.log(1-x)
# #Plot the histogram of y(i)
# fig6, ax6 = plt.subplots(2)
# ax6[0].hist(y, bins=30)
# #Plot the exponential distribution
# x_n = np.linspace(0, 10, 1000)
# ax6[0].plot(x_n, 220* stats.expon.pdf(x_n))
# ax6[0].set_title('Exponential distribution using inverse CDF method - Histogram')
# #Plot kernel density estimates
# ks_density = ksdensity(y, width=0.4) # width is the bandwidth
# ax6[1].set_title('Exponential distribution using inverse CDF method - KS density')
# ax6[1].plot(x_n, ks_density(x_n))
# ax6[1].plot(x_n, stats.expon.pdf(x_n)) # plot a scaled exponential distribution
# plt.show()

# """Estimate the mean and variance of the exponential distribution with lambda = 1 using the samples generated"""
# #Estimating mean via Monte Carlo method
# mean = 1/1000 * np.sum(y)
# variance = 1/1000 * np.sum(y**2 - mean**2)
# # the theoretical mean and variance of the exponential distribution are 1 and 1 respectively because
# # the exponential distribution has mean 1/lambda and variance 1/lambda^2
# print(mean, variance)
# # The estimates are very close to the theoretical values

# """Show that the monte carlo mean approaches the real mean by plotting the error as a function of N"""
# errors = []
# Ns = [100, 1000, 10000, 100000, 1000000]
# Ns_smooth = np.linspace(100, 1000000, 100000)
# for i in range(5):
#     means = []
#     #for each N, generate 20 samples and calculate the mean
#     for j in range(20):
        
#         N = 10**(i+1)
#         x = np.random.rand(N)
#         y = -np.log(1-x)
#         mean = 1/(N) * np.sum(y)
#         means.append(mean)
#     #calculate the error
#     error = abs(np.mean(means) - 1)
#     errors.append(error)
#     print(np.mean(means))
# #plot graph with Ns in logarithmic scale
# plt.plot(Ns, errors, 'x')
# #plot a curve of best fit using polyfit
# plt.plot(Ns_smooth, 0.9*1/np.sqrt(Ns_smooth))
# plt.xscale('log')
# plt.xlabel('N')
# plt.ylabel('Error')
# #plt.axis([0, 110000, 0, 0.2])
# plt.title('Error in Monte Carlo estimate of mean as a function of N')
# plt.legend(['Error ($\hat{\mu}^2 - \mu^2$)', '$K\sqrt{N}}$'])
# plt.show()


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
        ax7.hist(x, bins=30)
        ax7.set_title('alpha = ' + str(i) + ', beta = ' + str(j))
        plt.show()

"""With beta = 0 estimate the tail probability for the distribution"""
#Probability of X > t and X < -t for t = 0 , 3 , 6
t = [0, 3, 6]
beta = 0 #symmetric distribution
alpha = [0.5, 1.5] #0.5 has high tail probabilities whereas 1.5 has low tail probabilities

for i in alpha:
    b = (1/i) * np.arctan(0) #location parameter
    s = (1)**(1/(2*i)) #scale parameter
    #Generate a uniform random variable U from the interval [-pi,pi]
    u = np.random.uniform(-np.pi, np.pi, 1000)
    #Generate an exponential random variable with mean 1
    v = np.random.exponential(1, 1000)
    #Use U and V to calculate X
    x = s*np.sin(i*(u+b))/((np.cos(u))**(1/i))*(np.cos(u-i*(u+b))/v)**((1-i)/i)
    #cleanse x of nan values
    x = np.array([x[i] for i in range(len(x)) if not np.isnan(x[i])])
    #obtain the size of x
    len_x = len(x)
    #Plot the histogram of X
    fig7, ax7 = plt.subplots(1)
    ax7.hist(x, bins=30, density=True) # Density = True normalises the histogram so that large bins do not dominate the plot
    #Calculate the tail probability i.e. P(X > t) and P(X < -t)
    tail_prob_0 = np.sum(x > 0)/len_x + np.sum(x < 0)/len_x
    tail_prob_3 = np.sum(x > 3)/len_x + np.sum(x < -3)/len_x
    tail_prob_6 = np.sum(x > 6)/len_x + np.sum(x < -6)/len_x
    ax7.set_title('$ \\alpha $ = ' + str(i) + ', $ \\beta $ = 0' )
    ax7.text(0.1, 0.8, "Tail probabilities: \n $P(|X|>0) = 1$ \n $P(|X|>3) = $" + str(tail_prob_3.round(3)) + "\n $P(|X|>6) = $" + str(tail_prob_6.round(3)), bbox=dict(facecolor='white', alpha=0.5), transform=ax7.transAxes)
    plt.show()

# # the tail probability is very high for alpha = 0.5 and very low for alpha = 1.5 as expected
# # The tail probability is also very high for t = 0 and very low for t = 6 as expected
# # the tail probability for t = 0 is 1 because the distribution is symmetric about 0 but when alpha = 1.5, the tail probability is not exactly 0 because the distribution is not exactly symmetric about 0

"""Tail probabilities for the standard Gaussian distribution"""
#Calculate the tail probabilities
tail_prob_0 = (stats.norm.cdf(-0) + (1 - stats.norm.cdf(0)))
tail_prob_3 = (stats.norm.cdf(-3) + (1 - stats.norm.cdf(3)))
tail_prob_6 = (stats.norm.cdf(-6) + (1 - stats.norm.cdf(6)))
print("Gaussian tails: " + str(tail_prob_0), str(tail_prob_3), str(tail_prob_6))


"""For alpha = 0.5 and 1.5, determine the tail behaviour of p(x) = cx^(-gamma) for large |x|"""
beta = 0 #symmetric distribution
alpha = [0.5,1,1.5,1.75] #0.5 has high tail probabilities whereas 1.5 has low tail probabilities

for i in alpha:
    b = (1/i) * np.arctan(0) #location parameter
    s = (1)**(1/(2*i)) #scale parameter
    #Generate a uniform random variable U from the interval [-pi,pi]
    u = np.random.uniform(-np.pi, np.pi, 1000000)
    #Generate an exponential random variable with mean 1
    v = np.random.exponential(1, 1000000)
    #Use U and V to calculate X
    x = s*np.sin(i*(u+b))/((np.cos(u))**(1/i))*(np.cos(u-i*(u+b))/v)**((1-i)/i) 
    #cleanse x of nan values
    x = np.array([x[i] for i in range(len(x)) if not np.isnan(x[i])])
    #obtain the size of x
    len_x = len(x)
    #Plot the histogram of X
    fig7, ax7 = plt.subplots(1)
    #plot tail behaviour
    c1 = np.sin(np.pi*0.5/2)*math.gamma(0.5)/np.pi * 0.5 * s**0.5
    gamma1 = -1.5 # because alpha = 0.5 where alpha is the "tail exponent". A gaussian has tail exponent = 2, and its tail behaviour is x^(-2)
    
    c2 = np.sin(np.pi*1/2)*math.gamma(1)/np.pi* 1 * s**1
    gamma2 = -2.0 # because alpha = 1
    
    c3 = np.sin(np.pi*1.5/2)*math.gamma(1.5)/np.pi* 1.5 * s**1.5
    gamma3 = -2.5 # because alpha = 1.5

    c4 = np.sin(np.pi*1.75/2)*math.gamma(1.75)/np.pi* 1.75 * s**1.75
    gamma4 = -2.75 # because alpha = 1.75
    
    #print(c1, c2, c3, c4)
    c1 = c2 = c3 = c4 = 0.25 # Remove this line to obtain the accurate values for c1, c2, c3 and c4
    
    if i == 0.5: 
        x_n = np.linspace(10000, 5*10**10, 1000000)
        ax7.hist(x, bins=10000, density=True) # Density = True normalises the histogram so that large bins do not dominate the plot
        ax7.plot(x_n, c1*x_n**(gamma1)) 
        ax7.set_title('$ \\alpha $ = ' + str(i) + ', $ \\beta $ = 0' )
        ax7.text(0.65, 0.7, "Tail-bounding curve: \n $p(x) = 0.1994^{-1.5}$", bbox=dict(facecolor='white', alpha=0.5), transform=ax7.transAxes)

    elif i == 1: 
        x_n = np.linspace(100, 15000, 8000)
        ax7.hist(x, bins=8000, density=True)
        ax7.plot(x_n, c2*x_n**(gamma2))
        ax7.set_title('$ \\alpha $ = ' + str(i) + ', $ \\beta $ = 0' )
        ax7.text(0.65, 0.7, "Tail-bounding curve: \n $p(x) = 0.3183x^{-2}$", bbox=dict(facecolor='white', alpha=0.5), transform=ax7.transAxes)
    
    elif i == 1.75: 
        x_n = np.linspace(2, 4000, 6000)
        ax7.hist(x, bins=2000, density=True)
        ax7.plot(x_n, c4*x_n**(gamma4))
        ax7.set_title('$ \\alpha $ = ' + str(i) + ', $ \\beta $ = 0' )
        ax7.text(0.65, 0.7, "Tail-bounding curve: \n $p(x) = 0.1959x^{-2.75}$", bbox=dict(facecolor='white', alpha=0.5), transform=ax7.transAxes)
        
    else: # i = 1.5 
        x_n = np.linspace(-400, 400, 1600)
        ax7.hist(x, bins=2000, density=True) 
        ax7.plot(x_n, c3*x_n**(gamma3))
        ax7.set_title('$ \\alpha $ = ' + str(i) + ', $ \\beta $ = 0' )
        ax7.text(0.65, 0.7, "Tail-bounding curve: \n $p(x) = 0.2992x^{-2.5}$", bbox=dict(facecolor='white', alpha=0.5), transform=ax7.transAxes)

    plt.show()
    
"""Investigate the characteristics of the distribution as alpha approaches 2"""
#alpha = 2, beta = 0, plot the distribution and overlay a gaussian distribution
b = (1/2) * np.arctan(0) 
s = (1)**(1/(2*2)) 
#Generate a uniform random variable U from the interval [-pi,pi]
u = np.random.uniform(-np.pi, np.pi, 500000)
#Generate an exponential random variable with mean 1
v = np.random.exponential(1, 500000)
#Use U and V to calculate X
x = s*np.sin(2*(u+b))/((np.cos(u))**(1/2))*(np.cos(u-2*(u+b))/v)**((1-2)/2)
#Plot the histogram of X
fig7, ax7 = plt.subplots(1)
#plot histogram
ax7.hist(x, bins=500, density=True) # Density = True normalises the histogram so that large bins do not dominate the plot
#plot gaussian distribution
x_n = np.linspace(-10, 10, 2000)
ax7.plot(x_n, stats.norm.pdf(x_n, 0, 1.414213562)) 
ax7.set_title('$ \\alpha $ = 2, $ \\beta $ = 0' )
ax7.legend(['Gaussian distribution, $\mu=0, \sigma = \sqrt{2}$', 'Stable distribution, $ \\alpha = 2, \\beta = 0$'])
plt.show()
