def ddxdtt(x, dxdt, mu):
    return mu * (1 - x**2) * dxdt - x

def dudt(u, mu):
    x, dxdt = u
    dudt = array([dxdt, ddxdtt(x, dxdt, mu)])
    return dudt

from scipy.integrate import odeint

def run(u, mu, nsteps):
    x = empty(nsteps)
    for i in range(nsteps):
        u = odeint(lambda u,t : dudt(u, mu), u, [0, 0.01])[1]
        x[i] = u[0]
    return u, x
    
u, x = run([1, 1], 1, 5000)

mu_array = linspace(0.9, 1.1, 21)
x2_mean = empty(mu_array.size)
for i, mu in enumerate(mu_array):
    _, x = run([1, 1], mu, 5000)
    x2_mean[i] = (x**2).mean()
plot(mu_array, x2_mean, 'o-')    


mu_array = linspace(0.9, 1.1, 21)
x2_mean = empty(mu_array.size)
for i, mu in enumerate(mu_array):
    _, x = run([1, 1], mu, 50000)
    x2_mean[i] = (x**2).mean()
plot(mu_array, x2_mean, 'o-')


from fds import shadowing

def run_x2(u0, mu, nsteps):
    u, x = run(u0, mu, nsteps)
    return u, x**2

J, G = shadowing(run_x2, [1,1], 1, 1, 10, 500, 500)
