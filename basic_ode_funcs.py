import scipy
import numpy as np
from scipy.integrate import solve_ivp # ode solver
import matplotlib.pyplot as plt
import sympy
from math import *
from sympy.abc import x # for determining power series solving ODE in small time

INITIALIZE_FOR_Y3 = False
NUM_COEFFS = 36
T_MIN_ARB = 1e-5
T_MIN = 0.01
RAD_OF_CONVERGENCE = 200

# evaluate power series for initial condition
# b = y(0) is initial height, and t is initial time
# testing indicates that t = 0.01 is in the radius of convergence for b up to 200
def eval_initial_cond(b, t):
    if t >= 0.99 * T_MIN :
        assert(b < RAD_OF_CONVERGENCE)
    y = 0
    yd = 0
    for k in range(0, len(ps_coeffs), 2):
        sv = ps_coeffs[k].subs(x, b)
        # print(sv)
        y += sv * t**k
        if k > 0:
            yd += k * sv * t**(k-1)
    return y, yd

# t is time, y = [y(t), y'(t)] is phase space vector
# returns derivative [y'(t), y''(t)]
def my_ode(t, y):
    return [y[1], -2*y[1]/t + y[0] - y[0]**3]

def make_ode_p(p):
    def the_ode(t,y):
        return [y[1], -2*y[1]/t + y[0] - y[0]**p]
    return the_ode

# we use scipy.integrate.solve_ivp to solve ode, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
# there are many options, here are some important ones:
# - 'method' is one of RK45, RK23, DOP853, Radau, BDF, LSODA. RK45 is the default, DOP853 is good for high precision
# - 't_eval' is the times at which to evaluate the solution
# - 'events' allows one to detect and evalute the ODE when conditions occur, such as y = 0 or y' = 0
# - 'atol' and 'rtol' are absolute and relative error tolerances. Total error must be within atol + |y| * rtol

# b is the initial height, T is the time up to which to solve
# returns an object with several fields, the most important of which is sol.y, which gives the solution
def calc_sol(b, T, t_eval = 0, **kwargs):
    if t_eval == 0:
        t_eval = np.arange(T_MIN, T, 0.1)

    y, yd = eval_initial_cond(b, T_MIN)
    # print(b, y, yd,  T_MIN, kwargs)
    # print(my_ode, T_MIN, T, y, yd, kwargs)
    sol = solve_ivp(my_ode, [T_MIN, T], [y, yd], t_eval = t_eval, **kwargs)
    return sol

def calc_sol_arb(b, T, ode, t_eval = None, **kwargs):
    sol = solve_ivp(ode, [T_MIN_ARB, T], [b, 0], t_eval = t_eval, **kwargs)
    return sol

def plot_sol(b, T, t_eval = None, atol = 1e-6, rtol = 1e-6, **kwargs):
    if t_eval is None:
        t_eval = np.arange(T_MIN, T, 0.1)
    s = calc_sol(b, T, t_eval = t_eval, atol = atol, rtol = rtol, **kwargs)
    plt.plot(t_eval, s.y[0])
    
def plot_sol_arb(b, T, ode, t_eval = None, atol = 1e-6, rtol = 1e-6, **kwargs):
    if t_eval is None:
        t_eval = np.arange(T_MIN, T, 0.1)
    s = calc_sol_arb(b, T, ode, t_eval = t_eval, atol = atol, rtol = rtol, **kwargs)
    plt.plot(t_eval, s.y[0])

def at_large_time(b, T = 50, tol=1e-8):
    s = calc_sol(b, T, atol=tol, rtol=tol)
    # print(b, s)
    return s.y[0][-1]

# find an excited state inbetween l and u using binary search
def find_excited_state(l, u, tol = 1e-8, verbose=False):
    v1 = (at_large_time(u) > 0)
    v2 = (at_large_time(l) > 0)
    
    if v1 == v2:
        return "lower and upper go into same well"

    i = 1
    while u - l > tol:
        m = (u+l)/2
        if i % 1 == 0 and verbose:
            print("{:2d}: m = {:.10f}, rat = {:e}".format(i, m, (u-l)/tol))
        
        lt = at_large_time(m, tol = tol) > 0
        if lt == v1:
            u = m
        else:
            l = m
        
        i += 1
    
    return m

    # we use the "events" option in scipy.integrate.solve_ivp
def at_0(t, y): return y[0]
def at_left_well(t,y): return y[0] + 1
def at_right_well(t,y): return y[0] - 1
def stationary(t,y): return y[1]

def calc_event(b, T, func, n = 0, rtol=1e-5, atol=1e-7, **kwargs):
    y, yd = eval_initial_cond(b, T_MIN)
    
    sol = solve_ivp(my_ode, [T_MIN, T], [y, yd], events = func, dense_output = True, rtol=rtol, atol=atol, **kwargs)
    if len(sol.t_events[0]) <= n:
        return None
    te = sol.t_events[0][n]
    return te, sol.sol(te)

def calc_event_arb(b, T, ode, func, n = 0, rtol=1e-5, atol=1e-7, **kwargs):
    sol = solve_ivp(ode, [T_MIN_ARB, T], [b, 0], events = func, dense_output = True, rtol=rtol, atol=atol, **kwargs)
    if len(sol.t_events[0]) <= n:
        return None
    te = sol.t_events[0][n]
    return te, sol.sol(te)

def energy(y):
    return y[1]**2 / 2 + y[0]**4/4 - y[0]**2/2

# Because there is a singularity at 0, we jumpstart our numerical solution using a power series
# Here we compute the coefficients of a power series giving small time solutions 

def make_ode_force(f):
    def the_ode_f(t,y):
        return [y[1], -2*y[1]/t + -f(y[0])]
    return the_ode_f

b0 = 0
b1 = 0
b2 = 0
b4 = 0

def calculate_ps_coeffs(N):
    my_coeffs = [1.0 * x, sympy.Poly(0.0, x, domain='RR')] # y(x,t) = ps_coeffs[0] + ps_coeffs[1] * t + ... for t small

    def get_coeff(k):
        cubed_val = 0
        for i in range(k-2+1):
            for j in range(k-2-i+1):
                l = k-2 - i - j
                s = my_coeffs[i] * my_coeffs[j] * my_coeffs[l]
                cubed_val += s
        c = (my_coeffs[k-2] - cubed_val) / (k*(k+1))
        c = sympy.expand(c)
        return c

    print("computing power series coeffs")
    for k in range(2,N+1):
        print("On coefficient:", k , "/", N)
        my_coeffs.append(get_coeff(k))

    return my_coeffs

if INITIALIZE_FOR_Y3:
    ps_coeffs = calculate_ps_coeffs(NUM_COEFFS)


def calc_excited_states():
    global b0,b1,b2,b3,b4
    print("Finding excited states")
    b0 = find_excited_state(3, 7)
    print("Found b0")
    b1 = find_excited_state(7, 20)
    print("Found b1")
    b2 = find_excited_state(20, 40)
    print("Found b2")
    b3 = find_excited_state(40, 60)
    print("Found b3")
    b4 = find_excited_state(60, 80)
    print("Found b4")  
    bound_states = [b0, b1, b2, b3, b4]
    for i,b in enumerate(bound_states):
        print("b{:d} = {:13.10f}".format(i, b))



