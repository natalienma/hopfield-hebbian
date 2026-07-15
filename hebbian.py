
def bump_weight(n, a_pre, a_post):
    delta_w = n * a_pre *  a_post 
    return delta_w

def stdp(t_post, t_pre, A):
    A_plus = 0 # ???
    A_minus = 0 # ???
    tau_plus = 1 # ???
    tau_minus = -1 # ???
    e = 2.718281828459045 # ???

    delta_t = t_post - t_pre

    if delta_t > 0:
        delta_w = A_plus * e^(-1* delta_t/tau_plus)
    else:
        delta_w = (-1 * A_minus) * e^(delta_t/tau_minus)

    # Δt = t_post − t_pre
    # if Δt > 0 (pre fired before post):   Δw = +A⁺ · e^(−Δt/τ⁺)     → strengthen
    # if Δt < 0 (pre fired after post):    Δw = −A⁻ · e^(+Δt/τ⁻)     → weaken
    return delta_w