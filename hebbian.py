
def hebbian(n, a_pre, a_post):
    delta_w = n * a_pre *  a_post 
    return delta_w