from basic_ode_funcs import *
from scipy.interpolate import interp1d

def make_line(x0, x1, y0, y1):
        return (lambda x: (y0 + ((y1-y0)/(x1-x0)) * (x - x0)))
    
def make_zig_zag(num_negative,negative_width,negative_height,positive_width,positive_height, ep):
    last_neg = num_negative * negative_width
        
    ngleft = make_line(0, negative_width/2, 0, -negative_height) 
    ngright = make_line(negative_width/2, negative_width, -negative_height, 0) 
    ng1 = make_line(0, negative_width/2, -ep, -negative_height) 
    ng2 = make_line(negative_width/2, negative_width, -negative_height, -ep) 
    
    pgleft = make_line(0, positive_width/2, 0, positive_height) 
    pg1 = make_line(0, positive_width/2, ep, positive_height) 
    pg2 = make_line(positive_width/2, positive_width, positive_height, ep) 
    
    def negative_zig(y, ip):
        if y <= negative_width / 2:
            return ng1(y) if ip > 0 else ngleft(y)
        else:
            return ng2(y) if ip < (num_negative-1) else ngright(y)
    
    def positive_zig(y, ip):
        if y <= positive_width / 2:
            return pg1(y) if ip > 0 else pgleft(y)
        else:
            return pg2(y) 
        
    def zig_force(y):
        oy = y
        y = abs(y)
        v = 0
        if y <= last_neg:
            indp = floor(y / negative_width)
            yt = y - indp * negative_width
#             print(indp, yt)
            v = negative_zig(yt, indp)
            if indp % 2 == 1 and (indp+1)<num_negative:
                v = -ep
        else:
            a = y - last_neg
            indp = floor(a / positive_width)
            yt = a - indp * positive_width
            v = positive_zig(yt, indp) 
            if indp % 2 == 1:
                v = ep 
        return v if oy >= 0 else -v
    return zig_force

# goes down from 0 to -1 in 1 unit 
# then goes up from -1 to 0 in 1 unit
# then has a series of spikes going up to a and down to b in width w, and 
# then plateuing for length p
# end with a plateu up to y = MAX_LEN
MAX_LEN = 200
def make_my_interp(a, b, w, p, num_spikes, **kwargs):
    x = [0, 1, 2] 
    y = [0, -1, 0]
    last_x = x[-1]
    for i in range(num_spikes):
        # spike up
        x.append(last_x + w/2)
        y.append(a)

        # spike down
        x.append(last_x + w)
        y.append(b)

        # plateua
        x.append(last_x + w + p)
        y.append(b)

        last_x = x[-1]

    x.append(MAX_LEN)
    y.append(b)

    interpf = interp1d(x, y, **kwargs)
    def interpf_abs(y):
        # print(y, interpf(abs(y)))
        if y >= 0: 
            return interpf(y)
        else:
            return -interpf(-y)

    return interpf_abs

# one that worked before: 
# zf = make_my_interp(0.8, 0.02, 2, 2, 5, kind="slinear")



MAX_LEN = 200
# goes down to -1, then up to mh, then down to h, plateaus there for a while, 
# then goes up at slope m
# params that work: make_my_interp2(1.5, 5, 0.05, 3)
# 8 crosses, 15 doesnt
def make_my_interp2(mh, a, h, m, **kwargs):
    x = [0,  1, 2, 3, 4, 4+a, 4+a+MAX_LEN]
    y = [0, -1, 0, mh, h,   h, h+m * MAX_LEN]

    interpf = interp1d(x, y, **kwargs)
    def interpf_abs(y):
        # print(y, interpf(abs(y)))
        if y >= 0: 
            return interpf(y)
        else:
            return -interpf(-y)

    return interpf_abs