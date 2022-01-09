# coding: utf-8
def interp(arr, stretch=2):
    l, r = arr.T
    n = len(l)
    t = np.linspace(0,1,n)
    t_stretch = np.linspace(0,1,n*stretch)
    l_stretch = np.interp(t_stretch, t, l)
    r_stretch = np.interp(t_stretch, t, r)
    stretched_arr = np.column_stack((l_stretch, r_stretch))
    return stretched_arr.astype('int16')
def repeat_stretch(arr, stretch=2):
    l,r = arr.T
    ls = np.repeat(l, stretch, axis=0)
    rs = np.repeat(r, stretch, axis=0)
    return np.column_stack((ls,rs))
