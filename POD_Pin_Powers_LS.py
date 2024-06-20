#!/usr/bin/env python
# coding: utf-8

# # Imports (Need Python 3.10 Anaconda)

# In[1]:


import utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import image as mpimg
import pickle
from pathlib import Path
import sys
import scipy.integrate as si
import scipy as sp

sys.path.append("../eVinci-like")


# # Function Definitions

# In[22]:


def Qdict_proc(pklpath):
    sym = True

    P = 4e6 # do not change this for symmetry


    with open(pklpath, "rb") as f:
        powdict = pickle.load(f)

    zelems = np.linspace(-89.18, 89.18, 50)


    def c2s(s,q,r):
        ret = ""
        for p in [s,q,r]:
            if p == 0:
                ret += "p0"
            elif p < 0:
                ret += f"m{abs(p)}"
            else:
                ret += f"p{p}"
        return ret

    def l2s(l):
        return "f" + c2s(l[0],l[1],l[2]) + "_p" + c2s(l[3],l[4],l[5])

    #loading data about pin locs in model
    ppc = utils.pin_pipe_centers()[0]
    symppc = {}
    for k, v in ppc.items():
        tangle = np.arctan2(*v[::-1]) 
        if tangle < 0.00001 and tangle > -np.pi/6: 
            symppc[k] = v

    locs = np.array(list(symppc.values()))
    kyss = list(symppc.keys())

    def rotate_vector(vector, angle):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        return rotation_matrix@vector

    def symtrans(k):
        if k in list(kyss):
            return k
        angl = (np.arctan2(*ppc[k][::-1]) + 2*np.pi) % (2*np.pi)
        magn = np.sqrt(ppc[k][0]**2 + ppc[k][1]**2)
        hx = (angl + np.pi/6) // (np.pi/3)
        in_hx = rotate_vector(ppc[k], -hx*np.pi/3)

        if in_hx[1] > 0:
            in_hx[1] *= -1

        karg = np.linalg.norm(in_hx - locs, axis = 1).argmin()
        return kyss[karg]

    Qdict = {}
    for k,v in powdict.items():
        if not sym:
            ky_treat = k
        else:
            ky_treat = symtrans(k)

        pref = l2s(ky_treat)
        if not (pref in list(Qdict.keys())):
            Qdict[pref] = v
        else:
            Qdict[pref] += v


    half_pins = ["fp0p0p0_pm2p2p0","fm1p2m1_pp2m2p0","fm1p2m1_pp1m1p0","fm1p2m1_pm1p1p0","fm1p2m1_pm2p2p0",
                                      "fm2p4m2_pp2m2p0","fm2p4m2_pp1m1p0","fm2p4m2_pm1p1p0","fm2p4m2_pm2p2p0"]


    if sym: #renormalize from half pins
        for k in list(Qdict.keys()):
            if k[:len(half_pins[0])] in half_pins:
                Qdict[k] *= 2
                #Qdict[k] /= 1
        powrr = 0
        for v in Qdict.values():
            powrr += v.sum()
        for k in list(Qdict.keys()):
            Qdict[k] *= P/12/powrr
        powrr = 0
        for v in Qdict.values():
            powrr += v.sum()

    def cvt(n):
        fn = n[1:7]
        pn = n[-6:]
        l = []
        for i in range(3):
            if fn[2*i] == "p":
                l.append(int(fn[2*i+1]))
            elif fn[2*i] == "m":
                l.append(-int(fn[2*i+1]))
        for i in range(3):
            if pn[2*i] == "p":
                l.append(int(pn[2*i+1]))
            elif pn[2*i] == "m":
                l.append(-int(pn[2*i+1]))
        return tuple(l)

    cvt(half_pins[0])
    final = {}
    for k, v in Qdict.items():
        final[cvt(k)] = v

    return final

def plotDict_powers(inQ,clabel="Pin Power [kW]"):
    fig, ax = plt.subplots(1,2,figsize = (9.5, 3.6), width_ratios = [1.7,1])
    zelems = np.linspace(-89.18, 89.18, 50)

    ppc = utils.pin_pipe_centers()[0]
    locs = []
    clrs = []
    for k, v in inQ.items():
        locs.append(ppc[k])
        clrs.append(v.sum())

    locs = np.array(locs)
    clrs = np.array(clrs)


    cmap = matplotlib.colormaps['Spectral_r']
    for k, v in inQ.items():
        ax[1].plot(zelems, v/(zelems[1] - zelems[0]), "-", color = cmap((v.sum()- 2.2e3)/(4.e3 - 2.2e3)), linewidth = 0.6)


    img = mpimg.imread("FOAM_Model_radial_outline.png")
    #img += (1 - img)*.8
    rto = img.shape[0]/img.shape[1]
    mx = 0.78188/1.0176978*.9964
    x0 =-3.226e-2 + (0.05715-0.03793) -1.23e-3 +2.65e-3+1.3e-3
    y0 = -0.00957 + 0.00156-0.0001+7.4e-4#+0.8e-3
    im = ax[0].imshow(img, aspect = "equal", extent = [x0, x0 + mx, -y0 - mx*rto, -y0])
    #ax[0, 0].scatter(locs[:,0], locs[:,1], c = clrs, cmap = "autumn")
    im2 = ax[0].scatter(locs[:,0], locs[:,1], c = clrs/1e3, cmap = "Spectral_r", s = 75) #unit conversion from W to kW
    # im2.set_clim(1, 4.55)
    # im2.set_clim(2.2, 4)
    cbar = fig.colorbar(im2, ax = ax[0], fraction = 0.05, pad = 0.04, orientation = "horizontal", aspect = 50, format = lambda x, _ : "%.2f"%x)
    cbar.set_label(clabel, size = 13)
    cbar.ax.tick_params(labelsize = 13)
    ax[0].set_ylim([-0.412, 0])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xlabel("Axial Position [cm]")
    ax[1].set_ylabel("Linear Heat Rate [W/cm]")
    start, end = ax[1].get_xlim()
    ax[1].xaxis.set_ticks(np.arange(-100, 100 + 25, 25))


    plt.tight_layout()
    plt.show()

def plotDict_errors(inQ,clabel="Difference (%)"):
    
    fig, ax = plt.subplots(1,2,figsize = (9.5, 3.6), width_ratios = [1.7,1])
    zelems = np.linspace(-89.18, 89.18, 50)
    
    ppc = utils.pin_pipe_centers()[0]
    locs = []
    clrs = []
    for k, v in inQ.items():
        locs.append(ppc[k])
        clrs.append(v.mean())

    locs = np.array(locs)
    clrs = np.array(clrs)
    
    cmin = clrs.min()
    cmax = clrs.max()
    
    cmap = matplotlib.colormaps['Spectral_r']
    for k, v in inQ.items():
        ax[1].plot(zelems, v, "-", color = cmap((v.mean()-cmin)/(cmax-cmin)), linewidth = 0.6)
        


    img = mpimg.imread("FOAM_Model_radial_outline.png")
    #img += (1 - img)*.8
    rto = img.shape[0]/img.shape[1]
    mx = 0.78188/1.0176978*.9964
    x0 =-3.226e-2 + (0.05715-0.03793) -1.23e-3 +2.65e-3+1.3e-3
    y0 = -0.00957 + 0.00156-0.0001+7.4e-4#+0.8e-3
    im = ax[0].imshow(img, aspect = "equal", extent = [x0, x0 + mx, -y0 - mx*rto, -y0])
    #ax[0, 0].scatter(locs[:,0], locs[:,1], c = clrs, cmap = "autumn")
    im2 = ax[0].scatter(locs[:,0], locs[:,1], c = clrs, cmap = "Spectral_r", s = 75)
    # im2.set_clim(1, 4.55)
    # im2.set_clim(2.2, 4)
    cbar = fig.colorbar(im2, ax = ax[0], fraction = 0.05, pad = 0.04, orientation = "horizontal", aspect = 50, format = lambda x, _ : "%.2f"%x)
    cbar.set_label(clabel, size = 13)
    cbar.ax.tick_params(labelsize = 13)
    ax[0].set_ylim([-0.412, 0])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xlabel("Axial Position [cm]")
    ax[1].set_ylabel(clabel)
    start, end = ax[1].get_xlim()
    ax[1].xaxis.set_ticks(np.arange(-100, 100 + 25, 25))

    plt.tight_layout()
    plt.show()


def dict2array(d):
    idx=0
    x=np.zeros(50*123)            #hardcoded sizes, 50 planes and 123 pins in serpent models
    for k,v in d.items():
        x[idx:idx+50]=v
        idx+=50                   #Serpent results are tallied on 50 planes
    return x

def array2dict(x,din):
    idx=0
    dout=din
    for k,v in din.items():
        dout[k] = x[idx:idx+50]
        idx+=50
    return dout


# # Load pin power data

# In[3]:


# y and d are dictionaries. y is entries are arrays. d entries are dictionaries.
d = {}
y = {}

cases = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,105,120,135,150,165,180]
rel2abs_power = 4e6/(123*50*12)
for c in cases:
    i = str(c).zfill(3)
    d[i] = Qdict_proc("pinpowers_"+i+".pkl") #absolute power for 50 axial levels in 123 pins
    ytmp = dict2array(d[i])
    y[i] = ytmp/ytmp.mean() #Convert to relative pin powers


# # Plot Normalized Singular Values

# In[35]:


def plot_singular_values(dataset):
    # Build Data Matrix
    X0 = np.zeros([len(y["000"]),len(dataset)])
    idx=0
    for i in dataset:
        k = str(i).zfill(3)
        X0[:,idx] = y[k]
        idx+=1
    # Truncated SVD
    U, S, Vh = np.linalg.svd(X0)
    normS = S/S[0]
    plt.plot(normS)
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    return normS[len(dataset)-1]


# In[43]:


plot_singular_values(cases)


# # Compute Rank 2 POD Basis

# In[15]:


#Build Data Matrix
X0 = np.array([y["000"],y["180"]]).T #Data Matrix


# Truncated SVD
U, S, Vh = np.linalg.svd(X0)
U_r = U[:,0:len(S)]
a_r = np.matmul(np.diag(S),Vh)
amu = sp.interpolate.interp1d([0,180],a_r,axis=1) #piece-wise linear interpolation


#%%
# ## Linear Interpolation Coefficients and Errors



atilde_li = {}
ytilde = {}
error = {}
relerr = {}
rmserr = np.zeros([len(cases)-2])
maxabserr = np.zeros([len(cases)-2])

idx=0
for c in cases[1:-1]:
    i = str(c).zfill(3)
    atilde = amu(c)
    atilde_li[i] = atilde
    ytilde[i] = np.matmul(U_r,atilde)
    
    error[i] = ytilde[i] - y[i]
    relerr[i] = np.divide(error[i],y[i])
    rmserr[idx] = np.linalg.norm(relerr[i])/np.sqrt(len(relerr[i]))
    maxabserr[idx] = np.amax(error[i])
    idx+=1

# plot reconstruction errors
plt.plot(cases[1:-1], rmserr*100, label = "RMS Error") 
plt.plot(cases[1:-1], maxabserr*100, label = "Max. Abs. Error") 
fig = plt.gcf()
ax = plt.gca()
ax.set_title('Linear Interpolation')
ax.set_xlabel("Drum Position (Degrees Rotated Out)")
ax.set_ylabel("POD Reconstruction Error (%)")
plt.legend() 
plt.show()


#%% Moore-Penrose Inverse Coefficients and Errors 


atilde_mp = {}
ytilde = {}
error = {}
relerr = {}
rmserr = np.zeros([len(cases)-2])
maxabserr = np.zeros([len(cases)-2])

idx=0
for c in cases[1:-1]:
    i = str(c).zfill(3)
    atilde = np.matmul(np.linalg.pinv(U_r),y[i])
    atilde_mp[i] = atilde
    ytilde[i] = np.matmul(U_r,atilde)
    error[i] = ytilde[i] - y[i]
    relerr[i] = np.divide(error[i],y[i])
    rmserr[idx] = np.linalg.norm(relerr[i])/np.sqrt(len(relerr[i]))
    maxabserr[idx] = np.amax(error[i])
    idx+=1

# plot lines 
plt.plot(cases[1:-1], rmserr*100, label = "RMS Error") 
plt.plot(cases[1:-1], maxabserr*100, label = "Max. Abs. Error") 
fig = plt.gcf()
ax = plt.gca()
ax.set_title('Moore-Penrose Interpolation')
ax.set_xlabel("Drum Position (Degrees Rotated Out)")
ax.set_ylabel("POD Reconstruction Error (%)")
plt.legend() 
plt.show()


#%% QR Pivot Coefficients and Errors


atilde_qr = {}
ytilde = {}
error = {}
relerr = {}
rmserr = np.zeros([len(cases)-2])
maxabserr = np.zeros([len(cases)-2])

Q,R,P = sp.linalg.qr(U_r,pivoting=True)
theta = np.linalg.inv(U_r[P,:])
idx=0
for c in cases[1:-1]:
    i = str(c).zfill(3)
    atilde = np.matmul(theta,y[i][P])
    atilde_qr[i] = atilde
    ytilde[i] = np.matmul(U_r,atilde)
    error[i] = ytilde[i] - y[i]
    relerr[i] = np.divide(error[i],y[i])
    rmserr[idx] = np.linalg.norm(relerr[i])/np.sqrt(len(relerr[i]))
    maxabserr[idx] = np.amax(error[i])
    idx+=1

# plot lines 
plt.plot(cases[1:-1], rmserr*100, label = "RMS Error") 
plt.plot(cases[1:-1], maxabserr*100, label = "Max. Abs. Error") 
fig = plt.gcf()
ax = plt.gca()
ax.set_title('QR Pivot Interpolation')
ax.set_xlabel("Drum Position (Degrees Rotated Out)")
ax.set_ylabel("POD Reconstruction Error (%)")
plt.legend() 
plt.show()


#%% Increasing Rank


def plot_err_hist(errors):
    fig, ax = plt.subplots(1,1)
    N, bins, patches = ax.hist(errors,bins=100,density=True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
#     fracs = N/N.max()
#     norm = colors.Normalize()
    plt.show()
    
def add_pod_and_plot_error(dataset,coeff="lin"):
    #Build Data Matrix
    X0 = np.zeros([len(y["000"]),len(dataset)])
    idx=0
    for i in dataset:
        k = str(i).zfill(3)
        X0[:,idx] = y[k]
        idx+=1

    # Truncated SVD
    U, S, Vh = np.linalg.svd(X0)
    U_r = U[:,0:len(S)] #TRUNCATE
    a_r = np.matmul(np.diag(S),Vh) #TRUNCATE

    amu = sp.interpolate.interp1d(dataset,a_r,axis=1) #polynomial interpolation

    ytilde = {}
    error = {}
    relerr = {}
    rmserr = np.zeros([len(cases)])
    maxabserr = np.zeros([len(cases)])

    Q,R,P = sp.linalg.qr(U_r,pivoting=True)
    theta = np.linalg.inv(U_r[P,:])
    idx=0
    for c in cases:
        i = str(c).zfill(3)
        if coeff == "lin":
             atilde = amu(c)
        elif coeff == "mp":
             atilde = np.matmul(np.linalg.pinv(U_r),y[i])  
        elif coeff == "qr":
             atilde = np.matmul(theta,y[i][P])
        
        ytilde[i] = np.matmul(U_r,atilde)
        error[i] = ytilde[i] - y[i]
        relerr[i] = np.divide(error[i],y[i])
        rmserr[idx] = np.linalg.norm(relerr[i])/np.sqrt(len(relerr[i]))
        maxabserr[idx] = np.amax(error[i])
        idx+=1


    ## Plot Reconstruction Errors

    # plot lines 
    plt.plot(cases, rmserr*100, label = "RMS Error") 
    plt.plot(cases, maxabserr*100, label = "Max. Abs. Error") 
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(f'{coeff} Interpolation for ROM with {len(dataset)} modes')
    ax.set_xlabel("Drum Position (Degrees Rotated Out)")
    ax.set_ylabel("POD Reconstruction Error (%)")
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(f'POD-Reconstruction-Error-{coeff}Interp-{len(dataset)}modes.png',dpi=300,bbox_inches='tight')
    plt.legend() 
    # plt.show()
    
    maxerr_case = str(cases[maxabserr.argmax()]).zfill(3)
    plot_err_hist(error[maxerr_case]*100)
    plotDict_powers(array2dict(y[maxerr_case]*rel2abs_power,d["000"]))
    plotDict_powers(array2dict(ytilde[maxerr_case]*rel2abs_power,d["000"]))
    plotDict_errors(array2dict(error[maxerr_case]*100,d["000"]))
#     plotDict_errors(inQ=array2dict(error[maxerr_case]*100,d["000"]),
#                     inPQ=array2dict(y[maxerr_case]*rel2abs_power,d["000"]))
    
    return cases[maxabserr.argmax()]

def add_trunc_pod_and_plot_error(dataset,nmodes,coeff="lin"):
    #Build Data Matrix
    if (len(dataset)<nmodes):
        print(f'Truncated POD requesting more modes ({nmodes}) than there are data columns ({len(dataset)}).')
    X0 = np.zeros([len(y["000"]),len(dataset)])
    idx=0
    for i in dataset:
        k = str(i).zfill(3)
        X0[:,idx] = y[k]
        idx+=1

    # Truncated SVD
    U, S, Vh = np.linalg.svd(X0)
    U_r = U[:,0:len(S)]
    a_r = np.matmul(np.diag(S),Vh)

    amu = sp.interpolate.interp1d(dataset,a_r,axis=1) #polynomial interpolation

    ytilde = {}
    error = {}
    relerr = {}
    rmserr = np.zeros([len(cases)])
    maxabserr = np.zeros([len(cases)])

    Q,R,P = sp.linalg.qr(U_r,pivoting=True)
    theta = np.linalg.inv(U_r[P,:])
    idx=0
    for c in cases:
        i = str(c).zfill(3)
        if coeff == "lin":
             atilde = amu(c)
        elif coeff == "mp":
             atilde = np.matmul(np.linalg.pinv(U_r),y[i])  
        elif coeff == "qr":
             atilde = np.matmul(theta,y[i][P])
        
        ytilde[i] = np.matmul(U_r,atilde)
        error[i] = ytilde[i] - y[i]
        relerr[i] = np.divide(error[i],y[i])
        rmserr[idx] = np.linalg.norm(relerr[i])/np.sqrt(len(relerr[i]))
        maxabserr[idx] = np.amax(error[i])
        idx+=1


    ## Plot Reconstruction Errors

    # plot lines 
    plt.plot(cases, rmserr*100, label = "RMS Error") 
    plt.plot(cases, maxabserr*100, label = "Max. Abs. Error") 
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(f'{coeff} Interpolation for ROM with {len(dataset)} modes')
    ax.set_xlabel("Drum Position (Degrees Rotated Out)")
    ax.set_ylabel("POD Reconstruction Error (%)")
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(f'POD-Reconstruction-Error-{coeff}Interp-{len(dataset)}modes.png',dpi=300,bbox_inches='tight')
    plt.legend() 
    # plt.show()
    
    maxerr_case = str(cases[maxabserr.argmax()]).zfill(3)
    plot_err_hist(error[maxerr_case]*100)
    plotDict_powers(array2dict(y[maxerr_case]*rel2abs_power,d["000"]))
    plotDict_powers(array2dict(ytilde[maxerr_case]*rel2abs_power,d["000"]))
    plotDict_errors(array2dict(error[maxerr_case]*100,d["000"]))
#     plotDict_errors(inQ=array2dict(error[maxerr_case]*100,d["000"]),
#                     inPQ=array2dict(y[maxerr_case]*rel2abs_power,d["000"]))
    
    return cases[maxabserr.argmax()]

# In[82]:


add_pod_and_plot_error([0,180],"mp")


# In[83]:


add_pod_and_plot_error([0,85,180],"mp")


# In[84]:


add_pod_and_plot_error([0,60,85,180],"mp")


# In[85]:


add_pod_and_plot_error([0,60,85,105,180],"mp")


# In[24]:


add_pod_and_plot_error([0,30,60,85,105,180],"mp")


# In[ ]:

add_pod_and_plot_error([0,15,30,45,60,75,90,105,180],"mp")

#%% Calculate residuals

# ROM using Moore-Penrose

atilde_mp = {}
ytilde = {}
error = {}
relerr = {}
rmserr = np.zeros([len(cases)-2])
maxabserr = np.zeros([len(cases)-2])

idx=0
cov = {}
for c in cases[1:-1]:
    i = str(c).zfill(3)
    atilde = np.matmul(np.linalg.pinv(U_r),y[i])
    atilde_mp[i] = atilde
    ytilde[i] = np.matmul(U_r,atilde)
    error[i] = ytilde[i] - y[i]
    cov[i] = np.cov(error[i])
    relerr[i] = np.divide(error[i],y[i])
    rmserr[idx] = np.linalg.norm(relerr[i])/np.sqrt(len(relerr[i]))
    maxabserr[idx] = np.amax(error[i])
    idx+=1

#%%

fig, ax = plt.subp

# plot lines 
# plt.plot(cases[1:-1], rmserr*100, label = "RMS Error") 
# plt.plot(cases[1:-1], maxabserr*100, label = "Max. Abs. Error") 
# fig = plt.gcf()
# ax = plt.gca()
# ax.set_title('Moore-Penrose Interpolation')
# ax.set_xlabel("Drum Position (Degrees Rotated Out)")
# ax.set_ylabel("POD Reconstruction Error (%)")
# plt.legend() 
# plt.show()

