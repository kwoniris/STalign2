# import STalign
from STalign import STalign
#import scanpy as sc
import matplotlib.pyplot as plt
##### %%time
import torch
import copy
from torch.nn.functional import grid_sample
#import pims
#import ND2_Reader
#from nd2reader import ND2Reader
#from pims_nd2 import ND2_Reader
import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import torch
import torch.optim as optim  # Add this import statement
import STalign
from STalign import STalign

def rasterizePCA(x,y, G, dx=15, blur=1.5, expand=1.2, draw=0, wavelet_magnitude=False,use_windowing=True):
    """Rasterize a spatial transcriptomics dataset into a density image and perform PCA on it.
    
    Parameters
    ----------
    x : numpy array of length N
        x location of cells
    y : numpy array of length N
        y location of cells
    G : pandas Dataframe of shape (N,M)
        gene expression level of cells for M genes
       
    Returns
    -------
    X : numpy array
        A rasterized image for each gene with the channel along the first axis
    Y : numpy array
        A rasterized image unraveled to 1-D for each gene; data is centered
    W : numpy array
        The eigenvalues for the principal components in descending order
    V : numpy array
        The normalized eigenvectors for the principal components
        V[:, i] corresponds to eigenvalue at W[i]
    Z : numpy array
        X rotated to align with the principal component axes
    nrows : int
        Row dimension of rasterized image
    ncols : int
        Column dimension of rasterized image
    
    Notes
    -----
    Each value/row at the same index in x, y, and G should all correspond to the same cell.
    x[i] <-> y[i] <-> G[i,:]
    
    """
    
    nrows=0
    ncols=0
    
    for i in range(G.shape[1]):
        g = np.array(G.iloc[:,i])
        
        XI,YI,I = rasterize(x,y,g,dx=dx, blur=blur, expand=expand, draw=draw, wavelet_magnitude=wavelet_magnitude,use_windowing=use_windowing)
        
        if(i==0):
            # dimensions
            nrows=YI.size
            ncols=XI.size
            X = np.empty([G.shape[1], nrows, ncols])
            Y = np.empty([G.shape[1], nrows*ncols])
        
        # centers data
        X[i] = np.array(I)
        I_ = I.ravel()
        meanI = np.mean(I_)
        I_ -= meanI
        Y[i] = I_
        
        if(i % 50 == 0):
            print(f"{i} out of {G.shape[1]} genes rasterized.")
        
    S = np.cov(Y) # computes covariance matrix
    W,V = np.linalg.eigh(S) # W = eigenvalues, V = eigenvectors
    
    # reverses order to make it descending by eigenvalue
    W = W[::-1]
    V = V[:,::-1]
    Z = V.T @ Y
    
    return X, Y, W, V, Z, XI, YI



def normalize(arr, t_min=0, t_max=1):
    """Linearly normalizes an array between two specifed values.
    
    Parameters
    ----------
    arr : numpy array
        array to be normalized
    t_min : int or float
        Lower bound of normalization range
    t_max : int or float
        Upper bound of normalization range
    
    Returns
    -------
    norm_arr : numpy array
        1D array with normalized arr values
        
    """
    
    diff = t_max - t_min
    diff_arr = np.max(arr) - np.min(arr)
    min_ = np.min(arr)
    if diff_arr != 0:
        norm_arr = ((arr - min_)/diff_arr * diff) + t_min
    else:
        norm_arr = np.zeros(arr.shape)
    
    return norm_arr


def rasterize(x, y, g=np.ones(1), dx=30, blur=1.5, expand=1.2, draw=0, wavelet_magnitude=False,use_windowing=True):
    ''' Rasterize a spatial transcriptomics dataset into a density image
    
    Paramters
    ---------
    x : numpy array of length N
        x location of cells
    y : numpy array of length N
        y location of cells
    g : numpy array of length N
        RNA count of cells
        If not given, density image is created
    dx : float
        Pixel size to rasterize data (default 30.0, in same units as x and y)
    blur : float or list of floats
        Standard deviation of Gaussian interpolation kernel.  Units are in 
        number of pixels.  Can be aUse a list to do multi scale.
    expand : float
        Factor to expand sampled area beyond cells. Defaults to 1.1.
    draw : int
        If True, draw a figure every draw points return its handle. Defaults to False (0).
    wavelet_magnitude : bool
        If True, take the absolute value of difference between scales for raster images.
        When using this option blur should be sorted from greatest to least.
    
        
    Returns
    -------
    X  : numpy array
        Locations of pixels along the x axis
    Y  : numpy array
        Locations of pixels along the y axis
    M : numpy array
        A rasterized image with len(blur) channels along the first axis
    fig : matplotlib figure handle
        If draw=True, returns a figure handle to the drawn figure.
        
    Raises
    ------    
    Exception 
        If wavelet_magnitude is set to true but blur is not sorted from greatest to least.
        
        
    
    Examples
    --------
    Rasterize a dataset at 30 micron pixel size, with three kernels.
    
    >>> X,Y,M,fig = tools.rasterize(x,y,dx=30.0,blur=[2.0,1.0,0.5],draw=10000)
    
    Rasterize a dataset at 30 micron pixel size, with three kernels, using difference between scales.
    
    >>> X,Y,M,fig = tools.rasterize(x,y,dx=30.0,blur=[2.0,1.0,0.5],draw=10000, wavelet_magnitude=True)
        
        
    '''
    
    # set blur to a list
    if not isinstance(blur,list):
        blur = [blur]
    nb = len(blur)
    blur = np.array(blur)
    n = len(x)
    maxblur = np.max(blur) # for windowing
    
    
    
    if wavelet_magnitude and np.any(blur != np.sort(blur)[::-1]):
        raise Exception('When using wavelet magnitude, blurs must be sorted from greatest to least')
    
    minx = np.min(x)
    maxx = np.max(x)
    miny = np.min(y)
    maxy = np.max(y)
    minx,maxx = (minx+maxx)/2.0 - (maxx-minx)/2.0*expand, (minx+maxx)/2.0 + (maxx-minx)/2.0*expand
    miny,maxy = (miny+maxy)/2.0 - (maxy-miny)/2.0*expand, (miny+maxy)/2.0 + (maxy-miny)/2.0*expand
    X_ = np.arange(minx,maxx,dx)
    Y_ = np.arange(miny,maxy,dx)
    
    X = np.stack(np.meshgrid(X_,Y_)) # note this is xy order, not row col order

    W = np.zeros((X.shape[1],X.shape[2],nb))

    
    if draw: fig,ax = plt.subplots()
    count = 0
    
    g = np.resize(g,x.size)
    if(not (g==1.0).all()):
        g = normalize(g)
    #if np.sum(g) == 0:
    #    print('no gene exp')
    #    return X,Y,W, None
    for x_,y_,g_ in zip(x,y,g):
        # to speed things up I should index
        # to do this I'd have to find row and column indices
        #col = np.round((x_ - X_[0])/dx).astype(int)
        #row = np.round((y_ - X_[1])/dx).astype(int)
        #row0 = np.floor(row-blur*3).astype(int)
        #row1 = np.ceil(row+blur*3).astype(int)        
        #rows = np.arange(row0,row1+1)
        

        # this is incrementing one pixel at a time, it is way way faster, 
        # but doesn't use a kernel
        # I[c_,row,col] += 1.0
        # W[row,col] += 1.0
        if not use_windowing: # legacy version
            k = np.exp( - ( (X[0][...,None] - x_)**2 + (X[1][...,None] - y_)**2 )/(2.0*(dx*blur*2)**2)  )
            if np.sum(k,axis=(0,1)) != 0:
                k /= np.sum(k,axis=(0,1),keepdims=True)    
            k *= g_
            if wavelet_magnitude:
                for i in reversed(range(nb)):
                    if i == 0:
                        continue
                    k[...,i] = k[...,i] - k[...,i-1]

            W += k
        else: # use a small window
            r = int(np.ceil(maxblur*4))
            col = np.round((x_ - X_[0])/dx).astype(int)
            row = np.round((y_ - Y_[0])/dx).astype(int)
            
            row0 = np.floor(row-r).astype(int)
            row1 = np.ceil(row+r).astype(int)                    
            col0 = np.floor(col-r).astype(int)
            col1 = np.ceil(col+r).astype(int)
            
            # we need boundary conditions
            row0 = np.minimum(np.maximum(row0,0),W.shape[0]-1)
            row1 = np.minimum(np.maximum(row1,0),W.shape[0]-1)
            col0 = np.minimum(np.maximum(col0,0),W.shape[1]-1)
            col1 = np.minimum(np.maximum(col1,0),W.shape[1]-1)
            
           
            k =  np.exp( - ( (X[0][row0:row1+1,col0:col1+1,None] - x_)**2 + (X[1][row0:row1+1,col0:col1+1,None] - y_)**2 )/(2.0*(dx*blur*2)**2)  )
            if np.sum(k,axis=(0,1)) != 0:
                k /= np.sum(k,axis=(0,1),keepdims=True)  
            else:
                k=k
            k *= g_
            if wavelet_magnitude:
                for i in reversed(range(nb)):
                    if i == 0:
                        continue
                    k[...,i] = k[...,i] - k[...,i-1]
            W[row0:row1+1,col0:col1+1,:] += k #range of voxels -oka
            
        
            
        

        if draw:
            if not count%draw or count==(x.shape[0]-1):
                print(f'{count} of {x.shape[0]}')

                ax.cla()
                toshow = W-np.min(W,axis=(0,1),keepdims=True)
                toshow = toshow / np.max(toshow,axis=(0,1),keepdims=True)
                
                if nb >= 3:
                    toshow = toshow[...,:3]
                elif nb == 2:
                    toshow = toshow[...,[0,1,0]]
                elif nb == 1:
                    toshow = toshow[...,[0,0,0]]
                
                ax.imshow(np.abs(toshow))
                fig.canvas.draw()

        count += 1
    W = np.abs(W)
    # we will permute so channels are on first axis
    W = W.transpose((-1,0,1))
    extent = (X_[0],X_[-1],Y_[0],Y_[-1])
    
    # rename
    X = X_
    Y = Y_
    if draw:
        output = X,Y,W,fig
    else:
        output = X,Y,W
    return output

def rasterizeCellType(x, y, G, dx=15, blur=1.5, expand=1.2, draw=10000, wavelet_magnitude=False,use_windowing=True):
    nrows=0
    ncols=0
    
    for i in range(G.shape[1]):
        g = np.array(G.iloc[:,i])
        
        XI,YI,I = rasterize(x,y,g=g,dx=15,blur=1.5,expand = 1.2)
        
        if(i==0):
            # dimensions
            nrows=YI.size
            ncols=XI.size
            X = np.empty([G.shape[1], nrows, ncols])
            Y = np.empty([G.shape[1], nrows*ncols])
        
        # centers data
        X[i] = np.array(I)
        I_ = I.ravel()
        meanI = np.mean(I_)
        I_ -= meanI
        Y[i] = I_
        
        if(i % 50 == 0):
            print(f"{i} out of {G.shape[1]} genes rasterized.")

    
    return X, Y, XI, YI

import torch

def to_A(L,T):
    O = torch.tensor([0.,0.,1.],device=L.device,dtype=L.dtype)
    A = torch.cat((torch.cat((L,T[:,None]),1),O[None]))
    return A

def extent_from_x(xJ): #bounds for image to plot - ok
    dJ = [x[1]-x[0] for x in xJ] #step size between pixels along the axes - but is this difference between x and y?
    extentJ = ((xJ[1][0] - dJ[1]/2.0).item(),
               (xJ[1][-1] + dJ[1]/2.0).item(),
               (xJ[0][-1] + dJ[0]/2.0).item(),
               (xJ[0][0] - dJ[0]/2.0).item())
    
    return extentJ

def clip(I):

    Ic = torch.clone(I)
    Ic[Ic<0]=0
    Ic[Ic>1]=1
    return Ic

def interp(x,I,phii,**kwargs):
    # first we have to normalize phii to the range -1,1    
    I = torch.as_tensor(I)
    phii = torch.as_tensor(phii)
    phii = torch.clone(phii)
    for i in range(2):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    # note the above maps to 0,1
    phii *= 2.0
    # to 0 2
    phii -= 1.0
    # done

    # NOTE I should check that I can reproduce identity
    # note that phii must now store x,y,z along last axis
    # is this the right order?
    # I need to put batch (none) along first axis
    # what order do the other 3 need to be in?    
    out = grid_sample(I[None],phii.flip(0).permute((1,2,0))[None],align_corners=True,**kwargs)
    # note align corners true means square voxels with points at their centers
    # post processing, get rid of batch dimension

    return out[0]

def LDDMM(xI,I,xJ,J,Ig, Jg, pointsI=None,pointsJ=None,
          L=None,T=None,A=None,v=None,xv=None,
          a=500.0,p=2.0,expand=2.0,nt=3,
         niter=5000,diffeo_start=0, epL=2e-8, epT=2e-1, epV=2e3,
         sigmaM=1.0,sigmaMg=1.0, sigmaB=2.0,sigmaA=5.0,sigmaR=5e5,sigmaP=2e1,
          device='cpu',dtype=torch.float64, muB=None, muA=None):
 
    if A is not None:
        # if we specify an A
        if L is not None or T is not None:
            raise Exception('If specifying A, you must not specify L or T')
        L = torch.tensor(A[:2,:2],device=device,dtype=dtype,requires_grad=True)
        T = torch.tensor(A[:2,-1],device=device,dtype=dtype,requires_grad=True)   
    else:
        # if we do not specify A                
        if L is None: L = torch.eye(2,device=device,dtype=dtype,requires_grad=True)
        if T is None: T = torch.zeros(2,device=device,dtype=dtype,requires_grad=True)
    L = torch.tensor(L,device=device,dtype=dtype,requires_grad=True)
    T = torch.tensor(T,device=device,dtype=dtype,requires_grad=True)
    # change to torch
    I = torch.tensor(I,device=device,dtype=dtype)                         
    J = torch.tensor(J,device=device,dtype=dtype)
    Jg = torch.tensor(Jg,device=device,dtype=dtype)
    if v is not None and xv is not None:
        v = torch.tensor(v,device=device,dtype=dtype,requires_grad=True)
        xv = [torch.tensor(x,device=device,dtype=dtype) for x in xv]
        XV = torch.stack(torch.meshgrid(xv),-1)
        nt = v.shape[0]        
    elif v is None and xv is None:
        minv = torch.as_tensor([x[0] for x in xI],device=device,dtype=dtype)
        maxv = torch.as_tensor([x[-1] for x in xI],device=device,dtype=dtype)
        minv,maxv = (minv+maxv)*0.5 + 0.5*torch.tensor([-1.0,1.0],device=device,dtype=dtype)[...,None]*(maxv-minv)*expand # WHY DO IT THIS WAY??
        xv = [torch.arange(m,M,a*0.5,device=device,dtype=dtype) for m,M in zip(minv,maxv)] # "a" determines step size of velocity field
        XV = torch.stack(torch.meshgrid(xv),-1) #creating a meshgrid to apply transformation to all points in a grid (creating the 'field' in velocity field)
        v = torch.zeros((nt,XV.shape[0],XV.shape[1],XV.shape[2]),device=device,dtype=dtype,requires_grad=True) # why does XW have a shape of 3 (aren't there only x and y points? 2D)
        #velocity field start out as a zero vector field - how does it get updated? 
    else:
        raise Exception(f'If inputting an initial v, must input both xv and v')
    extentV = extent_from_x(xv) #for plotting purposes
    dv = torch.as_tensor([x[1]-x[0] for x in xv],device=device,dtype=dtype) #step in between each set of points
    
    
 
    fv = [torch.arange(n,device=device,dtype=dtype)/n/d for n,d in zip(XV.shape,dv)] # 5 dimensions of grid of XV/dv - should be similar to XV but not a meshgrid, just points along each axis
    extentF = extent_from_x(fv) #plotting
    FV = torch.stack(torch.meshgrid(fv),-1) # would this not just be the same as XV? print out both and compare 
    LL = (1.0 + 2.0*a**2* torch.sum( (1.0 - torch.cos(2.0*np.pi*FV*dv))/dv**2 ,-1))**(p*2.0) #smoothing kernel? what is this?

    K = 1.0/LL
    #fig,ax = plt.subplots()
    #ax.imshow(K,vmin=0.0,vmax=0.1,extent=extentF)
    
    #fig,ax = plt.subplots()
    #ax.imshow(K[0].cpu())
    DV = torch.prod(dv)
    Ki = torch.fft.ifftn(K).real
    fig,ax = plt.subplots()
    ax.imshow(Ki.clone().detach().cpu().numpy(),vmin=0.0,extent=extentV)
    ax.set_title('smoothing kernel')
    fig.canvas.draw()


    # nt = 3
    


    WM = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.5
    WB = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.4
    WA = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.1
    if pointsI is None and pointsJ is None:
        pointsI = torch.zeros((0,2),device=J.device,dtype=J.dtype)
        pointsJ = torch.zeros((0,2),device=J.device,dtype=J.dtype) 
    elif (pointsI is None and pointsJ is not None) or (pointsJ is None and pointsI is not None):
        raise Exception('Must specify corresponding sets of points or none at all')
    else:
        pointsI = torch.tensor(pointsI,device=J.device,dtype=J.dtype)
        pointsJ = torch.tensor(pointsJ,device=J.device,dtype=J.dtype)
    
    
    xI = [torch.tensor(x,device=device,dtype=dtype) for x in xI]
    xJ = [torch.tensor(x,device=device,dtype=dtype) for x in xJ]
    XI = torch.stack(torch.meshgrid(*xI,indexing='ij'),-1)
    XJ = torch.stack(torch.meshgrid(*xJ,indexing='ij'),-1)
    dJ = [x[1]-x[0] for x in xJ]
    extentJ = (xJ[1][0].item()-dJ[1].item()/2.0,
          xJ[1][-1].item()+dJ[1].item()/2.0,
          xJ[0][-1].item()+dJ[0].item()/2.0,
          xJ[0][0].item()-dJ[0].item()/2.0)
    

    if muA is None:
        estimate_muA = True
    else:
        estimate_muA = False
    if muB is None:
        estimate_muB = True
    else:
        estimate_muB = False
    
    fig,ax = plt.subplots(2,3)
    ax = ax.ravel()
    figE,axE = plt.subplots(1,5)
    Esave = []

    try:
        L.grad.zero_()
    except:
        pass
    try:
        T.grad.zero_()
    except:
        pass


    
    for it in range(niter):
        # make A
        A = to_A(L,T)
        # Ai
        Ai = torch.linalg.inv(A)
        # transform sample points
        Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]    
        # now diffeo, not semilagrange here
        for t in range(nt-1,-1,-1):
            Xs = Xs + interp(xv,-v[t].permute(2,0,1),Xs.permute(2,0,1)).permute(1,2,0)/nt
        # and points
        pointsIt = torch.clone(pointsI)
        if pointsIt.shape[0] >0:
            for t in range(nt):            
                pointsIt += interp(xv,v[t].permute(2,0,1),pointsIt.T[...,None])[...,0].T/nt
            pointsIt = (A[:2,:2]@pointsIt.T + A[:2,-1][...,None]).T
        #print(Xs.shape)
        # transform image
        #AI = interp(xI,I.swapaxes(1,2),Xs.permute(2,1,0),padding_mode="border")
        AI = interp(xI,I,Xs.permute(2,0,1),padding_mode="border")
        AIg = interp(xI,Ig,Xs.permute(2,0,1),padding_mode="border")
        #AI = AI.permute(0,2,1)
        #print(AI.shape)
        #print(J.shape)
        if it == 0:
            AI_orig = AIg.clone().detach()
        # objective function
        EMg = torch.sum((AIg - Jg)**2*WM)/2.0/sigmaMg**2
        EM = torch.sum((AI - J)**2*WM)/2.0/sigmaM**2
        ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v,dim=(1,2)))**2,dim=(0,-1))*LL)*DV/2.0/v.shape[1]/v.shape[2]/sigmaR**2
        # ERROR FUNCTION!
        E = EM + ER + EMg
        tosave = [E.item(), EM.item(), ER.item(), EMg.item()]
        if pointsIt.shape[0]>0:
            EP = torch.sum((pointsIt - pointsJ)**2)/2.0/sigmaP**2
            E += EP
            tosave.append(EP.item())
        
        Esave.append( tosave )
        # gradient update
        E.backward()
        with torch.no_grad():            
            L -= (epL/(1.0 + (it>=diffeo_start)*9))*L.grad
            T -= (epT/(1.0 + (it>=diffeo_start)*9))*T.grad

            L.grad.zero_()
            T.grad.zero_()
            

            # v grad
            vgrad = v.grad
            # smooth it
            vgrad = torch.fft.ifftn(torch.fft.fftn(vgrad,dim=(1,2))*K[...,None],dim=(1,2)).real
            if it >= diffeo_start:
                v -= vgrad*epV
            v.grad.zero_()


        # update weights
        if not it%5:
            with torch.no_grad():
                # M step for these params
                if estimate_muA:
                    muA = torch.sum(WA*J,dim=(-1,-2))/torch.sum(WA)
                if estimate_muB:
                    muB = torch.sum(WB*J,dim=(-1,-2))/torch.sum(WB)
                #if it <= 200:
                #    muA = torch.tensor([0.75,0.77,0.79],device=J.device,dtype=J.dtype)
                #    muB = torch.ones(J.shape[0],device=J.device,dtype=J.dtype)*0.9




        # draw
        if not it%10:
            AI_plt = AIg
            J_plt = Jg
            #print(AI.shape)
            if AI.shape[0]>=1:
                AI_plt = torch.mean(AIg, 0)
                AI_plt = torch.unsqueeze(AI_plt, 0)
                AI_orig = torch.mean(AI_orig, 0)
                AI_orig = torch.unsqueeze(AI_orig, 0)
                J_plt = torch.mean(Jg, 0)
                J_plt = torch.unsqueeze(J_plt, 0)
            #print(AI_plt.shape)
            ax[0].cla()
            ax[0].imshow(   ((AI_orig-torch.amin(AI_orig,(1,2))[...,None,None])/(torch.amax(AI_orig,(1,2))-torch.amin(AI_orig,(1,2)))[...,None,None]).permute(1,2,0).clone().detach().cpu(),extent=extentJ)
            ax[0].scatter(pointsIt[:,1].clone().detach().cpu(),pointsIt[:,0].clone().detach().cpu())
            ax[0].set_title('space tformed source')
            
            ax[1].cla()
            ax[1].imshow(   ((AI_plt-torch.amin(AI_plt,(1,2))[...,None,None])/(torch.amax(AI_plt,(1,2))-torch.amin(AI_plt,(1,2)))[...,None,None]).permute(1,2,0).clone().detach().cpu(),extent=extentJ)
            ax[1].scatter(pointsIt[:,1].clone().detach().cpu(),pointsIt[:,0].clone().detach().cpu())
            ax[1].set_title('space tformed source')
            
            ax[4].cla()
            ax[4].imshow(clip( (AI_plt - J_plt)/(torch.max(Jg).item())*3.0  ).permute(1,2,0).clone().detach().cpu()*0.5+0.5,extent=extentJ)
            ax[4].scatter(pointsIt[:,1].clone().detach().cpu(),pointsIt[:,0].clone().detach().cpu())
            ax[4].scatter(pointsJ[:,1].clone().detach().cpu(),pointsJ[:,0].clone().detach().cpu())
            ax[4].set_title('Error')

            ax[2].cla()
            ax[2].imshow(J_plt.permute(1,2,0).cpu()/torch.max(Jg).item(),extent=extentJ)
            ax[2].scatter(pointsJ[:,1].clone().detach().cpu(),pointsJ[:,0].clone().detach().cpu())
            ax[2].set_title('Target')


            toshow = v[0].clone().detach().cpu()
            toshow /= torch.max(torch.abs(toshow))
            toshow = toshow*0.5+0.5
            toshow = torch.cat((toshow,torch.zeros_like(toshow[...,0][...,None])),-1)   
            ax[3].cla()
            ax[3].imshow(clip(toshow),extent=extentV)
            ax[3].set_title('velocity')
            
            axE[0].cla()
            axE[0].plot([e[0] for e in Esave])
            axE[0].legend(['E'])
            axE[0].set_yscale('log')
            axE[1].cla()
            axE[1].plot([e[1] for e in Esave])
            axE[1].legend(['EM'])
            axE[1].set_yscale('log')
            axE[2].cla()
            axE[2].plot([e[2] for e in Esave])
            axE[2].legend(['ER'])
            axE[2].set_yscale('log')
            axE[3].cla()
            axE[3].plot([e[3] for e in Esave])
            axE[3].legend(['EMg'])
            axE[3].set_yscale('log')
            axE[4].cla()
            #if Esave[0][4].exists():
            #    axE[4].plot([e[4] for e in Esave])
            #    axE[4].legend(['EP'])
            #    axE[4].set_yscale('log')

            fig.canvas.draw()
            figE.canvas.draw()
            
    return {
        'A': A.clone().detach(), 
        'v': v.clone().detach(), 
        'xv': xv, 
        'WM': WM.clone().detach(),
        'WB': WB.clone().detach(),
        'WA': WA.clone().detach(),
        "AI_plt": AI_plt.clone().detach(),
        "AI": AI.clone().detach(),
        "J": J.clone().detach()
    }


def rasterizePCAmany(x, G, dx=10, blur=1.5, expand=1.2, draw=0, wavelet_magnitude=False, use_windowing=True):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    sample_num = len(x)
    X = []
    Y = []
    I = []
    nrows = []
    ncols = []
    
    for i in range(G[0].shape[1]):
        I = []
        for j in range(sample_num):
            g = G[j].iloc[:, i]
            X1, Y1, I1 = rasterize(x[j][0], x[j][1], g, dx=dx, blur=blur, expand=expand, draw=draw, wavelet_magnitude=wavelet_magnitude, use_windowing=use_windowing)
            I.append(I1)
            if i == 0:
                nrows.append(Y1.size)
                ncols.append(X1.size)
                X.append(X1)
                Y.append(Y1)
        
        if i == 0:
            num_cols_all = [nrows[k] * ncols[k] for k in range(len(ncols))]
            num_cols_all_sum = np.sum(np.array(num_cols_all))
            mat = np.empty([G[0].shape[1], num_cols_all_sum])
        
        IJ_ = np.concatenate([I1.ravel() for I1 in I])
        meanIJ = np.mean(IJ_)
        IJ_ -= meanIJ
        mat[i] = IJ_

        if (i % 10 == 0):
            print(f"{i} out of {G[0].shape[1]} genes rasterized.")
    
    df = pd.DataFrame(mat)
    scaler = StandardScaler()
    scaled_df = df.copy()
    scaled_df = pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)
    
    pca = PCA(n_components=len(mat))
    pca_fit = pca.fit(scaled_df)
    
    PC_values = np.arange(pca.n_components_)
    PC_values1 = PC_values + 1
    plt.plot(PC_values1, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.show()
    
    sig_PC = PC_values[np.cumsum(pca.explained_variance_ratio_) > 0.7]
    
    if len(sig_PC) > 1:
        num_PC = sig_PC[0]
    else:
        num_PC = sig_PC
    
    Z = pca.components_
    PC_img = []
    
    print(f"Number of samples: {sample_num}")
    print(f"Number of genes: {G[0].shape[1]}")
    print(f"Number of columns in each sample: {num_cols_all}")

    for z in range(sample_num):
        print(f"Processing sample {z + 1}/{sample_num}")
        if z == 0:
            PC_img_temp = Z[:, :num_cols_all[z]]
        elif z == sample_num - 1:
            print("last sample")
            PC_img_temp = Z[:, np.sum(num_cols_all[:z]):]
        else:
            PC_img_temp = Z[:, np.sum(num_cols_all[:z]):np.sum(num_cols_all[:(z + 1)])]
        
        try:
            PC_img_temp = PC_img_temp.reshape(G[0].shape[1], nrows[z], ncols[z])
        except ValueError as e:
            print(f"Error reshaping PC_img_temp for sample {z + 1}: {e}")
            print(f"Expected shape: ({G[0].shape[1]}, {nrows[z]}, {ncols[z]})")
            print(f"Actual shape before reshaping: {PC_img_temp.shape}")
            raise
        
        print(f"Shape of PC_img_temp for sample {z + 1}: {PC_img_temp.shape}")
        PC_img.append(PC_img_temp)
    
    return X, Y, pca_fit, pca, PC_values, num_PC, PC_img


def LDDMM_LBFGS(xI,I,xJ,J,Ig, Jg, pointsI=None,pointsJ=None,
          L=None,T=None,A=None,v=None,xv=None,
          a=500.0,p=2.0,expand=2.0,nt=3,
         niter=5000,diffeo_start=0, epL=2e-8, epT=2e-1, epV=2e3,
         sigmaM=1.0,sigmaMg=1.0, sigmaB=2.0,sigmaA=5.0,sigmaR=5e5,sigmaP=2e1,
          device='cpu',dtype=torch.float64, muB=None, muA=None):
 
    if A is not None:
        # if we specify an A
        if L is not None or T is not None:
            raise Exception('If specifying A, you must not specify L or T')
        L = torch.tensor(A[:2,:2],device=device,dtype=dtype,requires_grad=True)
        T = torch.tensor(A[:2,-1],device=device,dtype=dtype,requires_grad=True)   
    else:
        # if we do not specify A                
        if L is None: L = torch.eye(2,device=device,dtype=dtype,requires_grad=True)
        if T is None: T = torch.zeros(2,device=device,dtype=dtype,requires_grad=True)
    L = torch.tensor(L,device=device,dtype=dtype,requires_grad=True)
    T = torch.tensor(T,device=device,dtype=dtype,requires_grad=True)
    # change to torch
    I = torch.tensor(I,device=device,dtype=dtype)                         
    J = torch.tensor(J,device=device,dtype=dtype)
    Jg = torch.tensor(Jg,device=device,dtype=dtype)
    if v is not None and xv is not None:
        v = torch.tensor(v,device=device,dtype=dtype,requires_grad=True)
        xv = [torch.tensor(x,device=device,dtype=dtype) for x in xv]
        XV = torch.stack(torch.meshgrid(xv),-1)
        nt = v.shape[0]        
    elif v is None and xv is None:
        minv = torch.as_tensor([x[0] for x in xI],device=device,dtype=dtype)
        maxv = torch.as_tensor([x[-1] for x in xI],device=device,dtype=dtype)
        minv,maxv = (minv+maxv)*0.5 + 0.5*torch.tensor([-1.0,1.0],device=device,dtype=dtype)[...,None]*(maxv-minv)*expand # WHY DO IT THIS WAY??
        xv = [torch.arange(m,M,a*0.5,device=device,dtype=dtype) for m,M in zip(minv,maxv)] # "a" determines step size of velocity field
        XV = torch.stack(torch.meshgrid(xv),-1) #creating a meshgrid to apply transformation to all points in a grid (creating the 'field' in velocity field)
        v = torch.zeros((nt,XV.shape[0],XV.shape[1],XV.shape[2]),device=device,dtype=dtype,requires_grad=True) # why does XW have a shape of 3 (aren't there only x and y points? 2D)
        #velocity field start out as a zero vector field - how does it get updated? 
    else:
        raise Exception(f'If inputting an initial v, must input both xv and v')
    extentV = extent_from_x(xv) #for plotting purposes
    dv = torch.as_tensor([x[1]-x[0] for x in xv],device=device,dtype=dtype) #step in between each set of points
    
    
 
    fv = [torch.arange(n,device=device,dtype=dtype)/n/d for n,d in zip(XV.shape,dv)] # 5 dimensions of grid of XV/dv - should be similar to XV but not a meshgrid, just points along each axis
    extentF = extent_from_x(fv) #plotting
    FV = torch.stack(torch.meshgrid(fv),-1) # would this not just be the same as XV? print out both and compare 
    LL = (1.0 + 2.0*a**2* torch.sum( (1.0 - torch.cos(2.0*np.pi*FV*dv))/dv**2 ,-1))**(p*2.0) #smoothing kernel? what is this?

    K = 1.0/LL
    #fig,ax = plt.subplots()
    #ax.imshow(K,vmin=0.0,vmax=0.1,extent=extentF)
    
    #fig,ax = plt.subplots()
    #ax.imshow(K[0].cpu())
    DV = torch.prod(dv)
    Ki = torch.fft.ifftn(K).real
    fig,ax = plt.subplots()
    ax.imshow(Ki.clone().detach().cpu().numpy(),vmin=0.0,extent=extentV)
    ax.set_title('smoothing kernel')
    fig.canvas.draw()
    WM = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.5
    WB = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.4
    WA = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.1
    if pointsI is None and pointsJ is None:
        pointsI = torch.zeros((0,2),device=J.device,dtype=J.dtype)
        pointsJ = torch.zeros((0,2),device=J.device,dtype=J.dtype) 
    elif (pointsI is None and pointsJ is not None) or (pointsJ is None and pointsI is not None):
        raise Exception('Must specify corresponding sets of points or none at all')
    else:
        pointsI = torch.tensor(pointsI,device=J.device,dtype=J.dtype)
        pointsJ = torch.tensor(pointsJ,device=J.device,dtype=J.dtype)
    
    
    xI = [torch.tensor(x,device=device,dtype=dtype) for x in xI]
    xJ = [torch.tensor(x,device=device,dtype=dtype) for x in xJ]
    XI = torch.stack(torch.meshgrid(*xI,indexing='ij'),-1)
    XJ = torch.stack(torch.meshgrid(*xJ,indexing='ij'),-1)
    dJ = [x[1]-x[0] for x in xJ]
    extentJ = (xJ[1][0].item()-dJ[1].item()/2.0,
          xJ[1][-1].item()+dJ[1].item()/2.0,
          xJ[0][-1].item()+dJ[0].item()/2.0,
          xJ[0][0].item()-dJ[0].item()/2.0)
    

    if muA is None:
        estimate_muA = True
    else:
        estimate_muA = False
    if muB is None:
        estimate_muB = True
    else:
        estimate_muB = False
    
    fig,ax = plt.subplots(2,3)
    ax = ax.ravel()
    figE,axE = plt.subplots(1,5)
    Esave = []

    try:
        L.grad.zero_()
    except:
        pass
    try:
        T.grad.zero_()
    except:
        pass

    optimizer = torch.optim.LBFGS([L, T,v], lr=1.0, max_iter=niter, line_search_fn="strong_wolfe")
    def closure():
        nonlocal L, T, v
        optimizer.zero_grad()
        E = compute_objective(L, T, v, xv, XJ, AI, J, AIg, Jg, WM, sigmaM, sigmaMg, sigmaR, XV, DV, LL, a, p, nt, device)
        E.backward(retain_graph = True)
        return E
    
    for it in range(niter):
        # make A
        A = to_A(L,T)
        # Ai
        Ai = torch.linalg.inv(A)
        # transform sample points
        Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]    
        # now diffeo, not semilagrange here
        for t in range(nt-1,-1,-1):
            Xs = Xs + interp(xv,-v[t].permute(2,0,1),Xs.permute(2,0,1)).permute(1,2,0)/nt
        # and points
        pointsIt = torch.clone(pointsI)
        if pointsIt.shape[0] >0:
            for t in range(nt):            
                pointsIt += interp(xv,v[t].permute(2,0,1),pointsIt.T[...,None])[...,0].T/nt
            pointsIt = (A[:2,:2]@pointsIt.T + A[:2,-1][...,None]).T
        #print(Xs.shape)
        # transform image
        #AI = interp(xI,I.swapaxes(1,2),Xs.permute(2,1,0),padding_mode="border")
        AI = interp(xI,I,Xs.permute(2,0,1),padding_mode="border")
        AIg = interp(xI,Ig,Xs.permute(2,0,1),padding_mode="border")
        #AI = AI.permute(0,2,1)
        #print(AI.shape)
        #print(J.shape)
        if it == 0:
            AI_orig = AIg.clone().detach()
        # objective function
        EMg = torch.sum((AIg - Jg)**2*WM)/2.0/sigmaMg**2
        EM = torch.sum((AI - J)**2*WM)/2.0/sigmaM**2
        ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v,dim=(1,2)))**2,dim=(0,-1))*LL)*DV/2.0/v.shape[1]/v.shape[2]/sigmaR**2
        E = EM + ER + EMg
        optimizer.step(closure)
        tosave = [E.item(), EM.item(), ER.item(), EMg.item()]
        if pointsIt.shape[0]>0:
            EP = torch.sum((pointsIt - pointsJ)**2)/2.0/sigmaP**2
            E += EP
            tosave.append(EP.item())
        
        Esave.append( tosave )
        # gradient update
        #E.backward()
        with torch.no_grad():            
            L -= (epL/(1.0 + (it>=diffeo_start)*9))*L.grad
            T -= (epT/(1.0 + (it>=diffeo_start)*9))*T.grad

            L.grad.zero_()
            T.grad.zero_()
            

            # v grad
            vgrad = v.grad
            # smooth it
            vgrad = torch.fft.ifftn(torch.fft.fftn(vgrad,dim=(1,2))*K[...,None],dim=(1,2)).real
            if it >= diffeo_start:
                v -= vgrad*epV
            v.grad.zero_()


        # update weights
        if not it%5:
            with torch.no_grad():
                # M step for these params
                if estimate_muA:
                    muA = torch.sum(WA*J,dim=(-1,-2))/torch.sum(WA)
                if estimate_muB:
                    muB = torch.sum(WB*J,dim=(-1,-2))/torch.sum(WB)
                #if it <= 200:
                #    muA = torch.tensor([0.75,0.77,0.79],device=J.device,dtype=J.dtype)
                #    muB = torch.ones(J.shape[0],device=J.device,dtype=J.dtype)*0.9


        if not it%10:
            AI_plt = AIg
            J_plt = Jg
            #print(AI.shape)
            if AI.shape[0]>=1:
                AI_plt = torch.mean(AIg, 0)
                AI_plt = torch.unsqueeze(AI_plt, 0)
                AI_orig = torch.mean(AI_orig, 0)
                AI_orig = torch.unsqueeze(AI_orig, 0)
                J_plt = torch.mean(Jg, 0)
                J_plt = torch.unsqueeze(J_plt, 0)
            #print(AI_plt.shape)
            ax[0].cla()
            ax[0].imshow(   ((AI_orig-torch.amin(AI_orig,(1,2))[...,None,None])/(torch.amax(AI_orig,(1,2))-torch.amin(AI_orig,(1,2)))[...,None,None]).permute(1,2,0).clone().detach().cpu(),extent=extentJ)
            ax[0].scatter(pointsIt[:,1].clone().detach().cpu(),pointsIt[:,0].clone().detach().cpu())
            ax[0].set_title('space tformed source')
            
            ax[1].cla()
            ax[1].imshow(   ((AI_plt-torch.amin(AI_plt,(1,2))[...,None,None])/(torch.amax(AI_plt,(1,2))-torch.amin(AI_plt,(1,2)))[...,None,None]).permute(1,2,0).clone().detach().cpu(),extent=extentJ)
            ax[1].scatter(pointsIt[:,1].clone().detach().cpu(),pointsIt[:,0].clone().detach().cpu())
            ax[1].set_title('space tformed source')
            
            ax[4].cla()
            ax[4].imshow(clip( (AI_plt - J_plt)/(torch.max(Jg).item())*3.0  ).permute(1,2,0).clone().detach().cpu()*0.5+0.5,extent=extentJ)
            ax[4].scatter(pointsIt[:,1].clone().detach().cpu(),pointsIt[:,0].clone().detach().cpu())
            ax[4].scatter(pointsJ[:,1].clone().detach().cpu(),pointsJ[:,0].clone().detach().cpu())
            ax[4].set_title('Error')

            ax[2].cla()
            ax[2].imshow(J_plt.permute(1,2,0).cpu()/torch.max(Jg).item(),extent=extentJ)
            ax[2].scatter(pointsJ[:,1].clone().detach().cpu(),pointsJ[:,0].clone().detach().cpu())
            ax[2].set_title('Target')


            toshow = v[0].clone().detach().cpu()
            toshow /= torch.max(torch.abs(toshow))
            toshow = toshow*0.5+0.5
            toshow = torch.cat((toshow,torch.zeros_like(toshow[...,0][...,None])),-1)   
            ax[3].cla()
            ax[3].imshow(clip(toshow),extent=extentV)
            ax[3].set_title('velocity')
            
            axE[0].cla()
            axE[0].plot([e[0] for e in Esave])
            axE[0].legend(['E'])
            axE[0].set_yscale('log')
            axE[1].cla()
            axE[1].plot([e[1] for e in Esave])
            axE[1].legend(['EM'])
            axE[1].set_yscale('log')
            axE[2].cla()
            axE[2].plot([e[2] for e in Esave])
            axE[2].legend(['ER'])
            axE[2].set_yscale('log')
            axE[3].cla()
            axE[3].plot([e[3] for e in Esave])
            axE[3].legend(['EMg'])
            axE[3].set_yscale('log')
            axE[4].cla()
            axE[4].plot([e[3] for e in Esave])
            axE[4].legend(['EMg'])
            axE[4].set_yscale('log')

            fig.canvas.draw()
            figE.canvas.draw()
            
    return {
        'A': A.clone().detach(), 
        'v': v.clone().detach(), 
        'xv': xv, 
        'WM': WM.clone().detach(),
        'WB': WB.clone().detach(),
        'WA': WA.clone().detach(),
        "AI_plt": AI_plt.clone().detach(),
        "AI": AI.clone().detach(),
        "J": J.clone().detach()
    }


def compute_objective(L, T, v, xv, XJ, AI, J, AIg, Jg, WM, sigmaM, sigmaMg, sigmaR, XV, DV, LL, a, p, nt, device):
    # ... (existing code)

    # Compute the objective function
    EMg = torch.sum((AIg - Jg)**2 * WM) / 2.0 / sigmaMg**2
    EM = torch.sum((AI - J)**2 * WM) / 2.0 / sigmaM**2
    ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v, dim=(1, 2)))**2, dim=(0, -1)) * LL) * DV / 2.0 / v.shape[1] / v.shape[2] / sigmaR**2
    E = EM + ER + EMg
    # ... (existing code)

    return E


def PC_autocorrelation(PC_mat, num_PC, threshold = 700):
    PC_values = np.array(range(len(PC_mat)))
    correlation = []
    for pc in range(num_PC):
        f_image = np.fft.fft2(PC_mat[pc])
        autocorrelation = np.fft.ifft2(f_image*np.conj(f_image)).real
        correlation.append(np.sum(autocorrelation))
    print(correlation)
    #PC_values = PC_values-1
    high_corr_PC = PC_values[:num_PC][np.array(correlation)>700]
    low_corr_PC = PC_values[:num_PC][np.array(correlation)<700]

    return high_corr_PC, low_corr_PC


def process_data_subs(genes, data_subs):
    dfs = []
    indices = []
    raw_coords = []

    for data_sub in data_subs:
        # Create DataFrame
        df = pd.DataFrame(data_sub.X, columns=data_sub.var.index)
        dfs.append(df)

        # Get indices
        idx = [index for index, value in enumerate(data_sub.var.index) if value in genes]
        indices.append(idx)
        
        # Get raw coordinates
        raw_coords.append([data_sub.obs['raw_x'], data_sub.obs['raw_y']])

    # Filter DataFrames
    filtered_dfs = [df.iloc[:, idx] for df, idx in zip(dfs, indices)]

    # Reindex columns
    reference_columns = filtered_dfs[0].columns
    reindexed_dfs = [df.reindex(columns=reference_columns) for df in filtered_dfs]

    return reindexed_dfs, raw_coords


def process_raw_coords(raw_coords, Gs, dx=10, blur=1.5, expand=1.2, draw=10000):
    Xgs, Ys, XIs, YIs = [], [], [], []

    for coords, G in zip(raw_coords, Gs):
        x, y = coords
        Xg, Y, XI, YI = rasterizeCellType(x, y, G, dx=dx, blur=blur, expand=expand, draw=True)
        Xgs.append(Xg)
        Ys.append(Y)
        XIs.append(XI)
        YIs.append(YI)

    return Xgs, Ys, XIs, YIs


def normalize_pc_img(PC_img, subset_indices):
    # Extract the subset
    PC_img_sub = [PC_img[i][subset_indices] for i in range(len(PC_img))]
    
    # Normalize the subset
    for i in range(len(PC_img_sub)):
        for j in range(len(PC_img_sub[0])):
            temp = (PC_img_sub[i][j] - np.min(PC_img_sub[i][j])) / (np.max(PC_img_sub[i][j]) - np.min(PC_img_sub[i][j]))
            PC_img_sub[i][j] = temp

    return PC_img_sub


def plot_pc_img(PC_img_sub, indices_to_plot):
    fig, ax = plt.subplots(len(indices_to_plot), len(PC_img_sub), figsize=(15, 15))
    
    for row, pc in enumerate(indices_to_plot):
        for col in range(len(PC_img_sub)):
            ax[row, col].imshow(PC_img_sub[col][pc])
            ax[row, col].set_title(f'Slice {col+1}, PC {pc+1}')
            ax[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def LDDMM_Adam(xI,I,xJ,J,Ig, Jg, pointsI=None,pointsJ=None,
          L=None,T=None,A=None,v=None,xv=None,
          a=500.0,p=2.0,expand=2.0,nt=3,
         niter=5000,diffeo_start=0, epL=2e-8, epT=2e-1, epV=2e3,
         sigmaM=1.0,sigmaMg=1.0, sigmaB=2.0,sigmaA=5.0,sigmaR=5e5,sigmaP=2e1,
         early_stopping=True, patience=10, min_delta=1e-4, gradient_threshold=1e-6,
         device='cpu',dtype=torch.float64, muB=None, muA=None):
 
    if A is not None:
        # if we specify an A
        if L is not None or T is not None:
            raise Exception('If specifying A, you must not specify L or T')
        L = torch.tensor(A[:2,:2],device=device,dtype=dtype,requires_grad=True)
        T = torch.tensor(A[:2,-1],device=device,dtype=dtype,requires_grad=True)   
    else:
        # if we do not specify A                
        if L is None: L = torch.eye(2,device=device,dtype=dtype,requires_grad=True)
        if T is None: T = torch.zeros(2,device=device,dtype=dtype,requires_grad=True)
    L = torch.tensor(L,device=device,dtype=dtype,requires_grad=True)
    T = torch.tensor(T,device=device,dtype=dtype,requires_grad=True)
    # change to torch
    I = torch.tensor(I,device=device,dtype=dtype)                         
    J = torch.tensor(J,device=device,dtype=dtype)
    Jg = torch.tensor(Jg,device=device,dtype=dtype)
    if v is not None and xv is not None:
        v = torch.tensor(v,device=device,dtype=dtype,requires_grad=True)
        xv = [torch.tensor(x,device=device,dtype=dtype) for x in xv]
        XV = torch.stack(torch.meshgrid(xv),-1)
        nt = v.shape[0]        
    elif v is None and xv is None:
        minv = torch.as_tensor([x[0] for x in xI],device=device,dtype=dtype)
        maxv = torch.as_tensor([x[-1] for x in xI],device=device,dtype=dtype)
        minv,maxv = (minv+maxv)*0.5 + 0.5*torch.tensor([-1.0,1.0],device=device,dtype=dtype)[...,None]*(maxv-minv)*expand # WHY DO IT THIS WAY??
        xv = [torch.arange(m,M,a*0.5,device=device,dtype=dtype) for m,M in zip(minv,maxv)] # "a" determines step size of velocity field
        XV = torch.stack(torch.meshgrid(xv),-1) #creating a meshgrid to apply transformation to all points in a grid (creating the 'field' in velocity field)
        v = torch.zeros((nt,XV.shape[0],XV.shape[1],XV.shape[2]),device=device,dtype=dtype,requires_grad=True) # why does XW have a shape of 3 (aren't there only x and y points? 2D)
        #velocity field start out as a zero vector field - how does it get updated? 
    else:
        raise Exception(f'If inputting an initial v, must input both xv and v')
    extentV = extent_from_x(xv) #for plotting purposes
    dv = torch.as_tensor([x[1]-x[0] for x in xv],device=device,dtype=dtype) #step in between each set of points
    
    
 
    fv = [torch.arange(n,device=device,dtype=dtype)/n/d for n,d in zip(XV.shape,dv)] # 5 dimensions of grid of XV/dv - should be similar to XV but not a meshgrid, just points along each axis
    extentF = extent_from_x(fv) #plotting
    FV = torch.stack(torch.meshgrid(fv),-1) # would this not just be the same as XV? print out both and compare 
    LL = (1.0 + 2.0*a**2* torch.sum( (1.0 - torch.cos(2.0*np.pi*FV*dv))/dv**2 ,-1))**(p*2.0) #smoothing kernel? what is this?

    K = 1.0/LL
    #fig,ax = plt.subplots()
    #ax.imshow(K,vmin=0.0,vmax=0.1,extent=extentF)
    
    #fig,ax = plt.subplots()
    #ax.imshow(K[0].cpu())
    DV = torch.prod(dv)
    Ki = torch.fft.ifftn(K).real
    fig,ax = plt.subplots()
    ax.imshow(Ki.clone().detach().cpu().numpy(),vmin=0.0,extent=extentV)
    ax.set_title('smoothing kernel')
    fig.canvas.draw()


    # nt = 3
    


    WM = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.5
    WB = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.4
    WA = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.1
    if pointsI is None and pointsJ is None:
        pointsI = torch.zeros((0,2),device=J.device,dtype=J.dtype)
        pointsJ = torch.zeros((0,2),device=J.device,dtype=J.dtype) 
    elif (pointsI is None and pointsJ is not None) or (pointsJ is None and pointsI is not None):
        raise Exception('Must specify corresponding sets of points or none at all')
    else:
        pointsI = torch.tensor(pointsI,device=J.device,dtype=J.dtype)
        pointsJ = torch.tensor(pointsJ,device=J.device,dtype=J.dtype)
    
    
    xI = [torch.tensor(x,device=device,dtype=dtype) for x in xI]
    xJ = [torch.tensor(x,device=device,dtype=dtype) for x in xJ]
    XI = torch.stack(torch.meshgrid(*xI,indexing='ij'),-1)
    XJ = torch.stack(torch.meshgrid(*xJ,indexing='ij'),-1)
    dJ = [x[1]-x[0] for x in xJ]
    extentJ = (xJ[1][0].item()-dJ[1].item()/2.0,
          xJ[1][-1].item()+dJ[1].item()/2.0,
          xJ[0][-1].item()+dJ[0].item()/2.0,
          xJ[0][0].item()-dJ[0].item()/2.0)
    

    if muA is None:
        estimate_muA = True
    else:
        estimate_muA = False
    if muB is None:
        estimate_muB = True
    else:
        estimate_muB = False
    
    fig,ax = plt.subplots(2,3)
    ax = ax.ravel()
    figE,axE = plt.subplots(1,5)
    Esave = []

    # Initialize the Adam optimizer
    optimizer = optim.Adam([L, T, v], lr=1e-3)

    best_loss = float('inf')
    epochs_no_improve = 0

    for it in range(niter):
        optimizer.zero_grad()

        # make A
        A = to_A(L,T)
        # Ai
        Ai = torch.linalg.inv(A)
        # transform sample points
        Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]    
        # now diffeo, not semilagrange here
        for t in range(nt-1,-1,-1):
            Xs = Xs + interp(xv,-v[t].permute(2,0,1),Xs.permute(2,0,1)).permute(1,2,0)/nt
        # and points
        pointsIt = torch.clone(pointsI)
        if pointsIt.shape[0] >0:
            for t in range(nt):            
                pointsIt += interp(xv,v[t].permute(2,0,1),pointsIt.T[...,None])[...,0].T/nt
            pointsIt = (A[:2,:2]@pointsIt.T + A[:2,-1][...,None]).T
        #print(Xs.shape)
        # transform image
        #AI = interp(xI,I.swapaxes(1,2),Xs.permute(2,1,0),padding_mode="border")
        AI = interp(xI,I,Xs.permute(2,0,1),padding_mode="border")
        AIg = interp(xI,Ig,Xs.permute(2,0,1),padding_mode="border")
        #AI = AI.permute(0,2,1)
        #print(AI.shape)
        #print(J.shape)
        if it == 0:
            AI_orig = AIg.clone().detach()
        # objective function
        EMg = torch.sum((AIg - Jg)**2*WM)/2.0/sigmaMg**2
        EM = torch.sum((AI - J)**2*WM)/2.0/sigmaM**2
        ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v,dim=(1,2)))**2,dim=(0,-1))*LL)*DV/2.0/v.shape[1]/v.shape[2]/sigmaR**2
        E = EM + ER + EMg
        tosave = [E.item(), EM.item(), ER.item(), EMg.item()]
        if pointsIt.shape[0]>0:
            EP = torch.sum((pointsIt - pointsJ)**2)/2.0/sigmaP**2
            E += EP
            tosave.append(EP.item())
        
        Esave.append( tosave )
        # gradient update
        E.backward()

        optimizer.step()

        # Early stopping based on change in loss
        current_loss = E.item()
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if early_stopping and epochs_no_improve >= patience:
            print(f'Early stopping at iteration {it}')
            break

        # Gradient norm-based stopping
        grad_norm = torch.norm(torch.stack([L.grad.norm(), T.grad.norm(), v.grad.norm()]))
        if grad_norm < gradient_threshold:
            print(f'Stopping at iteration {it} due to small gradient norm')
            break

        # update weights
        if not it % 5:
            with torch.no_grad():
                # M step for these params
                if estimate_muA:
                    muA = torch.sum(WA*J,dim=(-1,-2))/torch.sum(WA)
                if estimate_muB:
                    muB = torch.sum(WB*J,dim=(-1,-2))/torch.sum(WB)
                #if it <= 200:
                #    muA = torch.tensor([0.75,0.77,0.79],device=J.device,dtype=J.dtype)
                #    muB = torch.ones(J.shape[0],device=J.device,dtype=J.dtype)*0.9

        # draw
        if not it % 10:
            AI_plt = AIg
            J_plt = Jg
            #print(AI.shape)
            if AI.shape[0] >= 1:
                AI_plt = torch.mean(AIg, 0)
                AI_plt = torch.unsqueeze(AI_plt, 0)
                AI_orig = torch.mean(AI_orig, 0)
                AI_orig = torch.unsqueeze(AI_orig, 0)
                J_plt = torch.mean(Jg, 0)
                J_plt = torch.unsqueeze(J_plt, 0)
            #print(AI_plt.shape)
            ax[0].cla()
            ax[0].imshow(((AI_orig - torch.amin(AI_orig, (1,2))[..., None, None]) / (torch.amax(AI_orig, (1,2)) - torch.amin(AI_orig, (1,2)))[..., None, None]).permute(1,2,0).clone().detach().cpu(), extent=extentJ)
            ax[0].scatter(pointsIt[:,1].clone().detach().cpu(), pointsIt[:,0].clone().detach().cpu())
            ax[0].set_title('space tformed source')
            
            ax[1].cla()
            ax[1].imshow(((AI_plt - torch.amin(AI_plt, (1,2))[..., None, None]) / (torch.amax(AI_plt, (1,2)) - torch.amin(AI_plt, (1,2)))[..., None, None]).permute(1,2,0).clone().detach().cpu(), extent=extentJ)
            ax[1].scatter(pointsIt[:,1].clone().detach().cpu(), pointsIt[:,0].clone().detach().cpu())
            ax[1].set_title('space tformed source')
            
            ax[4].cla()
            ax[4].imshow(clip((AI_plt - J_plt) / (torch.max(Jg).item()) * 3.0).permute(1,2,0).clone().detach().cpu() * 0.5 + 0.5, extent=extentJ)
            ax[4].scatter(pointsIt[:,1].clone().detach().cpu(), pointsIt[:,0].clone().detach().cpu())
            ax[4].scatter(pointsJ[:,1].clone().detach().cpu(), pointsJ[:,0].clone().detach().cpu())
            ax[4].set_title('Error')

            ax[2].cla()
            ax[2].imshow(J_plt.permute(1,2,0).cpu() / torch.max(Jg).item(), extent=extentJ)
            ax[2].scatter(pointsJ[:,1].clone().detach().cpu(), pointsJ[:,0].clone().detach().cpu())
            ax[2].set_title('Target')

            toshow = v[0].clone().detach().cpu()
            toshow /= torch.max(torch.abs(toshow))
            toshow = toshow * 0.5 + 0.5
            toshow = torch.cat((toshow, torch.zeros_like(toshow[..., 0][..., None])), -1)   
            ax[3].cla()
            ax[3].imshow(clip(toshow), extent=extentV)
            ax[3].set_title('velocity')
            
            axE[0].cla()
            axE[0].plot([e[0] for e in Esave])
            axE[0].legend(['E'])
            axE[0].set_yscale('log')
            axE[1].cla()
            axE[1].plot([e[1] for e in Esave])
            axE[1].legend(['EM'])
            axE[1].set_yscale('log')
            axE[2].cla()
            axE[2].plot([e[2] for e in Esave])
            axE[2].legend(['ER'])
            axE[2].set_yscale('log')
            axE[3].cla()
            axE[3].plot([e[3] for e in Esave])
            axE[3].legend(['EMg'])
            axE[3].set_yscale('log')
            axE[4].cla()
            axE[4].plot([e[4] for e in Esave])
            axE[4].legend(['EP'])
            axE[4].set_yscale('log')

            fig.canvas.draw()
            figE.canvas.draw()
            
    return {
        'A': A.clone().detach(), 
        'v': v.clone().detach(), 
        'xv': xv, 
        'WM': WM.clone().detach(),
        'WB': WB.clone().detach(),
        'WA': WA.clone().detach(),
        "AI_plt": AI_plt.clone().detach(),
        "AI": AI.clone().detach(),
        "J": J.clone().detach()
    }
