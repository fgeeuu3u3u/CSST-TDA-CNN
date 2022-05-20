"""
Pythonic utilities ported to C [Cython] for speedup.
"""
import numpy as np
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t
ctypedef np.int_t ITYPE_t
ctypedef np.uint_t UINT_t

import cython

cdef extern from "math.h":
    double fabs(double)

cdef extern from"stdio.h":
    extern int printf(const char *format, ...)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def interp_c(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] xp, np.ndarray[DTYPE_t, ndim=1] fp, double extrapolate=0., short int assume_sorted=1):
    """
    interp_c(x, xp, fp, extrapolate=0., assume_sorted=0)
    
    Fast interpolation: [`xp`, `fp`] interpolated at `x`.
    
    Extrapolated values are set to `extrapolate`.
    
    The default `assume_sorted`=1 assumes that the `x` array is sorted and single-
    valued, providing a significant gain in speed. (xp is always assumed to be sorted)
    
    """
    cdef unsigned long i, j, N, Np
    cdef DTYPE_t x1,x2,y1,y2,out
    cdef DTYPE_t fout, xval, xmin
    
    N, Np = len(x), len(xp)
    cdef np.ndarray[DTYPE_t, ndim=1] f = np.zeros(N)

    i=0
    j=0
    ### Handle left extrapolation
    xmin = xp[0]    
    if assume_sorted == 1:
        while x[j] < xmin:
            f[j] = extrapolate
            j+=1
            if j>=N:
                break
        
    while j < N:
        xval = x[j]
        if assume_sorted == 0:
            if x[j] < xmin:
                f[j] = extrapolate
                j+=1
                continue
            else:
                i=0
                
        while (xp[i] <= xval) & (i < Np-1): i+=1;
        
        if i == (Np-1):
            if x[j] != xp[i]:
                f[j] = extrapolate
            else:
                f[j] = fp[i]
            j+=1
            continue   
        
        #### x[i] is now greater than xval because the 
        #### expression (x[i]<xval) is false, assuming
        #### that xval < max(x).
        
        # x1 = xp[i];
        # x2 = xp[i+1];
        # y1 = fp[i];
        # y2 = fp[i+1];
        x1 = xp[i-1];
        x2 = xp[i];
        y1 = fp[i-1];
        y2 = fp[i];
        out = ((y2-y1)/(x2-x1))*(xval-x1)+y1;
        f[j] = out
        j+=1
                
    return f
        
    
@cython.boundscheck(False)
def interp_conserve_c(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] tlam, np.ndarray[DTYPE_t, ndim=1] tf, double left=0, double right=0, double integrate=0):
    """
    interp_conserve_c(x, xp, fp, left=0, right=0, integrate=0)
    
    Interpolate `xp`,`yp` array to the output x array, conserving flux.  
    `xp` can be irregularly spaced.
    """
    cdef np.ndarray[DTYPE_t, ndim=1] templmid
    cdef np.ndarray[DTYPE_t, ndim=1] tempfmid
    cdef np.ndarray[DTYPE_t, ndim=1] outy
    cdef unsigned long i,k,istart,ntlam,NTEMPL
    cdef DTYPE_t h, numsum
    
    # templmid = (x[1:]+x[:-1])/2. #2.+x[:-1]
    # templmid = np.append(templmid, np.array([x[0], x[-1]]))
    # templmid = templmid[np.argsort(templmid)]
    NTEMPL = len(x)
    ntlam = len(tlam)

    templmid = midpoint_c(x, NTEMPL)
    #tempfmid = np.interp(templmid, tlam, tf, left=left, right=right)
    tempfmid = interp_c(templmid, tlam, tf, extrapolate=0.)
    
    outy = np.zeros(NTEMPL, dtype=DTYPE)

    ###### Rebin template grid to master wavelength grid, conserving template flux
    i=0
    k=0
    
 
    while templmid[k] < tlam[0]:
        outy[k] = left
        k+=1
        if k >NTEMPL-1:
            break

    if(k>0) & (templmid[k-1] < tlam[0]) & (templmid[k] > tlam[0]):
        m = 1;
        numsum=0.;
        while (tlam[m] < templmid[k]):
            h = tlam[m]-tlam[m-1];
            numsum+=h*(tf[m]+tf[m-1]);
            m+=1;
            if m >= ntlam:
                break;
        #print 'test #%d, %d' %(m, ntlam) 

        if m == 1:
            h = templmid[k]-tlam[0];
            numsum+=h*(tempfmid[k]+tf[0]);
        else:  
            ##### Last point
            if m < ntlam:
                if (templmid[k] == tlam[m]):
                    h = tlam[m]-tlam[m-1];
                    numsum+=h*(tf[m]+tf[m-1]);
                else:
                    m-=1;
                    h = templmid[k]-tlam[m];
                    numsum+=h*(tempfmid[k]+tf[m]);                                   
            
        outy[k-1] = numsum*0.5;#/(templmid[k+1]-templmid[k]);
        if integrate == 0.:
            outy[k-1] /= (templmid[k]-templmid[k-1]);

    for k in range(k, NTEMPL):
        if templmid[k] > tlam[ntlam-1]:
            break
            
        numsum=0.;

        #### Go to where tlam is greater than the first midpoint
        while (tlam[i] < templmid[k]) & (i < ntlam): i+=1;
        istart=i;

        ####### First point
        if tlam[i] < templmid[k+1]: 
            h = tlam[i]-templmid[k];
            numsum+=h*(tf[i]+tempfmid[k]);
            i+=1;

        if i==0: i+=1;

        ####### Template points between master grid points
        while (tlam[i] < templmid[k+1]) & (i < ntlam):
            h = tlam[i]-tlam[i-1];
            numsum+=h*(tf[i]+tf[i-1]);
            i+=1;

        #### If no template points between master grid points, then just use interpolated midpoints
        if i == istart:
            h = templmid[k+1]-templmid[k];
            numsum=h*(tempfmid[k+1]+tempfmid[k]);
        else:  
            ##### Last point              
            if (templmid[k+1] == tlam[i]) & (i < ntlam):
                h = tlam[i]-tlam[i-1];
                numsum+=h*(tf[i]+tf[i-1]);
            else:
                i-=1;
                h = templmid[k+1]-tlam[i];
                numsum+=h*(tempfmid[k+1]+tf[i]);

        outy[k] = numsum*0.5;#/(templmid[k+1]-templmid[k]);
        if integrate == 0.:
            outy[k] /= (templmid[k+1]-templmid[k]);
            
    return outy
    
def midpoint(x):
    mp = (x[1:]+x[:-1])/2.
    mp = np.append(mp, np.array([x[0],x[-1]]))
    mp = mp[np.argsort(mp)]
    return mp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def midpoint_c(np.ndarray[DTYPE_t, ndim=1] x, long N):
    cdef long i
    cdef DTYPE_t xi,xi1
    # N = len(x)
    cdef np.ndarray[DTYPE_t, ndim=1] midpoint = np.zeros(N+1, dtype=DTYPE)
    midpoint[0] = x[0]
    midpoint[N] = x[N-1]
    xi1 = x[0]
    for i in range(1, N):
        xi = x[i]
        midpoint[i] = 0.5*xi+0.5*xi1
        xi1 = xi
    
    midpoint[0] = 2*x[0]-midpoint[1]
    midpoint[N] = 2*x[N-1]-midpoint[N-1]
    
    return midpoint    
