'''
Author: zx
Date: 2021-04-08 13:49:35
LastEditTime: 2023-02-24 00:53:19
LastEditors: xin zhangxinbjfu@gmail.com
Description: In User Settings Edit
FilePath: /undefined/Users/zhangxin/Work/SlitlessSim/sls_lit_demo/SpecDisperser/SpecDisperser.py
'''

from astropy.table import Table

import astropy.constants as cons

import collections
from collections import OrderedDict
from scipy import interpolate

from astropy import units as u

# from numpy import *

import numpy as np

from pylab import *

import galsim

import sys
import os


def rotate90(array_orig=None, xc=0, yc=0, isClockwise=0):
    if array_orig is None:
        return

    l1 = array_orig.shape[0]
    l2 = array_orig.shape[1]

    if xc < 0 or xc > l2 - 1 or yc < 0 or yc > l1 - 1:
        return

    n_xc = xc
    n_yc = yc

    array_final = np.zeros_like(array_orig.T)

    if isClockwise == 1:
        for i in np.arange(l2):
            array_final[i] = array_orig[:, l2 - i - 1]
        n_xc = yc
        n_yc = l2 - 1 - xc
    else:
        for i in np.arange(l2):
            array_final[i] = array_orig[::-1, i]

        n_xc = l1 - 1 - yc
        n_yc = xc
    return array_final, n_xc, n_yc


class SpecDisperser(object):
    def __init__(self, orig_img=None, xcenter=0, ycenter=0, origin=[100, 100], tar_spec=None, band_start=2550,
                 band_end=10000, isAlongY=0, conf='../param/CONF/csst.conf', gid=0, deltLamb = 0.01):
        """
        orig_img: normal image,galsim image
        xcenter, ycenter: the center of galaxy in orig_img
        origin : [int, int]
            `origin` defines the lower left pixel index (y,x) of the `direct`
            cutout from a larger detector-frame image
        tar_spec: galsim.SED

        """

        # self.img_x = orig_img.shape[1]
        # self.img_y = orig_img.shape[0]

        self.thumb_img = np.abs(orig_img.array)
        self.thumb_x = orig_img.center.x
        self.thumb_y = orig_img.center.y
        self.img_sh = orig_img.array.shape

        self.id = gid

        self.xcenter = xcenter
        self.ycenter = ycenter

        self.isAlongY = isAlongY
        if self.isAlongY == 1:
            self.thumb_img, self.thumb_x, self.thumb_y = rotate90(array_orig=self.thumb_img, xc=orig_img.center.x,
                                                                       yc=orig_img.center.y, isClockwise=1)

            self.img_sh = orig_img.array.T.shape
            self.xcenter = ycenter
            self.ycenter = xcenter

        self.origin = origin
        self.band_start = band_start
        self.band_end = band_end
        self.spec = tar_spec
        self.deltLamb = deltLamb

        self.beam_flux = OrderedDict()

        self.grating_conf = aXeConf(conf)
        self.grating_conf.get_beams()
        self.grating_conf_file = conf

    def compute_spec_orders(self):

        all_orders = OrderedDict()
        beam_names = self.grating_conf.beams

        for beam in beam_names:
            all_orders[beam] = self.compute_spec(beam)

        return all_orders

    def compute_spec(self, beam):

        from .disperse_c import interp
        from .disperse_c import disperse
        # from MockObject.disperse_c import disperse

        dx = self.grating_conf.dxlam[beam]
        xoff = 0
        ytrace_beam, lam_beam = self.grating_conf.get_beam_trace(x=self.xcenter, y=self.ycenter, dx=(dx + xoff),
                                                                 beam=beam)

        ### Account for pixel centering of the trace
        yfrac_beam = ytrace_beam - floor(ytrace_beam)

        ysens = lam_beam * 0
        lam_index = argsort(lam_beam)
        conf_sens = self.grating_conf.sens[beam]

        lam_intep = np.linspace(self.band_start, self.band_end, int((self.band_end - self.band_start) / self.deltLamb))

        thri = interpolate.interp1d(conf_sens['WAVELENGTH'], conf_sens['SENSITIVITY'])
        spci = interpolate.interp1d(self.spec['WAVELENGTH'], self.spec['FLUX'])

        beam_thr = thri(lam_intep)
        spec_sample = spci(lam_intep)

        bean_thr_spec = beam_thr * spec_sample

        ysensitivity = lam_beam * 0

        ysensitivity[lam_index] = interp.interp_conserve_c(lam_beam[lam_index], lam_intep,
                                                           beam_thr * math.pi * 100 * 100 * 1e-7 / (
                                                                       cons.h.value * cons.c.value / (
                                                                           lam_intep * 1e-10)), integrate=0, left=0,
                                                           right=0)
        
        # self.writerSensitivityFile(conffile = self.grating_conf_file,  beam = beam, w = lam_beam[lam_index], sens = ysensitivity[lam_index])
        
        
        ysens[lam_index] = interp.interp_conserve_c(lam_beam[lam_index], lam_intep, bean_thr_spec, integrate=1, left=0,
                                                    right=0)

        sensitivity_beam = ysens

        len_spec_x = len(dx)
        len_spec_y = int(ceil(yfrac_beam[-1]) - floor(yfrac_beam[0]) + 1)

        beam_sh = (self.img_sh[0] + len_spec_y, self.img_sh[1] + len_spec_x)
        modelf = zeros(product(beam_sh), dtype=float)
        model = modelf.reshape(beam_sh)
        idx = np.arange(modelf.size, dtype=int64).reshape(beam_sh)
        x0 = array((self.thumb_y, self.thumb_x), dtype=int64)

        dxpix = dx - dx[0] + x0[1]

        dyc = cast[int](ytrace_beam)

        dypix = dyc - dyc[0] + x0[0]
        flat_index = idx[dypix, dxpix]

        nonz = sensitivity_beam != 0

        status = disperse.disperse_grism_object(self.thumb_img,
                                                flat_index[nonz], yfrac_beam[nonz],
                                                sensitivity_beam[nonz],
                                                modelf, x0,
                                                array(self.img_sh, dtype=int64),
                                                array(beam_sh, dtype=int64))

        model = modelf.reshape(beam_sh)
        self.beam_flux[beam] = sum(modelf)

        origin_in = zeros_like(self.origin)
        dx0_in = dx[0]
        dy0_in = dyc[0]
        if self.isAlongY == 1:
            model, tmx, tmy = rotate90(array_orig=model, isClockwise=0)
            origin_in[0] = self.origin[0]
            origin_in[1] = self.origin[1] - len_spec_y
            dx0_in = -dyc[0]
            dy0_in = dx[0]
        else:
            origin_in[0] = self.origin[0]
            origin_in[1] = self.origin[1]
            dx0_in = dx[0]
            dy0_in = dyc[0]
        originOut_x = origin_in[1] + dx0_in - 1
        originOut_y = origin_in[0] + dy0_in - 1

        return model, originOut_x, originOut_y, dxpix, dypix, lam_beam,ysens

    # def writerSensitivityFile(self, conffile = '', beam = '', w = None, sens = None):
    #     orders={'A':'1st','B':'0st','C':'2st'}
    #     sens_file_name = conffile[0:-5]+'_sensitivity_'+ orders[beam] + '.fits'
    #     if not os.path.exists(sens_file_name) == True:
    #         senstivity_out = Table(array([w,sens]).T, names=('WAVELENGTH', 'SENSITIVITY'))
    #         senstivity_out.write(sens_file_name, format='fits')
    def writerSensitivityFile(self, conffile='', beam='', w=None, sens=None):
        orders = {'A': '1st', 'B': '0st', 'C': '2st', 'D': '-1st', 'E': '-2st'}
        sens_file_name = conffile[0:-5] + '_sensitivity_' + orders[beam] + '.fits'
        if not os.path.exists(sens_file_name) == True:
            senstivity_out = Table(array([w, sens]).T, names=('WAVELENGTH', 'SENSITIVITY'))
            senstivity_out.write(sens_file_name, format='fits')


"""
Demonstrate aXe trace polynomials.
"""

class aXeConf():
    def __init__(self, conf_file='WFC3.IR.G141.V2.5.conf'):
        """Read an aXe-compatible configuration file
        
        Parameters
        ----------
        conf_file: str
            Filename of the configuration file to read
        
        """
        if conf_file is not None:
            self.conf = self.read_conf_file(conf_file)
            self.conf_file = conf_file
            self.count_beam_orders()

            ## Global XOFF/YOFF offsets
            if 'XOFF' in self.conf.keys():
                self.xoff = np.float(self.conf['XOFF'])
            else:
                self.xoff = 0.

            if 'YOFF' in self.conf.keys():
                self.yoff = np.float(self.conf['YOFF'])
            else:
                self.yoff = 0.

    def read_conf_file(self, conf_file='WFC3.IR.G141.V2.5.conf'):
        """Read an aXe config file, convert floats and arrays
        
        Parameters
        ----------
        conf_file: str
            Filename of the configuration file to read.
        
        Parameters are stored in an OrderedDict in `self.conf`.
        """
        from collections import OrderedDict

        conf = OrderedDict()
        lines = open(conf_file).readlines()
        for line in lines:
            ## empty / commented lines
            if (line.startswith('#')) | (line.strip() == '') | ('"' in line):
                continue

            ## split the line, taking out ; and # comments
            spl = line.split(';')[0].split('#')[0].split()
            param = spl[0]
            if len(spl) > 2:
                value = np.cast[float](spl[1:])
            else:
                try:
                    value = float(spl[1])
                except:
                    value = spl[1]

            conf[param] = value

        return conf

    def count_beam_orders(self):
        """Get the maximum polynomial order in DYDX or DLDP for each beam
        """
        self.orders = {}
        for beam in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
            order = 0
            while 'DYDX_{0:s}_{1:d}'.format(beam, order) in self.conf.keys():
                order += 1

            while 'DLDP_{0:s}_{1:d}'.format(beam, order) in self.conf.keys():
                order += 1

            self.orders[beam] = order - 1

    def get_beams(self):
        """Get beam parameters and read sensitivity curves
        """
        import os
        from collections import OrderedDict
        from astropy.table import Table, Column

        self.dxlam = OrderedDict()
        self.nx = OrderedDict()
        self.sens = OrderedDict()
        self.beams = []

        for beam in self.orders:
            if self.orders[beam] > 0:
                self.beams.append(beam)
                self.dxlam[beam] = np.arange(self.conf['BEAM{0}'.format(beam)].min(),
                                             self.conf['BEAM{0}'.format(beam)].max(), dtype=int)
                self.nx[beam] = int(self.dxlam[beam].max() - self.dxlam[beam].min()) + 1
                self.sens[beam] = Table.read(
                    '{0}/{1}'.format(os.path.dirname(self.conf_file), self.conf['SENSITIVITY_{0}'.format(beam)]))
                # self.sens[beam].wave = np.cast[np.double](self.sens[beam]['WAVELENGTH'])
                # self.sens[beam].sens = np.cast[np.double](self.sens[beam]['SENSITIVITY'])

                ### Need doubles for interpolating functions
                for col in self.sens[beam].colnames:
                    data = np.cast[np.double](self.sens[beam][col])
                    self.sens[beam].remove_column(col)
                    self.sens[beam].add_column(Column(data=data, name=col))

        self.beams.sort()

    def field_dependent(self, xi, yi, coeffs):
        """aXe field-dependent coefficients
        
        See the `aXe manual <http://axe.stsci.edu/axe/manual/html/node7.html#SECTION00721200000000000000>`_ for a description of how the field-dependent coefficients are specified.
        
        Parameters
        ----------
        xi, yi : float or array-like
            Coordinate to evaluate the field dependent coefficients, where
            `xi = x-REFX` and `yi = y-REFY`.
        
        coeffs : array-like
            Field-dependency coefficients
        
        Returns
        -------
        a : float or array-like
            Evaluated field-dependent coefficients
            
        """
        ## number of coefficients for a given polynomial order
        ## 1:1, 2:3, 3:6, 4:10, order:order*(order+1)/2
        if isinstance(coeffs, float):
            order = 1
        else:
            order = int(-1 + np.sqrt(1 + 8 * len(coeffs))) // 2

        ## Build polynomial terms array
        ## $a = a_0+a_1x_i+a_2y_i+a_3x_i^2+a_4x_iy_i+a_5yi^2+$ ...
        xy = []
        for p in range(order):
            for px in range(p + 1):
                # print 'x**%d y**%d' %(p-px, px)
                xy.append(xi ** (p - px) * yi ** (px))

        ## Evaluate the polynomial, allowing for N-dimensional inputs
        a = np.sum((np.array(xy).T * coeffs).T, axis=0)

        return a

    def evaluate_dp(self, dx, dydx):
        """Evalate arc length along the trace given trace polynomial coefficients
        
        Parameters
        ----------
        dx : array-like
            x pixel to evaluate
        
        dydx : array-like
            Coefficients of the trace polynomial
        
        Returns
        -------
        dp : array-like
            Arc length along the trace at position `dx`.
            
        For `dydx` polynomial orders 0, 1 or 2, integrate analytically.  
        Higher orders must be integrated numerically.
        
        **Constant:** 
            .. math:: dp = dx

        **Linear:** 
            .. math:: dp = \sqrt{1+\mathrm{DYDX}[1]}\cdot dx
        
        **Quadratic:** 
            .. math:: u = \mathrm{DYDX}[1] + 2\ \mathrm{DYDX}[2]\cdot dx
            
            .. math:: dp = (u \sqrt{1+u^2} + \mathrm{arcsinh}\ u) / (4\cdot \mathrm{DYDX}[2])
        
        """
        ## dp is the arc length along the trace
        ## $\lambda = dldp_0 + dldp_1 dp + dldp_2 dp^2$ ...

        poly_order = len(dydx) - 1
        if (poly_order == 2):
            if np.abs(np.unique(dydx[2])).max() == 0:
                poly_order = 1

        if poly_order == 0:  ## dy=0
            dp = dx
        elif poly_order == 1:  ## constant dy/dx
            dp = np.sqrt(1 + dydx[1] ** 2) * (dx)
        elif poly_order == 2:  ## quadratic trace
            u0 = dydx[1] + 2 * dydx[2] * (0)
            dp0 = (u0 * np.sqrt(1 + u0 ** 2) + np.arcsinh(u0)) / (4 * dydx[2])
            u = dydx[1] + 2 * dydx[2] * (dx)
            dp = (u * np.sqrt(1 + u ** 2) + np.arcsinh(u)) / (4 * dydx[2]) - dp0
        else:
            ## high order shape, numerical integration along trace
            ## (this can be slow)
            xmin = np.minimum((dx).min(), 0)
            xmax = np.maximum((dx).max(), 0)
            xfull = np.arange(xmin, xmax)
            dyfull = 0
            for i in range(1, poly_order):
                dyfull += i * dydx[i] * (xfull - 0.5) ** (i - 1)

            ## Integrate from 0 to dx / -dx
            dpfull = xfull * 0.
            lt0 = xfull < 0
            if lt0.sum() > 1:
                dpfull[lt0] = np.cumsum(np.sqrt(1 + dyfull[lt0][::-1] ** 2))[::-1]
                dpfull[lt0] *= -1

            #
            gt0 = xfull > 0
            if gt0.sum() > 0:
                dpfull[gt0] = np.cumsum(np.sqrt(1 + dyfull[gt0] ** 2))

            dp = np.interp(dx, xfull, dpfull)
            if dp[-1] == dp[-2]:
                dp[-1] = dp[-2] + np.diff(dp)[-2]

        return dp

    def get_beam_trace(self, x=507, y=507, dx=0., beam='A'):
        """Get an aXe beam trace for an input reference pixel and list of output x pixels `dx`
        
        Parameters
        ----------
        x, y : float or array-like
            Evaluate trace definition at detector coordinates `x` and `y`.
            
        dx : float or array-like
            Offset in x pixels from `(x,y)` where to compute trace offset and 
            effective wavelength
            
        beam : str
            Beam name (i.e., spectral order) to compute.  By aXe convention, 
            `beam='A'` is the first order, 'B' is the zeroth order and 
            additional beams are the higher positive and negative orders.
        
        Returns
        -------
        dy : float or array-like
            Center of the trace in y pixels offset from `(x,y)` evaluated at
            `dx`.
            
        lam : float or array-like
            Effective wavelength along the trace evaluated at `dx`.
            
        """
        NORDER = self.orders[beam] + 1

        xi, yi = x - self.xoff, y - self.yoff
        xoff_beam = self.field_dependent(xi, yi, self.conf['XOFF_{0}'.format(beam)])
        yoff_beam = self.field_dependent(xi, yi, self.conf['YOFF_{0}'.format(beam)])

        ## y offset of trace (DYDX)
        dydx = np.zeros(NORDER)  # 0 #+1.e-80
        dydx = [0] * NORDER

        for i in range(NORDER):
            if 'DYDX_{0:s}_{1:d}'.format(beam, i) in self.conf.keys():
                coeffs = self.conf['DYDX_{0:s}_{1:d}'.format(beam, i)]
                dydx[i] = self.field_dependent(xi, yi, coeffs)

        # $dy = dydx_0+dydx_1 dx+dydx_2 dx^2+$ ...

        dy = yoff_beam
        for i in range(NORDER):
            dy += dydx[i] * (dx - xoff_beam) ** i

        ## wavelength solution    
        dldp = np.zeros(NORDER)
        dldp = [0] * NORDER

        for i in range(NORDER):
            if 'DLDP_{0:s}_{1:d}'.format(beam, i) in self.conf.keys():
                coeffs = self.conf['DLDP_{0:s}_{1:d}'.format(beam, i)]
                dldp[i] = self.field_dependent(xi, yi, coeffs)

        self.eval_input = {'x': x, 'y': y, 'beam': beam, 'dx': dx}
        self.eval_output = {'xi': xi, 'yi': yi, 'dldp': dldp, 'dydx': dydx,
                            'xoff_beam': xoff_beam, 'yoff_beam': yoff_beam,
                            'dy': dy}

        dp = self.evaluate_dp(dx - xoff_beam, dydx)
        # ## dp is the arc length along the trace
        # ## $\lambda = dldp_0 + dldp_1 dp + dldp_2 dp^2$ ...
        # if self.conf['DYDX_ORDER_%s' %(beam)] == 0:   ## dy=0
        #     dp = dx-xoff_beam                      
        # elif self.conf['DYDX_ORDER_%s' %(beam)] == 1: ## constant dy/dx
        #     dp = np.sqrt(1+dydx[1]**2)*(dx-xoff_beam)
        # elif self.conf['DYDX_ORDER_%s' %(beam)] == 2: ## quadratic trace
        #     u0 = dydx[1]+2*dydx[2]*(0)
        #     dp0 = (u0*np.sqrt(1+u0**2)+np.arcsinh(u0))/(4*dydx[2])
        #     u = dydx[1]+2*dydx[2]*(dx-xoff_beam)
        #     dp = (u*np.sqrt(1+u**2)+np.arcsinh(u))/(4*dydx[2])-dp0
        # else:
        #     ## high order shape, numerical integration along trace
        #     ## (this can be slow)
        #     xmin = np.minimum((dx-xoff_beam).min(), 0)
        #     xmax = np.maximum((dx-xoff_beam).max(), 0)
        #     xfull = np.arange(xmin, xmax)
        #     dyfull = 0
        #     for i in range(1, NORDER):
        #         dyfull += i*dydx[i]*(xfull-0.5)**(i-1)
        #     
        #     ## Integrate from 0 to dx / -dx
        #     dpfull = xfull*0.
        #     lt0 = xfull <= 0
        #     if lt0.sum() > 1:
        #         dpfull[lt0] = np.cumsum(np.sqrt(1+dyfull[lt0][::-1]**2))[::-1]
        #         dpfull[lt0] *= -1
        #     #
        #     gt0 = xfull >= 0
        #     if gt0.sum() > 0:
        #         dpfull[gt0] = np.cumsum(np.sqrt(1+dyfull[gt0]**2))
        #       
        #     dp = np.interp(dx-xoff_beam, xfull, dpfull)

        ## Evaluate dldp    
        lam = dp * 0.
        for i in range(NORDER):
            lam += dldp[i] * dp ** i

        return dy, lam

    def show_beams(self, beams=['E', 'D', 'C', 'B', 'A']):
        """
        Make a demo plot of the beams of a given configuration file
        """
        import matplotlib.pyplot as plt

        x0, x1 = 507, 507
        dx = np.arange(-800, 1200)

        if 'WFC3.UV' in self.conf_file:
            x0, x1 = 2073, 250
            dx = np.arange(-1200, 1200)
        if 'G800L' in self.conf_file:
            x0, x1 = 2124, 1024
            dx = np.arange(-1200, 1200)

        s = 200  # marker size
        fig = plt.figure(figsize=[10, 3])
        plt.scatter(0, 0, marker='s', s=s, color='black', edgecolor='0.8',
                    label='Direct')

        for beam in beams:
            if 'XOFF_{0}'.format(beam) not in self.conf.keys():
                continue

            xoff = self.field_dependent(x0, x1, self.conf['XOFF_{0}'.format(beam)])
            dy, lam = self.get_beam_trace(x0, x1, dx=dx, beam=beam)
            xlim = self.conf['BEAM{0}'.format(beam)]
            ok = (dx >= xlim[0]) & (dx <= xlim[1])
            plt.scatter(dx[ok] + xoff, dy[ok], c=lam[ok] / 1.e4, marker='s', s=s,
                        alpha=0.5, edgecolor='None')
            plt.text(np.median(dx[ok]), np.median(dy[ok]) + 1, beam,
                     ha='center', va='center', fontsize=14)
            print('Beam {0}, lambda=({1:.1f} - {2:.1f})'.format(beam, lam[ok].min(), lam[ok].max()))

        plt.grid()
        plt.xlabel(r'$\Delta x$')
        plt.ylabel(r'$\Delta y$')

        cb = plt.colorbar(pad=0.01, fraction=0.05)
        cb.set_label(r'$\lambda\,(\mu\mathrm{m})$')
        plt.title(self.conf_file)
        plt.tight_layout()
        plt.savefig('{0}.pdf'.format(self.conf_file))

    # def load_grism_config(conf_file):
#     """Load parameters from an aXe configuration file

#     Parameters
#     ----------
#     conf_file : str
#         Filename of the configuration file

#     Returns
#     -------
#     conf : `~grizli.grismconf.aXeConf`
#         Configuration file object.  Runs `conf.get_beams()` to read the 
#         sensitivity curves.
#     """
#     conf = aXeConf(conf_file)
#     conf.get_beams()
#     return conf
