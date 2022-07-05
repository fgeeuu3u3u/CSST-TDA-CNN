
'''
Author: zx
Date: 2021-04-08 13:49:35
LastEditTime: 2022-07-05 14:58:42
LastEditors: xin zhangxinbjfu@gmail.com
Description: In User Settings Edit
FilePath: /undefined/Users/zhangxin/Work/SlitlessSim/sls_lit_demo/simDemo.py
'''

import galsim
import SpecDisperser
# from numpy import *
import numpy as np
from scipy import interpolate
import astropy.constants as acon
from astropy.table import Table
import math
from astropy.io import fits
import random

from astropy.table import Table
import matplotlib.pyplot as plt

import mpi4py.MPI as MPI

import os,sys

from . import Config


class SpecGenerator(object):
    def __init__(self,sedFn = 'a.txt', grating = 'GI', beam = 'A', aper = 2.0, xcenter = 5000,ycenter = 5000, p_size = 0.074, psf = None, skybg = 0.3, dark = 0.02, readout = 5, t = 150, expNum = 1, config = None):
        self.sedFile = sedFn
        self.grating = grating
        self.beam = beam
        self.aper = aper
        self.xcenter = xcenter
        self.ycenter = ycenter
        self.p_size = p_size
        self.psf = psf
        self.skybg = skybg
        self.dark = dark
        self.readout = readout
        self.t = t
        self.expNum = expNum
        self.config = config
    
    '''
    @description: 
    @param {*} fn: file name, include 2 column, wavelength(A)  and flux(erg/s/cm2/A) 
    @param {*} s: band start , unit A
    @param {*} e; end, unit A
    @param {*} deltL: sample interval for SED
    @return {*} sed, unit photo/s/m2/A
    '''
    def generateSEDfromFiles(self, fn, s, e, deltL):
        """
        s: lambda start, unit A
        e: lambda end, unit A

        return:
        SEDs is array, 2-dim, (gal_num+1)*(wavelength size), last row is wavelength
        """
        lamb = np.arange(s, e + deltL, deltL)
        spec_orig = np.loadtxt(fn)

        speci = interpolate.interp1d(spec_orig[:, 0], spec_orig[:, 1])
        y = speci(lamb)
        # erg/s/cm2/A --> photo/s/m2/A
        flux = y * lamb / (acon.h.value * acon.c.value) * 1e-13

        SED = Table(np.array([lamb, flux]).T,names=('WAVELENGTH', 'FLUX'))

        return SED


    def generateSpec1dforGal(self, s_n = 1.0, re = 1, pa = 90,q_ell = 0.6,limitfluxratio=0.9):

        specConfile = self.config.conFiles[self.grating]

        throughput_f = self.config.senFisle[self.grating] + '.' + self.config.orderIDs[self.beam] + '.fits'

        sed = self.generateSEDfromFiles(self.sedFile,2000,10000,0.5)

        # print(skybg)
        # print(specConfile)
        # print(throughput_f)

        # plt.figure()
        # plt.plot(sed['WAVELENGTH'], sed['FLUX'])

        gal = galsim.Sersic(s_n, half_light_radius=re)

        gal_pa = pa * galsim.degrees
        gal_ell = gal.shear(q=q_ell, beta=gal_pa)

        conv_gal = galsim.Convolve([gal_ell,self.psf])


        stamp = conv_gal.drawImage(wcs=galsim.PixelScale(self.p_size))*self.t*self.expNum*math.pi*(self.aper/2)*(self.aper/2)

        origin_star = [self.ycenter - (stamp.center.y - stamp.ymin),
                                self.xcenter - (stamp.center.x - stamp.xmin)]

        sdp = SpecDisperser.SpecDisperser(orig_img=stamp, xcenter=self.xcenter,
                                            ycenter=self.ycenter, origin=origin_star,
                                            tar_spec=sed,
                                            conf=specConfile,
                                            isAlongY=0)

        spec_orders = sdp.compute_spec_orders()
        
        thp = Table.read(throughput_f)
        thp_i = interpolate.interp1d(thp['WAVELENGTH'], thp['SENSITIVITY'])

        Aimg_orig = spec_orders[self.beam][0]
        Aimg = Aimg_orig

        Aimg = Aimg + (self.skybg + self.dark)*self.t*self.expNum

        Aimg = np.random.poisson(Aimg)
        for i in np.arange(self.expNum):
            Aimg = self.addReadoutNois(img = Aimg, readout = self.readout)

        Aimg = Aimg - (self.skybg + self.dark)*self.t*self.expNum


        wave_pix = spec_orders[self.beam][5]
        wave_pos = spec_orders[self.beam][3]

        wave_pos_y=spec_orders[self.beam][4]

        sh = Aimg.shape
        spec_pix = np.zeros(sh[1])
        err2_pix = np.zeros(sh[1])

        # print(spec_orders[beamOrder][4])
        # print(sh)
        # plt.figure()
        # plt.imshow(Aimg)
        y_cent_pos = int(np.round(np.mean(wave_pos_y)))

        tFlux = np.sum(spec_orders[self.beam][0])
        # print(tFlux)
        fluxRatio = 0
        for i in range(int(sh[0]/2)):
            pFlux = np.sum(spec_orders[self.beam][0][y_cent_pos-i:y_cent_pos+i+1])
            
            fluxRatio = pFlux/tFlux
            if fluxRatio>limitfluxratio:
                break
        y_range = i
        # print(y_range, fluxRatio)
        y_len_pix = 2 * y_range + 1
        for i in range(sh[1]):
            spec_pix[i] = sum(Aimg[y_cent_pos-y_range:y_cent_pos+y_range+1, i])
            err2_pix[i] = sum(Aimg_orig[y_cent_pos-y_range:y_cent_pos+y_range+1, i]) + (self.skybg + self.dark)*self.t * y_len_pix * self.expNum + self.readout*self.readout * y_len_pix * self.expNum

        bRange = self.config.bandRanges[self.grating]
        wave_flux = np.zeros(wave_pix.shape[0])
        err_flux = np.zeros(wave_pix.shape[0])
        for i in np.arange(1, wave_pix.shape[0] - 1):
            w = wave_pix[i]

            if (bRange[0] <= w <= bRange[1]):
                thp_w = thp_i(w)
                deltW = (w - wave_pix[i - 1]) / 2 + (wave_pix[i + 1] - w) / 2
                f = spec_pix[wave_pos[0] - 1 + i]
                f = f / t / thp_w / deltW /self.expNum
                err = err2_pix[wave_pos[0] - 1 + i]
                # err = err/ t / deltW
                err = np.sqrt(err)/ self.t / deltW/ thp_w /self.expNum
                # err = err / thp_w 
            else:
                f = 0
                err = 0

            wave_flux[i] = f
            err_flux[i] = err
        
        idx = (wave_pix >= bRange[0]-100)
        idx1 = (wave_pix[idx] <= bRange[1]+100)

        specTab = Table(np.array([wave_pix[idx][idx1],  wave_flux[idx][idx1], err_flux[idx][idx1]]).T,names=('WAVELENGTH', 'FLUX','ERR'))

        # spec_orig = np.loadtxt(sedFile)

        # plt.figure()
        # plt.plot(spec_orig[:,0], spec_orig[:,1])

        # plt.figure()
        # plt.errorbar(wave_pix[idx][idx1], wave_flux[idx][idx1],err_flux[idx][idx1])
        # plt.legend([self.sedFile])
        # # plt.plot(wave_pix[idx][idx1], wave_flux[idx][idx1])
        # plt.show()
        return specTab, Aimg, stamp.array


    def generateSpec1dforStar(self,limitfluxratio = 0.8):

        specConfile = self.config.conFiles[self.grating]

        throughput_f = self.config.senFisle[self.grating] + '.' + self.config.orderIDs[self.beam] + '.fits'

        sed = self.generateSEDfromFiles(self.sedFile,2000,10000,0.5)


        stamp = self.psf.drawImage(wcs=galsim.PixelScale(self.p_size))*self.t*self.expNum*math.pi*(self.aper/2)*(self.aper/2)

        origin_star = [self.ycenter - (stamp.center.y - stamp.ymin),
                                self.xcenter - (stamp.center.x - stamp.xmin)]

        sdp = SpecDisperser.SpecDisperser(orig_img=stamp, xcenter=self.xcenter,
                                            ycenter=self.ycenter, origin=origin_star,
                                            tar_spec=sed,
                                            conf=specConfile,
                                            isAlongY=0)

        spec_orders = sdp.compute_spec_orders()
        
        thp = Table.read(throughput_f)
        thp_i = interpolate.interp1d(thp['WAVELENGTH'], thp['SENSITIVITY'])

        Aimg_orig = spec_orders[self.beam][0]
        Aimg = Aimg_orig

        Aimg = Aimg + (self.skybg + self.dark)*self.t*self.expNum

        Aimg = np.random.poisson(Aimg)
        for i in np.arange(self.expNum):
            Aimg = self.addReadoutNois(img = Aimg, readout = self.readout)

        Aimg = Aimg - (self.skybg + self.dark)*self.t*self.expNum


        wave_pix = spec_orders[self.beam][5]
        wave_pos = spec_orders[self.beam][3]

        wave_pos_y=spec_orders[self.beam][4]

        sh = Aimg.shape
        spec_pix = np.zeros(sh[1])
        err2_pix = np.zeros(sh[1])

        # print(spec_orders[beamOrder][4])
        # print(sh)
        # plt.figure()
        # plt.imshow(Aimg)
        y_cent_pos = int(np.round(np.mean(wave_pos_y)))

        tFlux = np.sum(spec_orders[self.beam][0])
        # print(tFlux)
        fluxRatio = 0
        for i in range(int(sh[0]/2)):
            pFlux = np.sum(spec_orders[self.beam][0][y_cent_pos-i:y_cent_pos+i+1])
            
            fluxRatio = pFlux/tFlux
            if fluxRatio>limitfluxratio:
                break
        y_range = i
        # print(y_range, fluxRatio)
        y_len_pix = 2 * y_range + 1
        for i in range(sh[1]):
            spec_pix[i] = sum(Aimg[y_cent_pos-y_range:y_cent_pos+y_range+1, i])
            err2_pix[i] = sum(Aimg_orig[y_cent_pos-y_range:y_cent_pos+y_range+1, i]) + (self.skybg + self.dark)*self.t * y_len_pix * self.expNum + self.readout*self.readout * y_len_pix * self.expNum

        bRange = self.config.bandRanges[self.grating]
        wave_flux = np.zeros(wave_pix.shape[0])
        err_flux = np.zeros(wave_pix.shape[0])
        for i in np.arange(1, wave_pix.shape[0] - 1):
            w = wave_pix[i]

            if (bRange[0] <= w <= bRange[1]):
                thp_w = thp_i(w)
                deltW = (w - wave_pix[i - 1]) / 2 + (wave_pix[i + 1] - w) / 2
                f = spec_pix[wave_pos[0] - 1 + i]
                f = f / self.t / thp_w / deltW /self.expNum
                err = err2_pix[wave_pos[0] - 1 + i]
                # err = err/ t / deltW
                err = np.sqrt(err)/ self.t / deltW/ thp_w /self.expNum
                # err = err / thp_w 
            else:
                f = 0
                err = 0

            wave_flux[i] = f
            err_flux[i] = err
        
        idx = (wave_pix >= bRange[0]-100)
        idx1 = (wave_pix[idx] <= bRange[1]+100)

        specTab = Table(np.array([wave_pix[idx][idx1],  wave_flux[idx][idx1], err_flux[idx][idx1]]).T,names=('WAVELENGTH', 'FLUX','ERR'))

        # spec_orig = np.loadtxt(sedFile)

        # plt.figure()
        # plt.plot(spec_orig[:,0], spec_orig[:,1])

        # plt.figure()
        # plt.errorbar(wave_pix[idx][idx1], wave_flux[idx][idx1],err_flux[idx][idx1])
        # plt.legend([self.sedFile])
        # # plt.plot(wave_pix[idx][idx1], wave_flux[idx][idx1])
        # plt.show()
        return specTab, Aimg, stamp.array, fluxRatio

    def addReadoutNois(self, img = None, readout = 5):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i,j] += round(random.gauss(mu = 0, sigma = readout))

        return img

