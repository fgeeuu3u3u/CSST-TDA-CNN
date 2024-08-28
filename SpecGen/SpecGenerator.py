
'''
Author: zx
Date: 2021-04-08 13:49:35
LastEditTime: 2024-08-28 16:45:32
LastEditors: Zhang Xin zhangx@bao.ac.cn
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
import time

import mpi4py.MPI as MPI

import os,sys
import photutils

from . import Config


class SpecGenerator(object):
    def __init__(self,sedFn = 'a.txt', grating = 'GI', beam = 'A', aper = 2.0, xcenter = 5000,ycenter = 5000, p_size = 0.074, psf = None, skybg = 0.3, dark = 0.02, readout = 5, t = 150, expNum = 1, config = None, saturation = 90000):
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
        self.saturation = saturation
    
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


    def generateSpec1dforGal(self, s_n = 1.0, re = 1, pa = 90,q_ell = 0.6,limitfluxratio=0.9,deltLamb = 0.01):

        specConfile = self.config.conFiles[self.grating]

        throughput_f = self.config.senFisle[self.grating] + self.config.orderIDs[self.beam] + '.fits'

        sed = self.generateSEDfromFiles(self.sedFile,2500,10000,deltLamb)

        x_nominal = int(np.floor(self.xcenter + 0.5))
        y_nominal = int(np.floor(self.ycenter + 0.5))
        dx = self.xcenter - x_nominal+0.5
        dy = self.ycenter - y_nominal+0.5
        offset = galsim.PositionD(dx, dy)

        # print(skybg)
        # print(specConfile)
        # print(throughput_f)

        # plt.figure()
        # plt.plot(sed['WAVELENGTH'], sed['FLUX'])

        gal = galsim.Sersic(s_n, half_light_radius=re)

        gal_pa = pa * galsim.degrees
        gal_ell = gal.shear(q=q_ell, beta=gal_pa)

        conv_gal = galsim.Convolve([gal_ell,self.psf])


        stamp = conv_gal.drawImage(wcs=galsim.PixelScale(self.p_size), offset=offset)*self.t*self.expNum*math.pi*(self.aper/2)*(self.aper/2)
        stamp.setOrigin(0,0)
        origin_star = [y_nominal - (stamp.center.y - stamp.ymin),
                                x_nominal - (stamp.center.x - stamp.xmin)]


        origin_star = [y_nominal - (stamp.center.y - stamp.ymin),
                                x_nominal - (stamp.center.x - stamp.xmin)]

        sdp = SpecDisperser.SpecDisperser(orig_img=stamp, xcenter=x_nominal,
                                            ycenter=y_nominal, origin=origin_star,
                                            tar_spec=sed,
                                            conf=specConfile,
                                            isAlongY=0, deltLamb = deltLamb/2.)

        spec_orders = sdp.compute_spec_orders()
        
        thp = Table.read(throughput_f)
        thp_i = interpolate.interp1d(thp['WAVELENGTH'], thp['SENSITIVITY'])

        Aimg_orig = spec_orders[self.beam][0]
        Aimg_ = Aimg_orig

        Aimg_ = Aimg_ + (self.skybg + self.dark)*self.t*self.expNum

        np.random.seed(int(time.time()))
        Aimg_ = np.random.poisson(Aimg_)
        for i in np.arange(self.expNum):
            Aimg_ = self.addReadoutNois(img = Aimg_, readout = self.readout)

        Aimg = Aimg_ - (self.skybg + self.dark)*self.t*self.expNum


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
        
        f1 = spec_orders[self.beam][0][y_cent_pos-i:y_cent_pos+i+1].sum(0)
        f2 = spec_orders[self.beam][0].sum(0)
        ratio_vec = np.zeros_like(f1)
        nozero_flag = f2 != 0
        
        ratio_vec[nozero_flag] = f1[nozero_flag]/f2[nozero_flag]
        # ratio_vec = spec_orders[self.beam][0][y_cent_pos-i:y_cent_pos+i+1].sum(0)/spec_orders[self.beam][0].sum(0)
        y_range = i
        # print(y_range, fluxRatio)
        y_len_pix = 2 * y_range + 1
        for i in range(sh[1]):
            spec_pix[i] = sum(Aimg[y_cent_pos-y_range:y_cent_pos+y_range+1, i])
            err2_pix[i] = sum(Aimg_orig[y_cent_pos-y_range:y_cent_pos+y_range+1, i]) + (self.skybg + self.dark)*self.t * y_len_pix * self.expNum + self.readout*self.readout * y_len_pix * self.expNum

        bRange = self.config.bandRanges[self.grating]
        wave_flux = np.zeros(wave_pix.shape[0])
        err_flux = np.zeros(wave_pix.shape[0])
        specRangeImg = []

        true_center = stamp.center + galsim.PositionD(self.xcenter-x_nominal, self.ycenter-y_nominal)
        wavePos_x = true_center.x + wave_pos - wave_pos[0]

        wavePos_x_interp = np.arange(int(wavePos_x[0]), int(wavePos_x[-1]))
        lam_trace = np.interp(wavePos_x_interp,wavePos_x,wave_pix)

        wave_flux = np.zeros(lam_trace.shape[0])
        err_flux = np.zeros(lam_trace.shape[0])



        for i in np.arange(1, lam_trace.shape[0] - 1):
            w = lam_trace[i]
            wave2pix_pos=wavePos_x_interp[i]

            if (bRange[0] <= w <= bRange[1]):
                thp_w = thp_i(w)
                deltW = np.abs(w - lam_trace[i - 1]) / 2 + np.abs(lam_trace[i + 1] - w) / 2
                f = spec_pix[wave2pix_pos]
                f_ratio = ratio_vec[wave2pix_pos]
                if f_ratio==0:
                    f_ratio=1
                f = f / self.t / thp_w / deltW /self.expNum/f_ratio
                err = err2_pix[wave2pix_pos]
                # err = err/ t / deltW
                err = np.sqrt(err)/ self.t / deltW/ thp_w /self.expNum/f_ratio
                specRangeImg.append(wave2pix_pos)
                # err = err / thp_w 
            else:
                f = 0
                err = 0

            wave_flux[i] = f
            err_flux[i] = err
        
        Aimg_cal = Aimg_[y_cent_pos-y_range:y_cent_pos+y_range+1, specRangeImg]
        ids = Aimg_cal > self.saturation

        #1. saturation pixel number, 2. total pixel number, 3 saturation ratio, 4.flux ratio in photo aperture,5.max value,6.min value
        saturePix = np.zeros(6)

        saturePix[0] = Aimg_cal[ids].shape[0]
        saturePix[1] = Aimg_cal.shape[0]*Aimg_cal.shape[1]
        saturePix[2] = saturePix[0]/saturePix[1]
        saturePix[3] = 1
        saturePix[4] = np.amax(Aimg_cal)
        saturePix[5] = np.amin(Aimg_cal)
        

        idx = (lam_trace >= bRange[0]-100)
        idx1 = (lam_trace[idx] <= bRange[1]+100)

        w_select = lam_trace[idx][idx1]
        f_select = wave_flux[idx][idx1]
        e_select =  err_flux[idx][idx1]
        lam_index = np.argsort(w_select)

        specTab = Table(np.array([w_select[lam_index], f_select[lam_index], e_select[lam_index]]).T,names=('WAVELENGTH', 'FLUX','ERR'))

        # spec_orig = np.loadtxt(sedFile)

        # plt.figure()
        # plt.plot(spec_orig[:,0], spec_orig[:,1])

        # plt.figure()
        # plt.errorbar(wave_pix[idx][idx1], wave_flux[idx][idx1],err_flux[idx][idx1])
        # plt.legend([self.sedFile])
        # # plt.plot(wave_pix[idx][idx1], wave_flux[idx][idx1])
        # plt.show()
        return specTab, Aimg, stamp.array, saturePix


    def generateSpec1dforStar(self,limitfluxratio = 0.8, deltLamb = 0.01):
        import matplotlib.pyplot as plt
        specConfile = self.config.conFiles[self.grating]

        throughput_f = self.config.senFisle[self.grating] + self.config.orderIDs[self.beam] + '.fits'

        sed = self.generateSEDfromFiles(self.sedFile,2500,10000,deltLamb)

        x_nominal = int(np.floor(self.xcenter + 0.5))
        y_nominal = int(np.floor(self.ycenter + 0.5))
        dx = self.xcenter - x_nominal+0.5
        dy = self.ycenter - y_nominal+0.5
        offset = galsim.PositionD(dx, dy)

        star = galsim.DeltaFunction()
            # star = star.withFlux(tel.pupil_area * exptime)
        star = galsim.Convolve(self.psf, star)
        stamp = star.drawImage(wcs=galsim.PixelScale(self.p_size), offset=offset,nx=100, ny=100)*self.t*self.expNum*math.pi*(self.aper/2)*(self.aper/2)
        stamp.setOrigin(0,0)
        # print(stamp.center)
        # kk = np.where(stamp.array == np.amax(stamp.array))
        # print(kk[0][0], kk[1][0])
        # plt.figure()
        # plt.imshow(stamp.array)

        # ttt = np.sum(stamp.array)
        # stamp.array[:,:]=0
        # stamp.array[stamp.center.y, stamp.center.x] = ttt
        


        # stamp = self.psf.drawImage(wcs=galsim.PixelScale(self.p_size))*self.t*self.expNum*math.pi*(self.aper/2)*(self.aper/2)

        origin_star = [y_nominal - (stamp.center.y - stamp.ymin),
                                x_nominal - (stamp.center.x - stamp.xmin)]

        sdp = SpecDisperser.SpecDisperser(orig_img=stamp, xcenter=x_nominal,
                                            ycenter=y_nominal, origin=origin_star,
                                            tar_spec=sed,
                                            conf=specConfile,
                                            isAlongY=0,deltLamb = deltLamb/2.)

        spec_orders = sdp.compute_spec_orders()
        
        thp = Table.read(throughput_f)
        thp_i = interpolate.interp1d(thp['WAVELENGTH'], thp['SENSITIVITY'])

        Aimg_orig = spec_orders[self.beam][0]
        Aimg_ = Aimg_orig

        Aimg_ = Aimg_ + (self.skybg + self.dark)*self.t*self.expNum
        np.random.seed(int(time.time()))
        Aimg_ = np.random.poisson(Aimg_)
        for i in np.arange(self.expNum):
            Aimg_ = self.addReadoutNois(img = Aimg_, readout = self.readout)

        Aimg = Aimg_ - (self.skybg + self.dark)*self.t*self.expNum


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
        f1 = spec_orders[self.beam][0][y_cent_pos-i:y_cent_pos+i+1].sum(0)
        f2 = spec_orders[self.beam][0].sum(0)
        ratio_vec = np.zeros_like(f1)
        nozero_flag = f2 != 0
        
        ratio_vec[nozero_flag] = f1[nozero_flag]/f2[nozero_flag]
        # ratio_vec = spec_orders[self.beam][0][y_cent_pos-i:y_cent_pos+i+1].sum(0)/spec_orders[self.beam][0].sum(0)
        y_range = i
        # print(y_range, fluxRatio)
        y_len_pix = 2 * y_range + 1
        for i in range(sh[1]):
            spec_pix[i] = sum(Aimg[y_cent_pos-y_range:y_cent_pos+y_range+1, i])
            err2_pix[i] = sum(Aimg_orig[y_cent_pos-y_range:y_cent_pos+y_range+1, i]) + (self.skybg + self.dark)*self.t * y_len_pix * self.expNum + self.readout*self.readout * y_len_pix * self.expNum

        bRange = self.config.bandRanges[self.grating]
        

        specRangeImg = []

        true_center = stamp.center + galsim.PositionD(self.xcenter-x_nominal, self.ycenter-y_nominal)
        wavePos_x = true_center.x + wave_pos - wave_pos[0]

        wavePos_x_interp = np.arange(int(wavePos_x[0]), int(wavePos_x[-1]))
        lam_trace = np.interp(wavePos_x_interp,wavePos_x,wave_pix)

        wave_flux = np.zeros(lam_trace.shape[0])
        err_flux = np.zeros(lam_trace.shape[0])

        for i in np.arange(1, lam_trace.shape[0] - 1):
            w = lam_trace[i]
            wave2pix_pos=wavePos_x_interp[i]

            if (bRange[0] <= w <= bRange[1]):
                thp_w = thp_i(w)
                deltW = np.abs(w - lam_trace[i - 1]) / 2 + np.abs(lam_trace[i + 1] - w) / 2
                f = spec_pix[wave2pix_pos]
                f_ratio = ratio_vec[wave2pix_pos]
                if f_ratio==0:
                    f_ratio=1
                f = f / self.t / thp_w / deltW /self.expNum/f_ratio
                err = err2_pix[wave2pix_pos]
                # err = err/ t / deltW
                err = np.sqrt(err)/ self.t / deltW/ thp_w /self.expNum/f_ratio
                specRangeImg.append(wave2pix_pos)
                # err = err / thp_w 
            else:
                f = 0
                err = 0

            wave_flux[i] = f
            err_flux[i] = err

        Aimg_cal = Aimg_[y_cent_pos-y_range:y_cent_pos+y_range+1, specRangeImg]
        ids = Aimg_cal > self.saturation
        
        #1. saturation pixel number, 2. total pixel number, 3 saturation ratio, 4.flux ratio in photo aperture,5.max value,6.min value
        saturePix = np.zeros(6)

        saturePix[0] = Aimg_cal[ids].shape[0]
        saturePix[1] = Aimg_cal.shape[0]*Aimg_cal.shape[1]
        saturePix[2] = saturePix[0]/saturePix[1]
        saturePix[3] = 1
        saturePix[4] = np.amax(Aimg_cal)
        saturePix[5] = np.amin(Aimg_cal)
        
        idx = (lam_trace >= bRange[0]-100)
        idx1 = (lam_trace[idx] <= bRange[1]+100)

        w_select = lam_trace[idx][idx1]
        f_select = wave_flux[idx][idx1]
        e_select =  err_flux[idx][idx1]
        lam_index = np.argsort(w_select)

        specTab = Table(np.array([w_select[lam_index], f_select[lam_index], e_select[lam_index]]).T,names=('WAVELENGTH', 'FLUX','ERR'))

        # spec_orig = np.loadtxt(sedFile)

        # plt.figure()
        # plt.plot(spec_orig[:,0], spec_orig[:,1])

        # plt.figure()
        # plt.errorbar(wave_pix[idx][idx1], wave_flux[idx][idx1],err_flux[idx][idx1])
        # plt.legend([self.sedFile])
        # # plt.plot(wave_pix[idx][idx1], wave_flux[idx][idx1])
        # plt.show()
        return specTab, Aimg, stamp.array, saturePix
    
    def generateSpec1dforInputImg(self, img = None,img_pixel_scale = 0.06, limitfluxratio=0.9,deltLamb = 0.01, pixel_size = 0.074):

        specConfile = self.config.conFiles[self.grating]

        throughput_f = self.config.senFisle[self.grating] + self.config.orderIDs[self.beam] + '.fits'

        sed = self.generateSEDfromFiles(self.sedFile,2500,10000,deltLamb)

        x_nominal = int(np.floor(self.xcenter + 0.5))
        y_nominal = int(np.floor(self.ycenter + 0.5))
        dx = self.xcenter - x_nominal+0.5
        dy = self.ycenter - y_nominal+0.5
        offset = galsim.PositionD(dx, dy)

        folding_threshold=5.e-3
        if img is None:
            print("ERROR: input Image Error")
            return
        img = img/img.sum()
        gal_img = galsim.ImageF(img, scale=img_pixel_scale)
        gsp = galsim.GSParams(folding_threshold=folding_threshold)
   
        gal = galsim.InterpolatedImage(gal_img, gsparams=gsp)
            # if g_order in ['C','D','E']:
            #     add_psf  = galsim.Gaussian(sigma=contam_order_sigma[g_order], flux=1.0)
            #     self.psf = galsim.Convolve(self.psf, add_psf)
        wcs_in = galsim.PixelScale(img_pixel_scale)
        wcs = galsim.PixelScale(pixel_size)
        gal = wcs.toWorld(wcs_in.toImage(gal))

        # print(skybg)
        # print(specConfile)
        # print(throughput_f)

        # plt.figure()
        # plt.plot(sed['WAVELENGTH'], sed['FLUX'])

        # gal = galsim.Sersic(s_n, half_light_radius=re)

        # gal_pa = pa * galsim.degrees
        # gal_ell = gal.shear(q=q_ell, beta=gal_pa)

        conv_gal = galsim.Convolve([gal,self.psf])


        stamp = conv_gal.drawImage(wcs=galsim.PixelScale(self.p_size), offset=offset)*self.t*self.expNum*math.pi*(self.aper/2)*(self.aper/2)
        stamp.setOrigin(0,0)
        t_center = photutils.centroids.centroid_1dg(stamp.array)
        stamp.setCenter(t_center[0],t_center[1])

        origin_star = [y_nominal - (stamp.center.y - stamp.ymin),
                                x_nominal - (stamp.center.x - stamp.xmin)]


        origin_star = [y_nominal - (stamp.center.y - stamp.ymin),
                                x_nominal - (stamp.center.x - stamp.xmin)]

        sdp = SpecDisperser.SpecDisperser(orig_img=stamp, xcenter=x_nominal,
                                            ycenter=y_nominal, origin=origin_star,
                                            tar_spec=sed,
                                            conf=specConfile,
                                            isAlongY=0, deltLamb = deltLamb/2.)

        spec_orders = sdp.compute_spec_orders()
        
        thp = Table.read(throughput_f)
        thp_i = interpolate.interp1d(thp['WAVELENGTH'], thp['SENSITIVITY'])

        Aimg_orig = spec_orders[self.beam][0]
        Aimg_ = Aimg_orig

        Aimg_ = Aimg_ + (self.skybg + self.dark)*self.t*self.expNum

        np.random.seed(int(time.time()))
        Aimg_ = np.random.poisson(Aimg_)
        for i in np.arange(self.expNum):
            Aimg_ = self.addReadoutNois(img = Aimg_, readout = self.readout)

        Aimg = Aimg_ - (self.skybg + self.dark)*self.t*self.expNum


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
        
        f1 = spec_orders[self.beam][0][y_cent_pos-i:y_cent_pos+i+1].sum(0)
        f2 = spec_orders[self.beam][0].sum(0)
        ratio_vec = np.zeros_like(f1)
        nozero_flag = f2 != 0
        
        ratio_vec[nozero_flag] = f1[nozero_flag]/f2[nozero_flag]
        # ratio_vec = spec_orders[self.beam][0][y_cent_pos-i:y_cent_pos+i+1].sum(0)/spec_orders[self.beam][0].sum(0)
        y_range = i
        # print(y_range, fluxRatio)
        y_len_pix = 2 * y_range + 1
        for i in range(sh[1]):
            spec_pix[i] = sum(Aimg[y_cent_pos-y_range:y_cent_pos+y_range+1, i])
            err2_pix[i] = sum(Aimg_orig[y_cent_pos-y_range:y_cent_pos+y_range+1, i]) + (self.skybg + self.dark)*self.t * y_len_pix * self.expNum + self.readout*self.readout * y_len_pix * self.expNum

        bRange = self.config.bandRanges[self.grating]
        wave_flux = np.zeros(wave_pix.shape[0])
        err_flux = np.zeros(wave_pix.shape[0])
        specRangeImg = []

        true_center = stamp.center + galsim.PositionD(self.xcenter-x_nominal, self.ycenter-y_nominal)
        wavePos_x = true_center.x + wave_pos - wave_pos[0]

        wavePos_x_interp = np.arange(int(wavePos_x[0]), int(wavePos_x[-1]))
        lam_trace = np.interp(wavePos_x_interp,wavePos_x,wave_pix)

        wave_flux = np.zeros(lam_trace.shape[0])
        err_flux = np.zeros(lam_trace.shape[0])



        for i in np.arange(1, lam_trace.shape[0] - 1):
            w = lam_trace[i]
            wave2pix_pos=wavePos_x_interp[i]

            if (bRange[0] <= w <= bRange[1]):
                thp_w = thp_i(w)
                deltW = np.abs(w - lam_trace[i - 1]) / 2 + np.abs(lam_trace[i + 1] - w) / 2
                f = spec_pix[wave2pix_pos]
                f_ratio = ratio_vec[wave2pix_pos]
                if f_ratio==0:
                    f_ratio=1
                f = f / self.t / thp_w / deltW /self.expNum/f_ratio
                err = err2_pix[wave2pix_pos]
                # err = err/ t / deltW
                err = np.sqrt(err)/ self.t / deltW/ thp_w /self.expNum/f_ratio
                specRangeImg.append(wave2pix_pos)
                # err = err / thp_w 
            else:
                f = 0
                err = 0

            wave_flux[i] = f
            err_flux[i] = err
        
        Aimg_cal = Aimg_[y_cent_pos-y_range:y_cent_pos+y_range+1, specRangeImg]
        ids = Aimg_cal > self.saturation

        #1. saturation pixel number, 2. total pixel number, 3 saturation ratio, 4.flux ratio in photo aperture,5.max value,6.min value
        saturePix = np.zeros(6)

        saturePix[0] = Aimg_cal[ids].shape[0]
        saturePix[1] = Aimg_cal.shape[0]*Aimg_cal.shape[1]
        saturePix[2] = saturePix[0]/saturePix[1]
        saturePix[3] = 1
        saturePix[4] = np.amax(Aimg_cal)
        saturePix[5] = np.amin(Aimg_cal)
        

        idx = (lam_trace >= bRange[0]-100)
        idx1 = (lam_trace[idx] <= bRange[1]+100)

        w_select = lam_trace[idx][idx1]
        f_select = wave_flux[idx][idx1]
        e_select =  err_flux[idx][idx1]
        lam_index = np.argsort(w_select)

        specTab = Table(np.array([w_select[lam_index], f_select[lam_index], e_select[lam_index]]).T,names=('WAVELENGTH', 'FLUX','ERR'))

        # spec_orig = np.loadtxt(sedFile)

        # plt.figure()
        # plt.plot(spec_orig[:,0], spec_orig[:,1])

        # plt.figure()
        # plt.errorbar(wave_pix[idx][idx1], wave_flux[idx][idx1],err_flux[idx][idx1])
        # plt.legend([self.sedFile])
        # # plt.plot(wave_pix[idx][idx1], wave_flux[idx][idx1])
        # plt.show()
        return specTab, Aimg, stamp.array, saturePix




    def addReadoutNois(self, img = None, readout = 5):
        random.seed(time.time())
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i,j] += round(random.gauss(mu = 0, sigma = readout))

        return img

