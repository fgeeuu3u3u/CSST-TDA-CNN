'''
Author: xin zhangxinbjfu@gmail.com
Date: 2022-08-18 23:13:26
LastEditors: Zhang Xin zhangx@bao.ac.cn
LastEditTime: 2024-08-28 16:50:24
FilePath: /undefined/Users/zhangxin/Work/SlitlessSim/sls_1d_sim/sls_1d_gitlab/sls_1d_spec/example/sim_demo1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from SpecGen.SpecGenerator import SpecGenerator
from SpecGen.Config import Config

from astropy.io import fits
from astropy.table import Table

import galsim


if __name__ == '__main__':


    dataDir = '../data/'
    config = Config(dataDir = dataDir)
    sedFn = dataDir + 'sed/sed_44575.txt'
    psf = galsim.Gaussian(fwhm=0.39)

    # galaxy sersic model
    # specG = SpecGenerator(sedFn = sedFn, grating = 'GI', beam = 'A', aper = 2.0, xcenter = 2000.,ycenter = 5000., p_size = 0.074, psf = psf, skybg = 0.3, dark = 0.02, readout = 5, t = 150., expNum = 1,config = config)
    # specTab, specImg, img, satPix=specG.generateSpec1dforStar()

    #star
    # specG = SpecGenerator(sedFn = sedFn, grating = 'GI', beam = 'A', aper = 2.0, xcenter = 2000.,ycenter = 5000, p_size = 0.074, psf = psf, skybg = 0.3, dark = 0.02, readout = 5, t = 150., expNum = 1,config = config)
    # specTab, specImg, img, satPix=specG.generateSpec1dforGal(s_n = 1.0, re = 0.5, pa = 90,q_ell = 1.0,limitfluxratio=0.8)

    # galaxy input image
    galfn = dataDir + "galImg/gal2.fits"
    galimg = fits.getdata(galfn)
    specG = SpecGenerator(sedFn = sedFn, grating = 'GV', beam = 'A', aper = 2.0, xcenter = 2000.,ycenter = 5000., p_size = 0.074, psf = psf, skybg = 0.3, dark = 0.02, readout = 5, t = 1500., expNum = 1,config = config)
    specTab, specImg, img, satPix=specG.generateSpec1dforInputImg(img = galimg,img_pixel_scale = 0.06, limitfluxratio=0.9,deltLamb = 0.01, pixel_size = 0.074)

    fits.writeto("specImg.fits",specImg,overwrite=True)
    fits.writeto("DImg.fits",img,overwrite=True)
    specTab.write("specTab.fits",overwrite=True)

    
