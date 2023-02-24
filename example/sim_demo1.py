'''
Author: xin zhangxinbjfu@gmail.com
Date: 2022-08-18 23:13:26
LastEditors: xin zhangxinbjfu@gmail.com
LastEditTime: 2023-02-24 01:05:33
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
    specG = SpecGenerator(sedFn = sedFn, grating = 'GI', beam = 'A', aper = 2.0, xcenter = 2000.,ycenter = 5000., p_size = 0.074, psf = psf, skybg = 0.3, dark = 0.02, readout = 5, t = 150., expNum = 1,config = config)
    specTab, specImg, img, satPix=specG.generateSpec1dforStar()

    # specG = SpecGenerator(sedFn = sedFn, grating = 'GI', beam = 'A', aper = 2.0, xcenter = 2000.,ycenter = 5000, p_size = 0.074, psf = psf, skybg = 0.3, dark = 0.02, readout = 5, t = 150., expNum = 1,config = config)
    # specTab, specImg, img, satPix=specG.generateSpec1dforGal(s_n = 1.0, re = 0.5, pa = 90,q_ell = 1.0,limitfluxratio=0.8)

    fits.writeto("specImg.fits",specImg,overwrite=True)
    fits.writeto("DImg.fits",img,overwrite=True)
    specTab.write("specTab.fits",overwrite=True)

    
