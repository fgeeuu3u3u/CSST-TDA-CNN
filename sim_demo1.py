'''
Author: xin zhangxinbjfu@gmail.com
Date: 2022-05-18 08:37:51
LastEditors: xin zhangxinbjfu@gmail.com
LastEditTime: 2022-05-20 08:43:13
FilePath: /undefined/Users/zhangxin/Work/SlitlessSim/sls_lit_demo/example/sim_demo1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from SpecGen.SpecGenerator import SpecGenerator
from SpecGen.Config import Config

import galsim


if __name__ == '__main__':


    dataDir = 'data/'
    config = Config(dataDir = dataDir)
    sedFn = dataDir + 'sed/sed_44575.txt'
    psf = galsim.Gaussian(fwhm=0.39)
    specG = SpecGenerator(sedFn = sedFn, grating = 'GI', beam = 'A', aper = 2.0, xcenter = 5000,ycenter = 5000, p_size = 0.074, psf = psf, skybg = 0.3, dark = 0.02, readout = 5, t = 150, expNum = 1,config = config)
    specTab, specImg, img, fluxRa=specG.generateSpec1dforStar()

    
