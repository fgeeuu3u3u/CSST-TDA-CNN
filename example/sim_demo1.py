
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
    specG = SpecGenerator(sedFn = sedFn, grating = 'GI', beam = 'A', aper = 2.0, xcenter = 2000,ycenter = 5000, p_size = 0.074, psf = psf, skybg = 0.3, dark = 0.02, readout = 5, t = 150, expNum = 1,config = config)
    specTab, specImg, img, satPix=specG.generateSpec1dforStar()

    fits.writeto("specImg.fits",specImg,overwrite=True)
    fits.writeto("DImg.fits",img,overwrite=True)
    specTab.write("specTab.fits",overwrite=True)

    
