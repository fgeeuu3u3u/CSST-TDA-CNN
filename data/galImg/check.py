
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

files = [
    '/media/zhao/A0EE585EEE582F32/csst_msc_sim/sls_1d_spec/data/galImg/gal1.fits',
    '/media/zhao/A0EE585EEE582F32/csst_msc_sim/sls_1d_spec/data/galImg/gal2.fits',
    '/media/zhao/A0EE585EEE582F32/csst_msc_sim/sls_1d_spec/data/galImg/gal.fits'
]

for f in files:
    with fits.open(f) as hdul:
        data = hdul[0].data
        print(f"{f}: shape={data.shape}, dtype={data.dtype}, min={data.min():.3f}, max={data.max():.3f}, mean={data.mean():.3f}")
        plt.figure()
        plt.imshow(data, cmap='gray', origin='lower')
        plt.title(f)
        plt.colorbar()
plt.show()
