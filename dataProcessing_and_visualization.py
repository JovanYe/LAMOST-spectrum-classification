"""

Editor : jovan_ye

This is a temporary script file,aim is to process data and visualization.
take note of that the pytoch vision is gpu_11.3,while cpu vision is also okay at this program.

"""
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from laspec.normalization import normalize_spectrum_spline
from os import listdir


def readTrainData(filepath, binwidth=30, p=1E-9, niter=1, issave_pic=False):

    c = {0:'GALAXY', 1:'QSO', 2:'STAR'}
    wavelength = np.linspace(3900, 9000, 3000)

    filelist = listdir(filepath)

    GALAXY_flux = []
    QSO_flux = []
    STAR_flux = []

    for fn in filelist:
        hdulist = fits.open(filepath + fn)
        for (flux, inf) in zip(hdulist[0].data, hdulist[1].data):
            objid = inf['objid']
            label = inf['label']
            print(objid)
            # sp_ = normalization(flux)
            fnorm, fcont = normalize_spectrum_spline(wavelength, flux, binwidth=binwidth, p=p, niter=niter)
            fnorm_ = fnorm - 1
            fnorm_[fnorm_ < -1] = -1
            fnorm_[fnorm_ > 1] = 1

            if(issave_pic):
                # path =  filename.replace('data','img').replace('.csv','')
                # if(not os.path.exists(path)):
                #     os.mkdir(path)

                plt.figure(figsize=(16, 6))
                # plt.plot(wavelength, sp_, color='black', linestyle='solid', label='sp0')
                # plt.plot(wavelength, fcont, color='red', linestyle='solid', label='fcont')
                plt.plot(wavelength, fnorm_, color='black', linestyle='solid', label='fnorm')
                plt.title(f'ObjID={str(objid)}, Class={c[label]}')

                plt.xlabel('wavelength ({})'.format(f'$\AA$'))
                plt.ylabel('Normalization Flux')
                plt.legend(loc='lower right')
                # plt.savefig(path+'/{}.png'.format(id))
                plt.show()
                # plt.close()

            if c[label] == c[0]:
                GALAXY_flux.append(fnorm_)
            if c[label] == c[1]:
                QSO_flux.append(fnorm_)
            if c[label] == c[2]:
                STAR_flux.append(fnorm_)

    print(len(GALAXY_flux), len(QSO_flux), len(STAR_flux))

    return np.array(GALAXY_flux, dtype=np.float64), np.array(QSO_flux, dtype=np.float64), np.array(STAR_flux, dtype=np.float64)

def sample_expansion(sample, size=3500):
    if len(sample) < size:
        sample_diff = size - len(sample)
        indices_to_duplicate = np.random.choice(range(len(sample)), size=sample_diff, replace=True)  # 重采样
        sample_oversampled = np.vstack((sample, sample[indices_to_duplicate]))
    return sample_oversampled

if __name__ == "__main__":

    filepath = "./data/train_data/"

    GALAXY_flux, QSO_flux, STAR_flux = readTrainData(filepath)

    np.save("./data/train_data_norm/GALAXY_flux.npy", GALAXY_flux)
    np.save("./data/train_data_norm/QSO_flux.npy", QSO_flux)
    np.save("./data/train_data_norm/STAR_flux.npy", STAR_flux)



