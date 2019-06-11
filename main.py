from lion import main as lionmain
import tdpy
from tdpy.util import summgene

import numpy as np

import time as timemodu

import pickle

from sklearn.manifold import TSNE
from sklearn import manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
            
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

#from skimage import data
#from skimage.morphology import disk
#from skimage.filters.rank import median

#import pyod
#from pyod.models.abod import ABOD
#from pyod.models.cblof import CBLOF
#from pyod.models.feature_bagging import FeatureBagging
#from pyod.models.hbos import HBOS
#from pyod.models.iforest import IForest
#from pyod.models.knn import KNN
#from pyod.models.lof import LOF
#from pyod.models.loci import LOCI
#from pyod.models.mcd import MCD
#from pyod.models.ocsvm import OCSVM
#from pyod.models.pca import PCA
#from pyod.models.sos import SOS
#from pyod.models.lscp import LSCP

import scipy.signal

import os, sys, datetime, fnmatch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astroquery.mast import Catalogs
import astroquery

import astropy
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits
import astropy.time
from astropy.coordinates import SkyCoord

import multiprocessing


def plot_embe(gdat, lcurflat, X_embedded, strg, titl):
    
    X_embedded = (X_embedded - np.amin(X_embedded, 0)) / (np.amax(X_embedded, 0) - np.amin(X_embedded, 0))

    figr, axis = plt.subplots(figsize=(12, 12))
    axis.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, marker='x', color='r', lw=0.5)
    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(X_embedded.shape[0]):
        dist = np.sum((X_embedded[i] - shown_images) ** 2, 1)
        if np.min(dist) < 1e-3:
            continue
        shown_images = np.r_[shown_images, [X_embedded[i]]]
        axins3 = inset_axes(axis, width="100%", height="100%", \
                        bbox_to_anchor=(X_embedded[i, 0] - 0.02, X_embedded[i, 1] - 0.02, .04, .04), bbox_transform=axis.transData, loc='center', borderpad=0)
        axins3.plot(gdat.time, lcurflat[i, :], alpha=0.5, color='g')
        #axins3.set_ylim([0, 2])
        axins3.text(X_embedded[i, 0], X_embedded[i, 1] + 0.02, '%g %g' % (np.amin(lcurflat[i, :]), np.amax(lcurflat[i, :])), fontsize=12) 
        axins3.axis('off')
    axis.set_title(titl)
    plt.tight_layout()
    path = gdat.pathdata + '%s_%s.pdf' % (strg, gdat.strgcntp)
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()
    
            
def plot_imag(gdat, cntp, path, cbar='Greys_r', strgtitl='', boolresi=False, xposoffs=None, yposoffs=None, indxpixlcolr=None):
    
    if gdat.cntpscaltype == 'asnh':
        cntp = np.arcsinh(cntp)
   
    if boolresi:
        vmin = gdat.vmincntpresi
        vmax = gdat.vmaxcntpresi
        cbar = 'PuOr'
    else:
        vmin = gdat.vmincntpdata
        vmax = gdat.vmaxcntpdata
    
    if vmin <= 0. and not boolresi:
        print 'Warning! Minimum of the image went negative!'
    
    figr, axis = plt.subplots(figsize=(12, 12))
    imag = axis.imshow(cntp, origin='lower', interpolation='nearest', cmap=cbar, vmin=vmin, vmax=vmax)
    
    axis.scatter(gdat.catlrefr[0]['xpos'], gdat.catlrefr[0]['ypos'], alpha=1., s=30, color='b')
    axis.scatter(gdat.catlrefr[0]['xpos'][0], gdat.catlrefr[0]['ypos'][0], alpha=1., color='g', s=30)
    #axis.scatter(gdat.catlrefr[0]['xpos'][1:gdat.numbrefrfitt], gdat.catlrefr[0]['ypos'][1:gdat.numbrefrfitt], alpha=1., color='y', s=30)
    if xposoffs is not None:
        axis.scatter(gdat.catlrefr[0]['xpos'] + xposoffs, gdat.catlrefr[0]['ypos'] + yposoffs, alpha=1., s=20, color='r')
    
    if indxpixlcolr is not None:
        temp = np.zeros_like(cntp).flatten()
        temp[indxpixlcolr[-1]] = 1.
        temp = temp.reshape((gdat.numbside, gdat.numbside))
        alph = np.zeros_like(cntp).flatten()
        alph[indxpixlcolr[-1]] = 1.
        alph = alph.reshape((gdat.numbside, gdat.numbside))
        alph = np.copy(temp)
        axis.imshow(temp, origin='lower', interpolation='nearest', alpha=0.5)
    axis.set_position([0.1, 0.1, 0.7, 0.7])
    axiscbar = figr.add_axes([0.9, 0.1, 0.05, 0.73]) 
    cbar = figr.colorbar(imag, cax=axiscbar) 
    if gdat.cntpscaltype == 'asnh':
        tick = cbar.get_ticks()
        tick = np.sinh(tick)
        labl = ['%d' % tick[k] for k in range(len(tick))]
        cbar.set_ticklabels(labl)
    
    #plt.tight_layout()
    
    print 'Writing to %s...' % path
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    

def retr_magt(gdat, cntp):
    
    magt = -2.5 * np.log10(cntp / 1.5e4 / gdat.listcade) + 10
    
    return magt


def work(isec=None, icam=None, iccd=None, pathfile=None, rasctarg=None, decltarg=None, datatype='obsd'):
    
    # construct global object
    gdat = tdpy.util.gdatstrt()
    
    gdat.datatype = datatype

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # paths
    ## read PCAT path environment variable
    gdat.pathdata = os.environ['TESSTRAN_DATA_PATH'] + '/'
    ## define paths
    #gdat.pathdataorig = '/pdo/qlp-data/orbit-%d/ffi/cam%d/ccd%d/FITS/' % (isec, icam, iccd)
    gdat.pathdataorig = gdat.pathdata + 'ffis/'
    gdat.pathdatafilt = gdat.pathdata + 'filt/'
    gdat.pathdatainit = gdat.pathdata + 'init/'
    gdat.pathdatainitimag = gdat.pathdatainit + 'imag/'
    gdat.pathdatainitanim = gdat.pathdatainit + 'anim/'
    gdat.pathdatacomm = gdat.pathdata + 'comm/'
    ## make folders 
    os.system('mkdir -p %s' % gdat.pathdatafilt)
     
    # check inputs
    np.set_printoptions(precision=3, linewidth=180)
    print 'Sector: %d' % isec
    print 'Camera: %d' % icam
    print 'CCD: %d' % iccd
    
    verbtype = 1
    
    print 'gdat.datatype'
    print gdat.datatype

    np.random.seed(45)

    # fix the seed
    if gdat.datatype == 'mock':
        numbsour = 1000
        numbsupn = 10
    
    if pathfile is None:
        # get list of paths where FFIs live
        listrtag = fnmatch.filter(os.listdir(gdat.pathdataorig), 'tess*-s%04d-%d-%d-*-s_ffic.fits' % (isec, icam, iccd))
        
        #listrtag = listrtag#[:20]

        if len(listrtag) == 0:
            raise Exception('Could not find any files.')
        numbtime = len(listrtag)
    
    else:
        listhdunrefr = fits.open(pathfile)
        time = listhdunrefr[1].data['TIME']
        numbtime = time.size
        imagmemo = np.swapaxes(np.swapaxes(listhdunrefr[1].data['FLUX'], 0, 1), 1, 2)
        listobjtwcss = WCS(listhdunrefr[1].header)
    
    print 'numbtime'
    print numbtime
    
    # settings
    ## parameters
    numbsidecorr = 1
    numbneigaper = 1
    numbstrd = 1
    
    ## plotting
    boolplotsinc = True
    
    gdat.offscorr = numbsidecorr / 2

    numbmemo = 10
    gdat.numbneigback = 8
    
    if pathfile is not None and gdat.datatype == 'mock':
        raise Exception('')

    if pathfile is None:
        numbsideyposfull = 2078
        numbsidexposfull = 2136
        numbpixloffsypos = 30
        numbpixloffsxpos = 44
        print 'Cutout input data are not provided. Will work on the FFIs.'
    else:
        numbsideyposfull = imagmemo.shape[0]
        numbsidexposfull = imagmemo.shape[1]
        numbpixloffsypos = 0
        numbpixloffsxpos = 0
        print 'Cutout input data are provided.'
        
    print 'numbsideyposfull'
    print numbsideyposfull
    print 'numbsidexposfull'
    print numbsidexposfull
    numbsideypos = numbsideyposfull - numbpixloffsypos
    numbsidexpos = numbsidexposfull - numbpixloffsxpos
    
    numbside = numbsideypos
    print 'numbside'
    print numbside
    
    numbsideback = 2 * gdat.numbneigback + 1
    indxtime = np.arange(numbtime)
    indxmemo = np.arange(numbmemo)
    
    gdat.numbsideedge = gdat.numbneigback + gdat.offscorr
    numbsidesrch = (numbside - gdat.numbsideedge) / numbmemo
    numbsideshft = numbsidesrch + gdat.numbsideedge
    numbsidememo = numbsidesrch + 2 * gdat.numbneigback + 2 * gdat.offscorr
    indxsidememo = np.arange(numbsidememo)
    
    if (numbside - gdat.numbsideedge) % numbmemo != 0:
        print 'numbside - gdat.numbsideedge'
        print numbside - gdat.numbsideedge
        print 'numbmemo'
        print numbmemo
        raise Exception('')

    print 'numbsidememo'
    print numbsidememo
    print 'numbsidesrch'
    print numbsidesrch
    
    numbdata = numbsidesrch**2
    indxdata = np.arange(numbdata)
    
    indxsidesrch = np.arange(numbsidesrch)
    numbsrch = numbsidesrch**2
    
    indxsideyposmemo, indxsidexposmemo = np.meshgrid(indxsidememo, indxsidememo)
    indxsideyposdata = indxsideyposmemo[gdat.numbsideedge:-gdat.numbsideedge, gdat.numbsideedge:-gdat.numbsideedge] - gdat.numbsideedge
    indxsidexposdata = indxsidexposmemo[gdat.numbsideedge:-gdat.numbsideedge, gdat.numbsideedge:-gdat.numbsideedge] - gdat.numbsideedge
    
    if numbdata != indxsideyposdata.size:
        print 'numbdata'
        print numbdata
        print 'indxsideyposdata'
        summgene(indxsideyposdata)
        raise Exception('')

    indxsideyposdatatemp = np.empty((numbsidesrch, numbsidesrch, numbsidecorr, numbsidecorr))
    indxsidexposdatatemp = np.empty((numbsidesrch, numbsidesrch, numbsidecorr, numbsidecorr))
    indxsideyposdatatemp[:, :, :, :] = indxsideyposdata[:, :, None, None]
    indxsidexposdatatemp[:, :, :, :] = indxsidexposdata[:, :, None, None]
    gdat.indxsideyposdataflat = indxsideyposdatatemp.flatten()
    gdat.indxsidexposdataflat = indxsidexposdatatemp.flatten()
    numbpixlmemo = numbsidememo**2
            
    if gdat.datatype == 'mock':
        listlabltrue = np.zeros(numbdata, dtype=int)
        numbinli = numbdata - numbsour
        numboutl = numbsour
    
    thrsrmss = 0.01
    thrsmaxm = 1.5
    thrsdiff = 0.5
    numbsideaper = 2 * numbneigaper + 1
    numbpixlaper = numbsideaper**2
    
    if float(numbside) % numbstrd != 0:
        raise Exception('')

    # grid of flux space
    minmproj = 0.1
    maxmproj = 2
    limtproj = [minmproj, maxmproj]
    arry = np.linspace(minmproj, maxmproj, 100)
    xx, yy = np.meshgrid(arry, arry)
    
    magtminm = 15.
    magtmaxm = 23.
    print 'numbtime'
    print numbtime
    print 'numbsideback'
    print numbsideback
    print 'numbsideaper'
    print numbsideaper
    print 'numbstrd'
    print numbstrd
    print 'numbsrch'
    print numbsrch
    
    indxsidememosrch = np.arange(gdat.numbneigback + gdat.offscorr, numbsidesrch + gdat.numbsideedge)
    
    liststrgpara = ['rmss', 'maxm', 'mean', 'diff']
    listlablpara = ['R', 'Max', 'Mean', 'Difference']
    dictpara = {}
    dictpara['rmss'] = np.zeros((numbsidememo, numbsidememo)) - 1.
    dictpara['maxm'] = np.zeros((numbsidememo, numbsidememo)) - 1.
    dictpara['mean'] = np.zeros((numbsidememo, numbsidememo)) - 1.
    dictpara['diff'] = np.zeros((numbsidememo, numbsidememo)) - 1.
    
    # i indexes the y-axis
    for i in indxmemo: 
        # j indexes the y-axis
        for j in indxmemo: 
    
            if (i != 0 or j != 0) and gdat.datatype == 'mock':
                continue

            print 'Memory region i=%d, j=%d' % (i, j)
            
            # plots
            ## file name string extension for image plots
            gdat.strgcntp = '%s_%02d%d%d_%d_%d' % (gdat.datatype, isec, icam, iccd, i, j)
           
            # determine the initial and final pixel indices of the FFI data to be copied
            indxsideyposdatainit = numbpixloffsypos + j * numbsidememo - gdat.numbsideedge
            indxsideyposdatafinl = numbpixloffsypos + (j + 1) * numbsidememo - gdat.numbsideedge
            indxsidexposdatainit = numbpixloffsxpos + i * numbsidememo - gdat.numbsideedge
            indxsidexposdatafinl = numbpixloffsxpos + (i + 1) * numbsidememo - gdat.numbsideedge
            if j == 0:
                indxsideyposdatainit += gdat.numbneigback + gdat.offscorr
                indxsideyposdatafinl += gdat.numbneigback + gdat.offscorr
            #if j == numbmemo - 1:
            #    indxsideyposdatainit -= gdat.numbneigback + gdat.offscorr
            #    indxsideyposdatafinl -= gdat.numbneigback + gdat.offscorr
            if i == 0:
                indxsidexposdatainit += gdat.numbneigback + gdat.offscorr
                indxsidexposdatafinl += gdat.numbneigback + gdat.offscorr
            #if i == numbmemo - 1:
            #    indxsidexposdatainit -= gdat.numbneigback + gdat.offscorr
            #    indxsidexposdatafinl -= gdat.numbneigback + gdat.offscorr
            
            print 'indxsideyposdatainit'
            print indxsideyposdatainit
            print 'indxsideyposdatafinl'
            print indxsideyposdatafinl
            print 'indxsidexposdatainit'
            print indxsidexposdatainit
            print 'indxsidexposdatafinl'
            print indxsidexposdatafinl
            print
            
            path = gdat.pathdataorig + listrtag[0]
            listhduncatl = fits.open(path)
            
            #if rasctarg is not None:
            #    posipixl = listobjtwcss[0].all_world2pix(np.array([[rasctarg, decltarg]]), 1)
            
            if pathfile is None:
                
                pathsavedata = gdat.pathdata + '%s_data.npz' % gdat.strgcntp
                if not os.path.exists(pathsavedata):

                    listobjtwcss = []
                    gdat.time = []
                    if gdat.datatype == 'obsd':
                        imagmemo = []
                    for t in range(len(listrtag)):
                        if t % 100 == 0 and gdat.datatype == 'obsd':
                            print 'Loading the image into memory, t = %d' % t
                        path = gdat.pathdataorig + listrtag[t]
                        listhdunrefr = fits.open(path, memmap=False)
                        
                        objtheadseco = listhdunrefr[1].header
                        listobjtwcss.append(WCS(objtheadseco))
                        
                        objtheadfrst = listhdunrefr[0].header
                        timetemp = (objtheadfrst['TSTOP'] + objtheadfrst['TSTART']) / 2
                        gdat.time.append(timetemp)
                        
                        if gdat.datatype == 'obsd':
                            hdundata = listhdunrefr[1].data
                            imagmemo.append(hdundata[indxsideyposdatainit:indxsideyposdatafinl, indxsidexposdatainit:indxsidexposdatafinl])
                        
                        listhdunrefr.close()
                    
                    if gdat.datatype == 'obsd':
                        imagmemo = np.stack(imagmemo, axis=-1)
            
                    objtfile = open(pathsavedata, 'wb')
                    pickle.dump([listobjtwcss, gdat.time, imagmemo], objtfile)
                else:
                    objtfile = open(pathsavedata, "rb" )
                    listobjtwcss, gdat.time, imagmemo = pickle.load(objtfile)

            if gdat.datatype == 'mock':
                timeexpo = 1440.
                # Data generation
                ## image
                arrytime = np.empty((2, numbtime))
                arrytime[:, :] = np.linspace(-0.5, 0.5, numbtime)[None, :]
                indxsour = np.arange(numbsour)
                indxtime = np.arange(numbtime)
                posiquat = 5e-2 * np.random.randn(2 * numbtime).reshape((2, numbtime)) + arrytime * 0.1
                gdat.trueypos = numbsidememo * np.random.random(numbsour)[None, :] + posiquat[0, :, None]
                gdat.truexpos = numbsidememo * np.random.random(numbsour)[None, :] + posiquat[1, :, None]
                
                imagmemo = np.ones((numbsidememo, numbsidememo, numbtime)) * 60.
                
                indxsideyposmemocent = (j + 1.5) * numbsideshft + numbpixloffsypos + gdat.numbsideedge
                indxsidexposmemocent = (i + 1.5) * numbsideshft + numbpixloffsxpos + gdat.numbsideedge
                posiskyy = listobjtwcss[0].all_pix2world(indxsideyposmemocent, indxsidexposmemocent, 0)
                strgsrch = '%g %g' % (posiskyy[0], posiskyy[1])
                try:
                    catalogData = Catalogs.query_region(strgsrch, radius='0.1m', catalog = "TIC")
                    if len(catalogData) > 0:
                        tici = int(catalogData[0]['ID'])
                        titl += ', TIC %d' % tici
                except:
                    pass

                # inject signal
                indxsupn = np.arange(numbsupn)
                truecntpsour = np.empty((numbtime, numbsour))
                truemagt = np.empty((numbtime, numbsour))
                gdat.indxsoursupn = np.random.choice(indxsour, size=numbsupn, replace=False)
                for n in indxsour:
                    if n in gdat.indxsoursupn:
                        timenorm = -0.5 + (gdat.time / np.amax(gdat.time)) + 2. * (np.random.random(1) - 0.5)
                        print 'n'
                        print n
                        print 'timenorm'
                        summgene(timenorm)

                        objtrand = scipy.stats.skewnorm(10.).pdf(timenorm)
                        objtrand /= np.amax(objtrand)
                        truemagt[:, n] = 10. + 6. * (2. - objtrand)
                        
                        print 'truemagt[:, n]'
                        print truemagt[:, n]
                        summgene(truemagt[:, n])
                        print 
                    else:
                        truemagt[:, n] = np.random.rand() * 5 + 15.
                    
                    truecntpsour[:, n] = 10**((20.424 - truemagt[:, n]) / 2.5)

                indxsideypossour = np.round(np.mean(gdat.trueypos, 0)).astype(int)
                indxsidexpossour = np.round(np.mean(gdat.truexpos, 0)).astype(int)
                
                sigmpsfn = 1.

                for k in indxsour:
                    for t in indxtime:
                        imagmemo[:, :, t] += 1. / np.sqrt(sigmpsfn**2 * (2. * np.pi)**2) * \
                         truecntpsour[t, k] * np.exp(-0.5 * ((indxsidexposmemo - gdat.truexpos[t, k])**2 + \
                                                                    (indxsideyposmemo - gdat.trueypos[t, k])**2) / sigmpsfn**2)
                
                indxsideypossour[np.where(indxsideypossour == numbsidememo)] = numbsidememo - 1
                indxsidexpossour[np.where(indxsidexpossour == numbsidememo)] = numbsidememo - 1
                indxsideypossour[np.where(indxsideypossour < 0)] = 0
                indxsidexpossour[np.where(indxsidexpossour < 0)] = 0
                
                indxsourinsd = np.where((indxsideypossour > gdat.numbneigback + gdat.offscorr) & \
                               (indxsidexpossour > gdat.numbneigback + gdat.offscorr) & (indxsideypossour < numbsidememo - gdat.numbsideedge) & \
                               (indxsidexpossour < numbsidememo - gdat.numbsideedge))[0]
                
                indxsupninsd = np.where((indxsideypossour[gdat.indxsoursupn] > gdat.numbneigback + gdat.offscorr) & \
                               (indxsidexpossour[gdat.indxsoursupn] > gdat.numbneigback + gdat.offscorr) & \
                               (indxsideypossour[gdat.indxsoursupn] < numbsidememo - gdat.numbsideedge) & \
                               (indxsidexpossour[gdat.indxsoursupn] < numbsidememo - gdat.numbsideedge))[0]
                
                indxdatasour = (indxsideypossour - gdat.numbsideedge) * numbsidesrch + indxsidexpossour - gdat.numbsideedge
                indxdatasupn = (indxsideypossour[gdat.indxsoursupn] - gdat.numbsideedge) * numbsidesrch + \
                                                                    indxsidexpossour[gdat.indxsoursupn] - gdat.numbsideedge
                
                indxdataback = np.setdiff1d(indxdata, indxdatasour)
                listlabltrue[indxdatasour[indxsourinsd]] = 1
                imagmemo *= timeexpo
                imagmemo = np.random.poisson(imagmemo).astype(float)

            # spatial median
            print 'Performing the spatial median filter...'
            #imagmemo = imagmemo - scipy.signal.medfilt(imagmemo, (11, 11, 1))

            # temporal median filter
            numbtimefilt = min(9, numbtime)
            print 'Performing the temporal median filter...'
            imagmemo = scipy.signal.medfilt(imagmemo, (1, 1, numbtimefilt))
            
            if imagmemo.shape[0] != numbsidememo or imagmemo.shape[1] != numbsidememo:
                print 'imagmemo'
                summgene(imagmemo)
                print 'numbsidememo'
                print numbsidememo
                raise Exception('')

            # rebin in time
            if numbtime > 30:
                print 'Rebinning in time...'
                numbtimeoldd = numbtime
                numbtime = 30
                numbtimebins = numbtimeoldd / numbtime
                imagmemoneww = np.zeros((numbsidememo, numbsidememo, numbtime)) - 1.
                timeneww = np.zeros(numbtime)
                for t in range(numbtime):
                    if t == numbtime - 1:
                        imagmemoneww[:, :, t] = np.mean(imagmemo[:, :, (numbtime-1)*numbtimebins:], axis=2)
                        timeneww[t] = np.mean(time[(numbtime-1)*numbtimebins:])
                    else:
                        imagmemoneww[:, :, t] = np.mean(imagmemo[:, :, t*numbtimebins:(t+1)*numbtimebins], axis=2)
                        timeneww[t] = np.mean(time[t*numbtimebins:(t+1)*numbtimebins])
                imagmemo = imagmemoneww
                gdat.time = timeneww
                indxtime = np.arange(numbtime)
                
            minmtime = np.amin(gdat.time)
            if not np.isfinite(imagmemo).all():
                raise Excepion('')

            ## RMS image
            figr, axis = plt.subplots(figsize=(12, 6))
            imagmemomedi = np.median(imagmemo, 2)
            cntptemp = np.std(imagmemo - imagmemomedi[:, :, None], 2) / imagmemomedi
            if boolplotsinc:
                cntptemp = np.arcsinh(cntptemp)
            objtimag = axis.imshow(cntptemp, interpolation='nearest', cmap='Reds')
            plt.colorbar(objtimag)
            plt.tight_layout()
            path = gdat.pathdata + 'cntpstdv_%s.pdf' % gdat.strgcntp
            print 'Writing to %s...' % path
            plt.savefig(path)
            plt.close()

            strgtype = 'tsne'

            if i == 0 and j == 0:
                lcurarry = np.empty((numbsidesrch, numbsidesrch, numbtime, numbsidecorr, numbsidecorr)) 
                medilcur = np.empty((numbsidesrch, numbsidesrch)) 
           
            if not np.isfinite(imagmemo).all():
                raise Exception('')

            # normalize by the temporal median
            imagmemo /= np.mean(imagmemo, 2)[:, :, None]
            #imagmemo /= np.std(imagmemo, 2)[:, :, None]

            if not np.isfinite(imagmemo).all():
                print 'np.where(imagmemomedi == 0)[0].size'
                print np.where(imagmemomedi == 0)[0].size
                print 'imagmemomedi'
                summgene(imagmemomedi)
                raise Exception('')

            lcuravgd = np.empty(numbtime)
            cntr = 0
            prevfrac = -1
            k = 0
                        
            for a in np.arange(numbsidecorr):
                for b in np.arange(numbsidecorr):
                    if numbsidecorr == 3:
                        offs = -1
                    else:
                        offs = 0
                    indx = gdat.numbneigback + gdat.offscorr + a + offs
                    lcurarry[:, :, :, a, b] = imagmemo[indx:indx+numbsidesrch, indx:indx+numbsidesrch, :]

            if False:
                # k indexes the y-axis
                while k < numbsidememo:
                    l = 0
                    # l indexes the y-axis
                    while l < numbsidememo:
                        frac = 5 * int(20. * float(cntr) / numbpixlmemo)
                        if prevfrac != frac:
                            print '%d%% completed' % frac
                            prevfrac = frac
                        if verbtype > 1:
                            print 'k l'
                            print k, l
                            print 'cntr'
                            print cntr
                        
                        if k * numbstrd - gdat.numbsideedge < 0:
                            l += 1
                            cntr += 1
                            continue
                        if l * numbstrd - gdat.numbsideedge < 0:
                            l += 1
                            cntr += 1
                            continue
                        if gdat.numbneigback + gdat.offscorr + l * numbstrd >= numbsidememo:
                            l += 1
                            cntr += 1
                            continue
                        if gdat.numbneigback + gdat.offscorr + k * numbstrd >= numbsidememo:
                            l += 1
                            cntr += 1
                            continue
                        
                        indxsideypostemp = k * numbstrd
                        indxsidexpostemp = l * numbstrd
                        if not np.isfinite(imagmemo).all():
                            raise Exception('')

                        if strgtype == 'tsne' or strgtype == 'tmpt':
                            pass
                        else:
                            indxsideyposaperinit = indxsideypostemp - numbneigaper
                            indxsidexposaperinit = indxsidexpostemp - numbneigaper
                            indxsideyposaperfinl = indxsideypostemp + numbneigaper + 1
                            indxsidexposaperfinl = indxsidexpostemp + numbneigaper + 1
                            indxsideyposbackinit = indxsideypostemp - gdat.numbneigback
                            indxsidexposbackinit = indxsidexpostemp - gdat.numbneigback
                            indxsideyposbackfinl = indxsideypostemp + gdat.numbneigback + 1
                            indxsidexposbackfinl = indxsidexpostemp + gdat.numbneigback + 1
                            
                            imagbackmedi = np.median(imagmemo[indxsideyposbackinit:indxsideyposbackfinl, \
                                                                    indxsidexposbackinit:indxsidexposbackfinl, :], axis=(0, 1))
                            for t in indxtime:
                                lcur[t] = np.sum(imagmemo[indxsideyposaperinit:indxsideyposaperfinl, indxsidexposaperinit:indxsidexposaperfinl, t]) - \
                                                                                                            imagbackmedi[t] * numbpixlaper
                            if not np.isfinite(lcur).all():
                                print 'imagbackmedi'
                                print imagbackmedi
                                raise Exception('')
                            
                            # normalize
                            meanlcur = np.mean(lcur)
                            lcur /= meanlcur
                            #lcurmedi = scipy.signal.medfilt(lcur, 11)
                            dictpara['mean'][k, l] = meanlcur
                            
                            lcurdiff = lcur - lcuravgd
                            indxtimediff = np.argsort(lcurdiff)[::-1]
                            for t in indxtimediff:
                                if t < 0.2 * numbtime or (t >= 0.5 * numbtime and t <= 0.7 * numbtime):
                                    continue
                                if lcurdiff[t] > thrsdiff and lcurdiff[t-1] > thrsdiff:
                                    break
                            indxtimediffaccp = t
                            dictpara['diff'][k, l] = lcurdiff[indxtimediffaccp]

                            # acceptance condition
                            boolgood = False
                            if dictpara['diff'][k, l] > thrsdiff:
                            #if dictpara['maxm'][k, l] > thrsmaxm and (lcurtest[indxtimemaxm+1] > thrsmaxm or lcurtest[indxtimemaxm-1] > thrsmaxm):
                                boolgood = True
                            
                            if abs(k - 50) < 4 and abs(l - 50) < 4:
                                boolgood = True
                                
                            if boolgood:# or (abs(k - 53) < 5 and abs(l - 53) < 5):
                                
                                indxsideyposaccp.append(k * numbstrd)
                                indxsidexposaccp.append(l * numbstrd)
                    
                                # plot
                                figr, axis = plt.subplots(figsize=(12, 6))
                                axis.plot(time, lcur, ls='', marker='o', markersize=3, label='Raw')
                                #axis.plot(time, lcurmedi, ls='', marker='o', markersize=3, label='Median')
                                #axis.plot(time, lcurtest, ls='', marker='o', markersize=3, label='Cleaned')
                                axis.plot(time, lcurdiff + 1., ls='', marker='o', markersize=3, label='Diff')
                                
                                axis.set_xlabel('Time [days]')
                                axis.set_ylabel('Relative Flux')
                                axis.legend()
                                
                                axis.axhline(thrsdiff + 1., ls='--', alpha=0.3, color='gray')
                                axis.axvline(time[int(0.2*numbtime)], ls='--', alpha=0.3, color='red')
                                axis.axvline(time[int(0.5*numbtime)], ls='--', alpha=0.3, color='red')
                                axis.axvline(time[int(0.7*numbtime)], ls='--', alpha=0.3, color='red')
                                
                                posisili = np.empty((1, 2))
                                posisili[0, 0] = indxsideyposdatainit + k
                                posisili[0, 1] = indxsidexposdatainit + l
                                
                                titl = 'Diff: %g' % (dictpara['diff'][k, l])
                                if gdat.datatype == 'obsd':
                                    if pathfile is None:
                                        posiskyy = listobjtwcss[t].all_pix2world(posisili, 0)
                                    else:
                                        posiskyy = listobjtwcss.all_pix2world(posisili, 0)
                                    rasc = posiskyy[:, 0]
                                    decl = posiskyy[:, 1]
                                    strgsrch = '%g %g' % (rasc, decl)
                                    titl += 'k = %d, l = %d' % (k, l)
                                    #catalogData = Catalogs.query_region(strgsrch, radius='0.1m', catalog = "TIC")
                                    #if len(catalogData) > 0:
                                    #    tici = int(catalogData[0]['ID'])
                                    #    titl += ', TIC %d' % tici
                                
                                    #axis.axvspan(minmtime, 1328, alpha=0.5, color='red')
                                    #axis.axvspan(1338, 1342, alpha=0.5, color='red')
                                    #axis.axvspan(1345, 1350, alpha=0.5, color='red')
                                
                                axis.set_title(titl)
                                plt.tight_layout()
                                path = gdat.pathdata + 'lcur_%s_%04d_%04d.pdf' % (gdat.strgcntp, k, l)
                                print 'Writing to %s...' % path
                                plt.savefig(path)
                                plt.close()
                        
                            if verbtype > 1:
                                print
                        
                        cntr += 1
                        l += 1
                    k += 1
    
            lcurflat = lcurarry.reshape((numbsidesrch**2, numbtime, numbsidecorr, numbsidecorr))
            lcurflat = lcurflat.reshape((numbsidesrch**2, numbtime * numbsidecorr**2))
            
            n_neighbors = 30
           
            X = lcurflat

            indxdata = np.arange(numbdata)
            
            outliers_fraction = 0.15
            
            # define outlier/anomaly detection methods to be compared
            listobjtalgoanom = [
                                #("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)), \
                                #("Isolation Forest", IsolationForest(contamination=outliers_fraction)), \
                                ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction))
                               ]
            
            numbmeth = len(listobjtalgoanom)
            indxmeth = np.arange(numbmeth)
            
            listindxsideyposaccp = []
            listindxsidexposaccp = []
            listscor = []
            listlablmodl = []
            
            numbtimeplotscat = min(6, numbtime)
            limt = [np.amin(X), np.amax(X)]
            
            c = 0
            print 'Running anomaly-detection algorithms...'
            for name, objtalgoanom in listobjtalgoanom:
                t0 = timemodu.time()
                print 'name'
                print name
                print 'c'
                print c

                objtalgoanom.fit(X)
                t1 = timemodu.time()
            
                # fit the data and tag outliers
                if name == 'Local Outlier Factor':
                    scor = objtalgoanom.negative_outlier_factor_
                else:
                    scor = objtalgoanom.decision_function(X)
                if name == "Local Outlier Factor":
                    lablmodl = np.zeros(numbdata)
                    lablmodl[np.where(scor < -2)[0]] = 1.
                else:
                    lablmodl = objtalgoanom.fit(X).predict(X)
                
                indxdataposi = np.where(lablmodl == 1)[0]
                indxdatanega = np.setdiff1d(indxdata, indxdataposi)
                numbposi = indxdataposi.size
                gdat.numbpositext = min(20, numbposi)

                indxsideyposaccp = indxdataposi // numbsidememo
                indxsidexposaccp = indxdataposi % numbsidememo
                
                listindxsideyposaccp.append(indxsideyposaccp)
                listindxsidexposaccp.append(indxsidexposaccp)
                listscor.append(scor)
                listlablmodl.append(lablmodl)
                
                gdat.indxdatascorsort = np.argsort(listscor[c])
                
                # make plots
                ## animation of image frames 
                vmin = np.amin(imagmemo)
                vmax = np.amax(imagmemo)
                numbtimeplot = min(10, numbtime)
                indxtimeplot = np.linspace(0., numbtime - 1., numbtimeplot).astype(int)
                for t in indxtimeplot:
                    figr, axis = plt.subplots(figsize=(12, 6))
                    cntptemp = imagmemo[:, :, t]
                    if boolplotsinc:
                        cntptemp = np.arcsinh(cntptemp)
                        vmintemp = np.arcsinh(vmin)
                        vmaxtemp = np.arcsinh(vmax)
                    objtimag = axis.imshow(cntptemp, interpolation='nearest', vmin=vmintemp, vmax=vmaxtemp, cmap='Greys_r')
                    plot_catl(gdat, axis)
                    plt.colorbar(objtimag)
                    plt.tight_layout()
                    path = gdat.pathdata + 'cntp_%s_%05d.pdf' % (gdat.strgcntp, t)
                    print 'Writing to %s...' % path
                    plt.savefig(path)
                    plt.close()
                os.system('convert %scntp_%s_*.pdf %scntp_%s.gif' % (gdat.pathdata, gdat.strgcntp, gdat.pathdata, gdat.strgcntp))
                ### delete the frame plots
                path = gdat.pathdata + 'cntp_%s_*.pdf' % (gdat.strgcntp)
                os.system('rm %s' % path)
            
                ## labeled marginal distributions
                figr, axis = plt.subplots(numbtimeplotscat - 1, numbtimeplotscat - 1, figsize=(10, 10))
                for t in indxtime[:numbtimeplotscat-1]:
                    for tt in indxtime[:numbtimeplotscat-1]:
                        if t < tt:
                            axis[t][tt].axis('off')
                            continue
                        axis[t][tt].scatter(X[indxdatanega, t+1], X[indxdatanega, tt], s=20, color='r', alpha=0.3)#*listscor[c])
                        axis[t][tt].scatter(X[indxdataposi, t+1], X[indxdataposi, tt], s=20, color='b', alpha=0.3)#*listscor[c])
                        axis[t][tt].set_ylim(limt)
                        axis[t][tt].set_xlim(limt)
                        #axis[t][tt].set_xticks(())
                        #axis[t][tt].set_yticks(())
                path = gdat.pathdata + 'pmar_%s_%04d.pdf'% (gdat.strgcntp, c)
                plt.savefig(path)
                plt.close()
                
                ## median image with the labels
                figr, axis = plt.subplots(figsize=(12, 6))
                objtimag = axis.imshow(np.arcsinh(imagmemomedi), interpolation='nearest', cmap='Greys_r')
                plt.colorbar(objtimag)
                plot_catl(gdat, axis)
                plt.tight_layout()
                path = gdat.pathdata + 'cntpmedi_%s_%04d.pdf' % (gdat.strgcntp, c)
                print 'Writing to %s...' % path
                plt.savefig(path)
                plt.close()

                # plot data with colors based on predicted class
                figr, axis = plt.subplots(10, 4)
                for a in range(10):
                    for b in range(4):
                        p = a * 4 + b
                        if p >= numbdata:
                            continue
                        if False and gdat.datatype == 'mock':
                            if listlablmodl[c][p] == 1 and listlabltrue[p] == 1:
                                colr = 'g'
                            elif listlablmodl[c][p] == 0 and listlabltrue[p] == 0:
                                colr = 'r'
                            elif listlablmodl[c][p] == 0 and listlabltrue[p] == 1:
                                colr = 'b'
                            elif listlablmodl[c][p] == 1 and listlabltrue[p] == 0:
                                colr = 'orange'
                        else:
                            if listlablmodl[c][p] == 1:
                                colr = 'b'
                            else:
                                colr = 'r'
                        axis[a][b].plot(gdat.time, X[p, :].reshape((numbtime, numbsidecorr, numbsidecorr))[:, gdat.offscorr, gdat.offscorr], \
                                                                                            color=colr, alpha=0.1, ls='', marker='o', markersize=3)
                        if a != 9:
                            axis[a][b].set_xticks([])
                        if b != 0:
                            axis[a][b].set_yticks([])
                path = gdat.pathdata + 'datapred_%s_%04d.pdf' % (gdat.strgcntp, c)
                plt.savefig(path)
                plt.close()

                # plot a histogram of decision functions evaluated at the samples
                figr, axis = plt.subplots()
                axis.hist(listscor[c])
                axis.set_xlabel('Score')
                axis.set_yscale('log')
                path = gdat.pathdata + 'histscor_%s_%04d.pdf' % (gdat.strgcntp, c)
                plt.savefig(path)
                plt.close()
                
                # plot data with the least and highest scores
                figr, axis = plt.subplots(20, 2, figsize=(12, 24))

                for l in range(2):
                    for k in range(20):
                        if l == 0:
                            indx = gdat.indxdatascorsort[k]
                        else:
                            indx = gdat.indxdatascorsort[numbdata-k-1]
                        
                        #strg = 'y, x = %d %d' % (indxsideyposdata[indx], indxsidexposdata[indx])
                        #axis[k][l].text(.9, .9, strg, transform=plt.gca().transAxes, size=15)
                        #horizontalalignment='right')
                        
                        if not isinstance(indx, int):
                            indx = indx[0]
                        axis[k][l].plot(gdat.time, X[indx, :].reshape((numbtime, numbsidecorr, numbsidecorr))[:, gdat.offscorr, gdat.offscorr], \
                                                                                                color='black', ls='', marker='o', markersize=3)
                path = gdat.pathdata + 'datasort_%s_%04d.pdf' % (gdat.strgcntp, c)
                plt.savefig(path)
                plt.close()
    
                numbposi = indxsidexposaccp.size
                numbbins = 10
                numbpositrue = np.zeros(numbbins)
                binsmagt = np.linspace(magtminm, magtmaxm, numbbins + 1)
                meanmagt = (binsmagt[1:] + binsmagt[:-1]) / 2.
                reca = np.empty(numbbins)
                numbsupnmagt = np.zeros(numbbins)
                if gdat.datatype == 'mock':
                    for n in indxsupn:
                        indxmagt = np.digitize(np.amax(truemagt[:, n]), binsmagt) - 1
                        numbsupnmagt[indxmagt] += 1
                        indxpixlposi = np.where((abs(indxsidexpossour[n] - indxsidexposaccp) < 2) & (abs(indxsideypossour[n] - indxsideyposaccp) < 2))[0]
                        if indxpixlposi.size > 0:
                            numbpositrue[indxmagt] += 1
                    recamagt = numbpositrue.astype(float) / numbsupnmagt
                    prec = sum(numbpositrue).astype(float) / numbposi
                    figr, axis = plt.subplots(figsize=(12, 6))
                    axis.plot(meanmagt, recamagt, ls='', marker='o')
                    axis.set_ylabel('Recall')
                    axis.set_xlabel('Tmag')
                    plt.tight_layout()
                    path = gdat.pathdata + 'reca_%s_%04d.pdf' % (gdat.strgcntp, c)
                    print 'Writing to %s...' % path
                    plt.savefig(path)
                    plt.close()
    
                c += 1
            
            
            #continue

            ## clustering with pyod
            ## fraction of outliers
            #fracoutl = 0.25
            #
            ## initialize a set of detectors for LSCP
            #detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
            #                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
            #                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
            #                 LOF(n_neighbors=50)]
            #
            ## Show the statics of the data
            ## Define nine outlier detection tools to be compared
            #classifiers = {
            #    'Angle-based Outlier Detector (ABOD)':
            #        ABOD(contamination=fracoutl),
            #    'Cluster-based Local Outlier Factor (CBLOF)':
            #        CBLOF(contamination=fracoutl,
            #              check_estimator=False, random_state=random_state),
            #    'Feature Bagging':
            #        FeatureBagging(LOF(n_neighbors=35),
            #                       contamination=fracoutl,
            #                       random_state=random_state),
            #    #'Histogram-base Outlier Detection (HBOS)': HBOS(
            #    #    contamination=fracoutl),
            #    'Isolation Forest': IForest(contamination=fracoutl,
            #                                random_state=random_state),
            #    'K Nearest Neighbors (KNN)': KNN(
            #        contamination=fracoutl),
            #    'Average KNN': KNN(method='mean',
            #                       contamination=fracoutl),
            #    # 'Median KNN': KNN(method='median',
            #    #                   contamination=fracoutl),
            #    'Local Outlier Factor (LOF)':
            #        LOF(n_neighbors=35, contamination=fracoutl),
            #    # 'Local Correlation Integral (LOCI)':
            #    #     LOCI(contamination=fracoutl),
            #    
            #    #'Minimum Covariance Determinant (MCD)': MCD(
            #    #    contamination=fracoutl, random_state=random_state),
            #    
            #    'One-class SVM (OCSVM)': OCSVM(contamination=fracoutl),
            #    'Principal Component Analysis (PCA)': PCA(
            #        contamination=fracoutl, random_state=random_state, standardization=False),
            #    # 'Stochastic Outlier Selection (SOS)': SOS(
            #    #     contamination=fracoutl),
            #    'Locally Selective Combination (LSCP)': LSCP(
            #        detector_list, contamination=fracoutl,
            #        random_state=random_state)
            #}
            #
            #return
            #raise Exception('')

            ## Fit the model
            #plt.figure(figsize=(15, 12))
            #for i, (clf_name, clf) in enumerate(classifiers.items()):
            #    print(i, 'fitting', clf_name)

            #    # fit the data and tag outliers
            #    clf.fit(X)
            #    scores_pred = clf.decision_function(X) * -1
            #    y_pred = clf.predict(X)
            #    threshold = np.percentile(scores_pred, 100 * fracoutl)
            #    n_errors = np.where(y_pred != listlabltrue)[0].size
            #    # plot the levels lines and the points
            #    #if i == 1:
            #    #    continue
            #    #Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
            #    #Z = Z.reshape(xx.shape)
            #    Z = np.zeros((100, 100))
            #    subplot = plt.subplot(3, 4, i + 1)
            #    subplot.contourf(xx, yy, Z, #levels=np.linspace(Z.min(), threshold, 7),
            #                     cmap=plt.cm.Blues_r)
            #    subplot.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
            #    a = subplot.contour(xx, yy, Z, levels=[threshold],
            #                        linewidths=2, colors='red')
            #    subplot.contourf(xx, yy, Z, #levels=[threshold, Z.max()],
            #                     colors='orange')
            #    b = subplot.scatter(X[:-numboutl, 0], X[:-numboutl, 1], c='green', s=20, edgecolor='k')
            #    c = subplot.scatter(X[-numboutl:, 0], X[-numboutl:, 1], c='purple', s=20, edgecolor='k')
            #    subplot.axis('tight')
            #    subplot.legend(
            #        [a.collections[0], b, c],
            #        ['learned decision function', 'true inliers', 'true outliers'],
            #        prop=matplotlib.font_manager.FontProperties(size=10),
            #        loc='lower right')
            #    subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
            #    subplot.set_xlim(limtproj)
            #    subplot.set_ylim(limtproj)
            #plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
            #plt.suptitle("Outlier detection")
            #plt.savefig(pathplot + 'pyod.png', dpi=300)
            #plt.close()

            #
            #default_base = {'quantile': .3,
            #                'eps': .3,
            #                'damping': .9,
            #                'preference': -200,
            #                'n_neighbors': 10,
            #                'n_clusters': 3,
            #                'min_samples': 20,
            #                'xi': 0.05,
            #                'min_cluster_size': 0.1}
            #
            ## update parameters with dataset-specific values
            #
            #algo_params = {'damping': .77, 'preference': -240,
            #     'quantile': .2, 'n_clusters': 2,
            #     'min_samples': 20, 'xi': 0.25}

            #params = default_base.copy()
            #params.update(algo_params)
            #
            ## normalize dataset for easier parameter selection
            #X = StandardScaler().fit_transform(X)
            #
            ## estimate bandwidth for mean shift
            #bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
            #
            ## connectivity matrix for structured Ward
            #connectivity = kneighbors_graph(
            #    X, n_neighbors=params['n_neighbors'], include_self=False)
            ## make connectivity symmetric
            #connectivity = 0.5 * (connectivity + connectivity.T)
            #
            #ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            #two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
            #ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
            #spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
            #dbscan = cluster.DBSCAN(eps=params['eps'])
            #
            ##optics = cluster.OPTICS(min_samples=params['min_samples'],
            ##                        xi=params['xi'],
            ##                        min_cluster_size=params['min_cluster_size'])
            #
            #affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
            #average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", \
            #                                                                    n_clusters=params['n_clusters'], connectivity=connectivity)
            #birch = cluster.Birch(n_clusters=params['n_clusters'])
            #gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
            #
            #clustering_algorithms = (
            #    ('MiniBatchKMeans', two_means),
            #    ('AffinityPropagation', affinity_propagation),
            #    ('MeanShift', ms),
            #    ('SpectralClustering', spectral),
            #    ('Ward', ward),
            #    ('AgglomerativeClustering', average_linkage),
            #    ('DBSCAN', dbscan),
            #    #('OPTICS', optics),
            #    ('Birch', birch),
            #    ('GaussianMixture', gmm)
            #)
            #
            #figr, axis = plt.subplots(1, numbmeth)
            #k = 0
            #for name, algorithm in clustering_algorithms:
            #    t0 = timemodu.time()
            #    
            #    print 'name'
            #    print name
            #    print

            #    # catch warnings related to kneighbors_graph
            #    with warnings.catch_warnings():
            #        #warnings.filterwarnings(
            #        #    "ignore",
            #        #    message="the number of connected components of the " +
            #        #    "connectivity matrix is [0-9]{1,2}" +
            #        #    " > 1. Completing it to avoid stopping the tree early.",
            #        #    category=UserWarning)
            #        #warnings.filterwarnings(
            #        #    "ignore",
            #        #    message="Graph is not fully connected, spectral embedding" +
            #        #    " may not work as expected.",
            #        #    category=UserWarning)
            #        algorithm.fit(X)
            #
            #    t1 = timemodu.time()
            #    if hasattr(algorithm, 'labels_'):
            #        print 'Has labels_'
            #        lablmodl = algorithm.labels_.astype(np.int)
            #    else:
            #        lablmodl = algorithm.predict(X)
            #    
            #    print 'lablmodl'
            #    summgene(lablmodl)
            #    print ''

            #    axis[k].set_title(name, size=18)
            #
            #    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
            #                                         '#f781bf', '#a65628', '#984ea3',
            #                                         '#999999', '#e41a1c', '#dede00']),
            #                                  int(max(lablmodl) + 1))))
            #    # add black color for outliers (if any)
            #    colors = np.append(colors, ["#000000"])
            #    axis[k].scatter(X[:, 0], X[:, 1], s=10, color=colors[lablmodl])
            #
            #    axis[k].set_xlim(-2.5, 2.5)
            #    axis[k].set_ylim(-2.5, 2.5)
            #    axis[k].set_xticks(())
            #    axis[k].set_yticks(())
            #    axis[k].text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
            #             transform=plt.gca().transAxes, size=15,
            #             horizontalalignment='right')
            #    k += 1
            #    listlablmodl.append(lablmodl)
            #path = gdat.pathdata + 'clus.pdf'
            #plt.savefig(path)
            #plt.close()


            ## Random 2D projection using a random unitary matrix
            #print("Computing random projection")
            #rp = random_projection.SparseRandomProjection(n_components=2)
            #X_projected = rp.fit_transform(lcurflat)
            #print 'X_projected'
            #summgene(X_projected)
            #plot_embe(gdat, lcurflat, X_projected, 'rand', "Random Projection")
            #
            ## Projection on to the first 2 principal components
            #print("Computing PCA projection")
            #t0 = timemodl.time()
            #print 'lcurflat'
            #summgene(lcurflat)
            #X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(lcurflat)
            #plot_embe(gdat, lcurflat, X_pca, 'pcaa', "Principal Components projection (time %.2fs)" % (timemodl.time() - t0))
            #
            ## Projection on to the first 2 linear discriminant components
            ##print("Computing Linear Discriminant Analysis projection")
            ##X2 = lcurflat.copy()
            ##X2.flat[::lcurflat.shape[1] + 1] += 0.01  # Make X invertible
            ##t0 = timemodl.time()
            ##X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
            ##plot_embe(gdat, lcurflat, X_lda, 'ldap', "Linear Discriminant projection (time %.2fs)" % (timemodl.time() - t0))
            #
            ## t-SNE embedding dataset
            #print("Computing t-SNE embedding")
            #tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=30)
            #t0 = timemodl.time()
            #X_tsne = tsne.fit_transform(lcurflat)
            #plot_embe(gdat, lcurflat, X_tsne, 'tsne0030', "t-SNE embedding with perplexity 30")
            #
            ## t-SNE embedding dataset
            #print("Computing t-SNE embedding")
            #tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=5)
            #t0 = timemodl.time()
            #X_tsne = tsne.fit_transform(lcurflat)
            #plot_embe(gdat, lcurflat, X_tsne, 'tsne0005', "t-SNE embedding with perplexity 5")
            #
            ## Isomap projection dataset
            #print("Computing Isomap projection")
            #t0 = timemodl.time()
            #X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(lcurflat)
            #print("Done.")
            #plot_embe(gdat, lcurflat, X_iso, 'isop', "Isomap projection (time %.2fs)" % (timemodl.time() - t0))
            #
            ## Locally linear embedding dataset
            #print("Computing LLE embedding")
            #clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
            #t0 = timemodl.time()
            #X_lle = clf.fit_transform(lcurflat)
            #print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
            #plot_embe(gdat, lcurflat, X_lle, 'llep', "Locally Linear Embedding (time %.2fs)" % (timemodl.time() - t0))
            #
            ## Modified Locally linear embedding dataset
            #print("Computing modified LLE embedding")
            #clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='modified')
            #t0 = timemodl.time()
            #X_mlle = clf.fit_transform(lcurflat)
            #print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
            #plot_embe(gdat, lcurflat, X_mlle, 'mlle', "Modified Locally Linear Embedding (time %.2fs)" % (timemodl.time() - t0))
            #
            ## HLLE embedding dataset
            #print("Computing Hessian LLE embedding")
            #clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='hessian')
            #t0 = timemodl.time()
            #X_hlle = clf.fit_transform(lcurflat)
            #print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
            #plot_embe(gdat, lcurflat, X_hlle, 'hlle', "Hessian Locally Linear Embedding (time %.2fs)" % (timemodl.time() - t0))
            #
            ## LTSA embedding dataset
            #print("Computing LTSA embedding")
            #clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa')
            #t0 = timemodl.time()
            #X_ltsa = clf.fit_transform(lcurflat)
            #print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
            #plot_embe(gdat, lcurflat, X_ltsa, 'ltsa', "Local Tangent Space Alignment (time %.2fs)" % (timemodl.time() - t0))
            #
            ## MDS  embedding dataset
            #print("Computing MDS embedding")
            #clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
            #t0 = timemodl.time()
            #X_mds = clf.fit_transform(lcurflat)
            #print("Done. Stress: %f" % clf.stress_)
            #plot_embe(gdat, lcurflat, X_mds, 'mdse', "MDS embedding (time %.2fs)" % (timemodl.time() - t0))
            #
            ## Random Trees embedding dataset
            #print("Computing Totally Random Trees embedding")
            #hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5)
            #t0 = timemodl.time()
            #X_transformed = hasher.fit_transform(lcurflat)
            #pca = decomposition.TruncatedSVD(n_components=2)
            #X_reduced = pca.fit_transform(X_transformed)
            #plot_embe(gdat, lcurflat, X_reduced, 'rfep', "Random forest embedding (time %.2fs)" % (timemodl.time() - t0))
            #
            ## Spectral embedding dataset
            #print("Computing Spectral embedding")
            #embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
            #t0 = timemodl.time()
            #X_se = embedder.fit_transform(lcurflat)
            #plot_embe(gdat, lcurflat, X_se, 'csep', "Spectral embedding (time %.2fs)" % (timemodl.time() - t0))
            #
            ## NCA projection dataset
            ##print("Computing NCA projection")
            ##nca = neighbors.NeighborhoodComponentsAnalysis(n_components=2, random_state=0)
            ##t0 = timemodl.time()
            ##X_nca = nca.fit_transform(lcurflat, y)
            ##plot_embe(gdat, lcurflat, X_nca, 'ncap', "NCA embedding (time %.2fs)" % (timemodl.time() - t0))

            #indxsidexposaccp = np.array(indxsidexposaccp)
            #indxsideyposaccp = np.array(indxsideyposaccp)
            #figr, axis = plt.subplots(figsize=(12, 6))
            #objtimag = axis.imshow(np.std(imagmemo, axis=2), interpolation='nearest', cmap='Reds')
            #
            #if gdat.datatype == 'mock':
            #    for n in indxsupn:
            #        axis.scatter(indxsidexpossour[n], indxsideypossour[n], s=50, marker='o', color='g')
            #
            #for indxsideyposaccptemp, indxsidexposaccptemp in zip(indxsideyposaccp, indxsidexposaccp):
            #    axis.scatter(indxsidexposaccptemp, indxsideyposaccptemp, s=50, marker='x', color='b')

            #plt.colorbar(objtimag)
            #plt.tight_layout()
            #path = gdat.pathdata + 'cntpstdvfinl_%s.pdf' % gdat.strgcntp
            #print 'Writing to %s...' % path
            #plt.savefig(path)
            #plt.close()
            #
            #for strgvarb in ['diff']:
            #    figr, axis = plt.subplots(figsize=(12, 6))
            #    #if strgvarb == 'diff':
            #    #    varbtemp = np.arcsinh(dictpara[strgvarb])
            #    #else:
            #    #    varbtemp = dictpara[strgvarb]
            #    varbtemp = dictpara[strgvarb]
            #    vmin = -1
            #    vmax = 1
            #    objtimag = axis.imshow(varbtemp, interpolation='nearest', cmap='Greens', vmin=vmin, vmax=vmax)
            #    for indxsideyposaccptemp, indxsidexposaccptemp in zip(indxsideyposaccp, indxsidexposaccp):
            #        axis.scatter(indxsideyposaccptemp, indxsidexposaccptemp, s=5, marker='x', color='b', lw=0.5)
            #    plt.colorbar(objtimag)
            #    plt.tight_layout()
            #    path = gdat.pathdata + '%s_%s.pdf' % (strgvarb, gdat.strgcntp)
            #    print 'Writing to %s...' % path
            #    plt.savefig(path)
            #    plt.close()
    

def plot_catl(gdat, axis):

    for k in range(gdat.numbpositext):
        axis.text(gdat.indxsideyposdataflat[gdat.indxdatascorsort[k]] + gdat.numbsideedge, \
                  gdat.indxsidexposdataflat[gdat.indxdatascorsort[k]] + gdat.numbsideedge, '%d' % k, size=7, color='b')
    
    if gdat.datatype == 'mock':
        for k in gdat.indxsoursupn:
            axis.text(np.mean(gdat.trueypos[:, k]), np.mean(gdat.truexpos[:, k]), '*', size=7, color='g')


def cnfg_tici():

    # 272551828
    isec = 3
    icam = 4
    iccd = 3
    rasctarg = 121.865609
    decltarg = -76.533524
    work(isec, icam, iccd, rasctarg=rasctarg, decltarg=decltarg)


def cnfg_mock():
   
    work(9, 1, 1, datatype='mock')


def cnfg_tdie():
   
    pathdata = '/Users/tdaylan/Documents/work/data/tesstran/tdie/tesscut/'
    listpath = fnmatch.filter(os.listdir(pathdata), 'tess*')
    for p in range(len(listpath)):
        isec = int(listpath[p][6:10])
        icam = int(listpath[p][11])
        iccd = int(listpath[p][13])
        pathfile = pathdata + listpath[p]
        if isec == 7 or isec == 8:
            work(isec, icam, iccd, pathfile=pathfile)


def cnfg_defa():
    
    jobs = []
    isec = 1
    for i in range(1, 5):
        for j in range(1, 5):
            if i == 4 and  j == 1:
                work(isec, i, j)
                
            #p = multiprocessing.Process(target=work, args=(isec, i, j))
            #jobs.append(p)
            #p.start()


def cnfg_sect():
    
    for isec in range(9, 10):
        for icam in range(1, 2):
            for iccd in range(2, 3):
                work(isec, icam, iccd)

globals().get(sys.argv[1])()

