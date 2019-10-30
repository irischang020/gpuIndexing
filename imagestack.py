import psana
import h5py
import numpy as np

def getBragg4pcxi(fcxi=None,pidx=None,eventInd=None):
    ## pidx: idx in cxi file
    ## eventInd: real idx in psocake event
    if pidx is None and eventInd is None:
        f = h5py.File(fcxi, "r")
        xraw = f["entry_1/result_1/peakXPosRaw"].value.astype(int)
        yraw = f["entry_1/result_1/peakYPosRaw"].value.astype(int)
        nPeak = f["entry_1/result_1/nPeaks"].value.astype(int)
        intens = f["entry_1/result_1/peakTotalIntensity"].value
        eventNumber = f["LCLS/eventNumber"].value.astype(int)
        f.close()
        return yraw,xraw
    if eventInd is not None:
        f = h5py.File(fcxi, "r") 
        eventNumber = f["LCLS/eventNumber"].value.astype(int) 
        f.close()
        
        if eventInd not in eventNumber:    
            print("## This event is not saved")
            return 
        else:
            idx = np.where(eventNumber==eventInd)[0][0]
        
        f = h5py.File(fcxi, "r") 
        xraw = f["entry_1/result_1/peakXPosRaw"][idx].astype(int)
        yraw = f["entry_1/result_1/peakYPosRaw"][idx].astype(int)
        f.close()
        index = np.where((xraw!=0) | (yraw!=0))
        
        return yraw[index],xraw[index]
    if pidx is not None: 
        f = h5py.File(fcxi, "r")
        xraw = f["entry_1/result_1/peakXPosRaw"][pidx].astype(int)
        yraw = f["entry_1/result_1/peakYPosRaw"][pidx].astype(int) 
        f.close()
        index = np.where((xraw!=0) | (yraw!=0))
        return yraw[index],xraw[index]

def mapcheetch2stack(experimentName=None,detInfo=None):
    if 'cspad' in detInfo.lower() and 'cxi' in experimentName:
        dim0 = 8 * 185
        dim1 = 4 * 388
    elif 'rayonix' in detInfo.lower() and 'mfx' in experimentName:
        dim0 = 1920
        dim1 = 1920
    elif 'rayonix' in detInfo.lower() and 'xpp' in experimentName:
        dim0 = 1920
        dim1 = 1920
    else:
        print ("!! No such detector")
        return 
    
    ch2stkx = np.zeros((dim0,dim1)).astype(int)
    ch2stky = np.zeros((dim0,dim1)).astype(int)
    ch2stkz = np.zeros((dim0,dim1)).astype(int)
    
    counter = 0
    if 'cspad' in detInfo.lower() and 'cxi' in experimentName:
        for quad in range(4):
            for seg in range(8):
                x,y = np.meshgrid(np.arange(185),np.arange(388),indexing="ij")
                ch2stkx[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = counter
                ch2stky[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = x
                ch2stkz[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = y 
                counter += 1
    elif 'rayonix' in detInfo.lower() and 'mfx' in experimentName:
        x,y = np.meshgrid(np.arange(dim0),np.arange(dim1),indexing="ij")
        ch2stkx = counter
        ch2stky = x 
        ch2stkz = y  
    elif 'rayonix' in detInfo.lower() and 'xpp' in experimentName:
        x,y = np.meshgrid(np.arange(dim0),np.arange(dim1),indexing="ij")
        ch2stkx = counter
        ch2stky = x 
        ch2stkz = y  
        
    return ch2stkx,ch2stky,ch2stkz


def stack2image(stack=None,experimentName=None,runNumber=None,detInfo=None,eventInd=0):
    ds = psana.DataSource('exp='+str(experimentName)+':run='+str(runNumber)+':idx')
    run = ds.runs().next()
    times = run.times()
    env = ds.env()
    evt = run.event(times[0])
    det = psana.Detector(str(detInfo), env)
    evt = run.event(times[eventInd])
    image = det.image(evt,stack)
    return image

def getImage(experimentName=None,runNumber=None,detInfo=None,eventInd=0):
    ds = psana.DataSource('exp='+str(experimentName)+':run='+str(runNumber)+':idx')
    run = ds.runs().next()
    times = run.times()
    env = ds.env()
    evt = run.event(times[0])
    det = psana.Detector(str(detInfo), env)
    evt = run.event(times[eventInd])
    image = det.image(evt)
    return image


def getStack(experimentName=None,runNumber=None,detInfo=None,eventInd=0):
    ds = psana.DataSource('exp='+str(experimentName)+':run='+str(runNumber)+':idx')
    run = ds.runs().next()
    times = run.times()
    env = ds.env()
    evt = run.event(times[0])
    det = psana.Detector(str(detInfo), env)
    evt = run.event(times[eventInd])
    stack = det.calib(evt)
    return stack

def image2stack(image=None,experimentName=None,runNumber=None,detInfo=None,eventInd=0):

    ds = psana.DataSource('exp='+str(experimentName)+':run='+str(runNumber)+':idx')
    run = ds.runs().next()
    times = run.times()
    env = ds.env()
    evt = run.event(times[0])
    det = psana.Detector(str(detInfo), env)

    evt = run.event(times[eventInd]) 

    stack = det.ndarray_from_image(par=evt, image=image, pix_scale_size_um=None, xy0_off_pix=None)
    return stack 


def stack2cheetah(stack=None,experimentName=None,detInfo=None):
    
    if 'cspad' in detInfo.lower() and 'cxi' in experimentName:
        dim0 = 8 * 185
        dim1 = 4 * 388
    elif 'rayonix' in detInfo.lower() and 'mfx' in experimentName:
        dim0 = 1920
        dim1 = 1920
    elif 'rayonix' in detInfo.lower() and 'xpp' in experimentName:
        dim0 = 1920
        dim1 = 1920
    else:
        print ("!! No such detector")
        return 

    img = np.zeros((dim0, dim1))
    counter = 0
    if 'cspad' in detInfo.lower() and 'cxi' in experimentName:
        for quad in range(4):
            for seg in range(8):
                img[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = stack[counter, :, :]
                counter += 1
    elif 'rayonix' in detInfo.lower() and 'mfx' in experimentName:
        img = stack[counter, :, :]  
    elif 'rayonix' in detInfo.lower() and 'xpp' in experimentName:
        img = stack[counter, :, :]  

    return img


def cheetah2stack(cheetah=None,experimentName=None,detInfo=None):
    if 'cspad' in detInfo.lower() and 'cxi' in experimentName:
        dim0 = 8 * 185
        dim1 = 4 * 388
    elif 'rayonix' in detInfo.lower() and 'mfx' in experimentName:
        dim0 = 1920
        dim1 = 1920
    elif 'rayonix' in detInfo.lower() and 'xpp' in experimentName:
        dim0 = 1920
        dim1 = 1920
    else:
        print ("!! No such detector")
        return 
    
    counter = 0
    if 'cspad' in detInfo.lower() and 'cxi' in experimentName:
        stack = np.zeros((32,185,388))
        for quad in range(4):
            for seg in range(8):
                stack[counter, :, :] = cheetah[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388]
                counter += 1
    elif 'rayonix' in detInfo.lower() and 'mfx' in experimentName:
        stack = np.zeros((1,dim0,dim1))
        stack[counter, :, :] = cheetah
    elif 'rayonix' in detInfo.lower() and 'xpp' in experimentName:
        stack = np.zeros((1,dim0,dim1))
        stack[counter, :, :] = cheetah
    return stack


def spreadf(mask, expandSize=(1,1), expandValue=0): 
    (nx,ny) = mask.shape
    newMask = mask.copy()
    index = np.where(mask==expandValue)
    for i in range(-expandSize[0], expandSize[0]+1):
        for j in range(-expandSize[1], expandSize[1]+1):
            newMask[((index[0]+i)%nx, (index[1]+j)%ny)] = expandValue
    return newMask

def display(image,figsize=(8,8), clim=None):
    import matplotlib.pyplot as plt 
    plt.figure(figsize=figsize)
    if clim is None:
        plt.imshow(image[:,::-1].T)
    plt.imshow(image[:,::-1].T, clim=clim)
    plt.tight_layout()
    plt.show()