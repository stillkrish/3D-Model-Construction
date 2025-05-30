import numpy as np
import matplotlib.pyplot as plt
from visutils import *
from camutils import *


def decode(imprefix_color, imprefix, start, threshold_color, thresh):
    """
    Decode a binary-coded pattern from a single set of 10 images (no inverse images).
    
    Parameters
    ----------
    imprefix_color : str
        Prefix for the color images (to create a color mask).
    imprefix : str
        Prefix for where to find the pattern images. 
        For N=10 bits, we expect images from imprefix<start>.png to imprefix<start+9>.png.
    start : int
        The starting index of the pattern images.
    threshold_color : float
        Threshold for differentiating object from background using color.
    thresh : float
        Threshold for determining if a pixel is considered "on" for a given bit.
        If img > thresh, bit = 1, else bit = 0.
    
    Returns
    -------
    code : 2D numpy.array (dtype=float)
        The decoded binary code for each pixel.
    mask : 2D numpy.array (dtype=float)
        A mask indicating which pixels were successfully decoded (1) or not (0).
    color_mask : 2D numpy.array (dtype=float)
        A mask based on color difference that helps filter out background (1 means foreground).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    nbits = 10
    imgs = []

    for i in range(start, start + nbits):
        fname = f'{imprefix}{i:02d}.png'
        img = plt.imread(fname)

        # Convert to float [0,1] if needed
        if img.dtype == np.uint8:
            img = img.astype(float) / 255.0

        # Convert to grayscale if color
        if img.ndim > 2:
            img = np.mean(img, axis=2)

        imgs.append(img)

    (h, w) = imgs[0].shape

    # Determine bit values based on threshold
    bit_values = np.zeros((h, w, nbits), dtype=int)
    mask = np.ones((h, w), dtype=bool)

    for i in range(nbits):
        # If img > thresh, bit = 1, else bit = 0
        bit_values[:, :, i] = (imgs[i] > thresh).astype(int)
        # Update mask if needed: for single images, we might just trust the threshold
        # but if you want to ensure there's actual structure, you can add conditions.
        # For now, we just keep mask as is or set it to True where img > some min brightness:
        mask = mask & (imgs[i] > (thresh * 0.5))  # Example condition, adjust as needed.

    # Combine bits into code
    code = np.zeros((h, w), dtype=int)
    for i in range(nbits):
        code += (bit_values[:, :, i] << (nbits - 1 - i))

    # Color mask
    imc1 = plt.imread(imprefix_color + f"{0:02d}.png")
    imc2 = plt.imread(imprefix_color + f"{1:02d}.png")
    if imc1.dtype == np.uint8:
        imc1 = imc1.astype(float)/255.0
        imc2 = imc2.astype(float)/255.0

    color_diff = np.sum((imc1 - imc2)**2, axis=-1)
    color_mask = (color_diff > threshold_color)

    # Convert masks to float
    mask = mask.astype(float)
    color_mask = color_mask.astype(float)

    return code, mask, color_mask


def reconstruct(imprefixL1,imprefixL2,imprefixR1,imprefixR2,threshold1,threshold2,threshold3,camL,camR):
    """
    Simple reconstruction based on triangulating matched pairs of points
    between to view.

    Parameters
    ----------
    imprefix : str
      prefix for where the images are stored

    threshold : float
      decodability threshold

    camL,camR : Camera
      camera parameters

    Returns
    -------
    pts2L,pts2R : 2D numpy.array (dtype=float)

    pts3 : 2D numpy.array (dtype=float)

    """

    CLh,maskLh,cmaskL = decode(imprefixL1,imprefixL2,0,threshold1,threshold2)
    # plt.imshow(CLh)
    # plt.title("Code")

    # plt.figure(figsize=(4,3))
    # plt.imshow(maskLh)
    # plt.title("Mask")

    # plt.figure(figsize=(4,3))
    # plt.imshow(cmaskL)
    # plt.title("Color Mask")
    # plt.show()

    CLv,maskLv,_ = decode(imprefixL1,imprefixL2,10,threshold1,threshold3)
    # plt.imshow(CLv)
    # plt.title("Code")

    # plt.figure(figsize=(4,3))
    # plt.imshow(maskLv)
    # plt.title("Mask")
    # plt.show()

    CRh,maskRh,cmaskR = decode(imprefixR1,imprefixR2,0,threshold1,threshold2)
    # plt.imshow(CRh)
    # plt.title("Code")

    # plt.figure(figsize=(4,3))
    # plt.imshow(maskRh)
    # plt.title("Mask")

    # plt.figure(figsize=(4,3))
    # plt.imshow(cmaskR)
    # plt.title("Color Mask")

    # plt.show()

    CRv,maskRv,_ = decode(imprefixR1,imprefixR2,10,threshold1,threshold3)
    # plt.imshow(CRv)
    # plt.title("Code")

    # plt.figure(figsize=(4,3))
    # plt.imshow(maskRv)
    # plt.title("Mask")
    
    # plt.show()



    maskL = maskLh*maskLv*cmaskL
    CL = (CLh + 1024*CLv) * maskL
    # print("CL ", CL)
    # print("maskL ", maskL)
    # print(np.unique(maskL))


    maskR = maskRh*maskRv*cmaskR
    CR = (CRh + 1024*CRv) * maskR
    # print("CR ", CR)
    # print("maskR ", maskR)
    # print(np.unique(maskR))

    h = CR.shape[0]
    w = CR.shape[1]

    subR = np.nonzero(maskR.flatten())
    subL = np.nonzero(maskL.flatten())

    CRgood = CR.flatten()[subR]
    CLgood = CL.flatten()[subL]
    # crset = set(CRgood)
    # clset = set(CLgood)
    # print("\n CR GOOD ", CRgood)
    # print("\n CR GOOD ", crset)
    # print("\n LENCR ", len(CRgood))
    # print("\n LENCR ", len(crset))
    # print("\n CL GOOD ", CLgood)
    # print("\n CL GOOD ", clset)
    # print("\n LENCL ", len(CLgood))
    # print("\n LENCL ", len(clset))

    # common = np.isin(CRgood, CLgood)
    # print("Number of common elements:", np.sum(common))

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,5))
    # plt.subplot(121)
    # plt.hist(CRgood, bins=100, alpha=0.5, label='CRgood')
    # plt.hist(CLgood, bins=100, alpha=0.5, label='CLgood')
    # plt.legend()
    # plt.subplot(122)
    # plt.scatter(CRgood, np.zeros_like(CRgood), alpha=0.5, label='CRgood')
    # plt.scatter(CLgood, np.ones_like(CLgood), alpha=0.5, label='CLgood')
    # plt.legend()
    # plt.show()

    _,submatchR,submatchL = np.intersect1d(CRgood,CLgood,return_indices=True)

    # print("\n submatchR ", submatchR)
    # print("\n submatchL ", submatchL)

    matchR = subR[0][submatchR]
    matchL = subL[0][submatchL]

    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))

    pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
    pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)
    
    imageL= plt.imread(imprefixL1 +"%02d" % (1)+'.png')
    imageR = plt.imread(imprefixR1 +"%02d" % (1)+'.png')
    bvaluesL_list=[]
    bvaluesR_list=[]
    for i in range(pts2L.shape[1]):
        bvaluesL_list.append(imageL[pts2L[1][i]][pts2L[0][i]])
        bvaluesR_list.append(imageR[pts2R[1][i]][pts2R[0][i]])
    bvaluesL=np.array(bvaluesL_list).T
    bvaluesR=np.array(bvaluesR_list).T
    bvalues=(bvaluesL+bvaluesR)/2

    pts3 = triangulate(pts2L,camL,pts2R,camR)

    return pts2L,pts2R,pts3,bvalues
