import math
import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.morphology import label

class CartoonImage:
    def __init__(self, img, meanX, p, T, blurringSize, levels):
        self.img = img
        self.meanX = meanX
        self.p = p
        self.T = T
        self.blurringSize = blurringSize
        self.N = levels

    def applyEffect(self):
        img_g = self.img

        if(self.img.ndim != 2):
            img_g = self._gaussian(rgb2gray(self.img), self.blurringSize)
        else:
            img_g = self._gaussian(self.img, self.blurringSize)
        
        i_min = np.min(img_g)
        i_max = np.max(img_g)

        bins = np.zeros(self.N + 1)
        constant = (i_max - i_min) / self.N
        for i in range(self.N + 1):
            bins[i] = constant * i + i_min

        L = np.zeros(img_g.shape, dtype=np.uint8)

        for i in range(self.N):
            if(i == self.N - 1):
                mask = img_g >= bins[i]
            else:
                mask = np.logical_and(img_g >= bins[i], img_g < bins[i+1])

            L[mask] = i

        S = np.zeros(img_g.shape)
        
        if(self.img.ndim != 2):
            L_lab = label(L)
            for lev in range(255):
                blur_in = L_lab == lev
                blur_in_stacked = np.stack((blur_in, blur_in, blur_in), axis=2)
                S[blur_in] = np.mean(self.img, where=blur_in_stacked)

        else:
            S = S / self.N
        return S * self._extendedDoG(self.meanX, self.p, self.T)


    """HELPER FUNCTIONS"""

    def _convolve(self, img, kernel):
        '''Convenience method around ndimage.convolve.

        This calls ndimage.convolve with the boundary setting set to 'nearest'.  It
        also performs some checks on the input image.

        Parameters
        ----------
        img : numpy.ndarray
            input image
        kernel : numpy.ndarray
            filter kernel

        Returns
        -------
        numpy.ndarray
            filter image

        Raises
        ------
        ValueError
            if the image is not greyscale
        TypeError
            if the image or filter kernel are not a floating point type
        '''
        if img.ndim != 2:
            raise ValueError('Only greyscale images are supported.')

        if img.dtype != np.float32 and img.dtype != np.float64:
            raise TypeError('Image must be floating point.')

        if kernel.dtype != np.float32 and img.dtype != np.float64:
            raise TypeError('Filter kernel must be floating point.')

        return ndimage.convolve(img, kernel, mode='nearest')


    def _gaussian(self, img, sigma):
        '''Filter an image using a Gaussian kernel.

        The Gaussian is implemented internally as a two-pass, separable kernel.

        Note
        ----
        The kernel is scaled to ensure that its values all sum up to '1'.  The
        slight truncation means that the filter values may not actually sum up to
        one.  The normalization ensures that this is consistent with the other
        low-pass filters in this assignment.

        Parameters
        ----------
        img : numpy.ndarray
            a greyscale image
        sigma : float
            the width of the Gaussian kernel; must be a positive, non-zero value

        Returns
        -------
        numpy.ndarray
            the Gaussian blurred image; the output will have the same type as the
            input

        Raises
        ------
        ValueError
            if the value of sigma is negative
        '''
        # Check Sigma value (did it based off of test provided)
        if(sigma <= 0.01):
            raise ValueError("Error! Negative sigma value provided for gaussian filter")

        # Get kernel width size
        N = max(math.ceil(6 * sigma), 3)

        if (N % 2 == 0):
            N = N + 1

        # Initialize our gaussian kernel
        gaussianKernel = np.ones((N, N), dtype=np.float32)

        # Calculate the gaussian constant value
        constant = 1 / math.sqrt(2 * math.pi * sigma ** 2)

        # Calculate the appropriate gaussian values vertically
        for i in range(N):
            eVal = math.exp(-1.0 * (i - math.floor(N/2))**2/(2 * sigma ** 2))
            gaussianKernel[i] = eVal * constant

        # Calculate the appropriate gaussian values horizontally
        gaussianKernel = np.array(np.transpose(gaussianKernel) * gaussianKernel, dtype=float)

        # Normalize
        gaussianKernel = gaussianKernel / np.sum(gaussianKernel)

        # Convolve image and kernel
        return self._convolve(img, gaussianKernel)

    def _LineCleanup(self, img):
        kernel = np.array([1, 0, -1]).reshape(1, 3)

        E_horz = self._convolve(img, kernel)
        E_vert = self._convolve(img, np.transpose(kernel))
        
        C = np.logical_and(E_horz > 0, E_vert > 0)

        """
        C = np.zeros(img.shape)
        for row in range(len(pixelVals)):
            for col in range(len(pixelVals[row])):
                if(E_horz[row][col] > 0 and E_vert[row][col] < 0):
                    C[row][col] = E_horz[row][col]
                elif(E_horz[row][col] < 0 and E_vert[row][col] > 0):
                    C[row][col] = E_vert[row][col]
        """
        filter_kernel = 1/9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        B = self._convolve(img, filter_kernel)
        
        I_filter = np.copy(img)
        I_filter[C] = B[C]

        return I_filter

    def _extendedDoG(self, filterSize, strength, thresh):
        if self.img.ndim != 2:
            self.img = rgb2gray(self.img)
            
        img_copy = np.copy(self.img)

        G_img = self._gaussian(img_copy, filterSize)

        DoG = G_img - self._gaussian(img_copy, 1.6 * filterSize)

        U = G_img + strength * DoG
                
        I_xDoG = np.where(U > thresh, 1, 0).astype(np.float64)
        

        return self._LineCleanup(I_xDoG)
    
    def _adjust_contrast(self, scale, hist):
        '''Generate a LUT to adjust contrast without affecting brightness.

        Parameters
        ----------
        scale : float
            the value used to adjust the image contrast; a value greater than 1 will
            increase constrast while a value less than 1 will reduce it
        hist : numpy.ndarray
            a 256-element array containing the image histogram, which is used to
            calculate the image brightness

        Returns
        -------
        numpy.ndarray
            a 256-element LUT that can be provided to ``apply_lut()``

        Raises
        ------
        ValueError
            if the histogram is not 256-elements or if the scale is less than zero
        '''

        if(len(hist) != 256 or scale < 0):
            return ValueError("Error histogram is not 256 elements or scale is less than zero")

        brightness = 0
        total = 0

        for i in range(len(hist)):
            brightness = brightness + (i * hist[i])
            total = total + hist[i]

        brightness = brightness // total

        lut = np.arange(0, 256, 1)

        for i in range(len(lut)):
            newVal = (scale * i) + ((1-scale) * brightness)
            if(newVal > 255):
                newVal = 255

            if(newVal < 0):
                newVal = 0

            lut[i] = newVal

        return lut.astype(np.uint8)

    def _apply_lut(self, img, lut):
        '''Apply a look-up table to an image.

        The look-up table can be be used to quickly adjust the intensities within an
        image.  For colour images, the same LUT can be applied equally to each
        colour channel.

        Parameters
        ----------
        img : numpy.ndarray
            a ``H x W`` greyscale or ``H x W x C`` colour 8bpc image
        lut : numpy.ndarray
            a 256-element, 8-bit array

        Returns
        -------
        numpy.ndarray
            a new ``H x W`` or ``H x W x C`` image derived from applying the LUT

        Raises
        ------
        ValueError
            if the LUT is not 256-elements long
        TypeError
            if either the LUT or images are not 8bpc
        '''
        if(len(lut) != 256):
            raise ValueError("Error! LUT is not 256-elements long")

        if(img.dtype != np.uint8 or lut.dtype != np.uint8):
            raise TypeError("Error! Can only support 8-bit images and LUTs!")

        if(len(img.shape) >= 3):
            height, width, colours = img.shape
        else:
            height, width = img.shape
            colours = 1

        pixelVals = img.astype(np.uint8)

        if(len(img.shape) >= 3):
            newImage = np.zeros((height, width, colours), dtype=np.uint8)
        else:
            newImage = np.zeros((height, width), dtype=np.uint8)

        for row in range(len(pixelVals)):
            for col in range(len(pixelVals[row])):
                if(len(img.shape) < 3):
                    greyVal = pixelVals[row][col]
                    newVal = lut[greyVal]
                    newImage[row][col] = newVal
                else:
                    R, G, B = pixelVals[row][col]
                    newR = lut[R]
                    newG = lut[G]
                    newB = lut[B]
                    newImage[row][col][0] = newR
                    newImage[row][col][1] = newG
                    newImage[row][col][2] = newB
        return newImage
    
    def histogram(self, img):
        '''Compute the histogram of an image.

        This function can only support processing 8bpc images, greyscale or colour.
        Colour images will produce three histograms (one per colour channel).

        Parameters
        ----------
        img : numpy.ndarray
            a ``H x W`` greyscale image

        Returns
        -------
        numpy.ndarray
            a 256-element, linear array containing the computed histogram

        Raises
        ------
        ValueError
            if the image isn't greyscale
        TypeError
            if the image isn't the ``numpy.uint8`` data type
        '''
        if(len(img.shape) >= 3):
            raise ValueError("Error! This is not a greyscale image")

        if(img.dtype != np.uint8):
            raise TypeError("Error! This image is not a numpy.uint8 data type")

        hist = np.zeros(256)
        pixelVals = img.astype(np.uint8)

        for row in range(len(pixelVals)):
            for col in range(len(pixelVals[row])):
                pixelVal = pixelVals[row][col]
                hist[pixelVal] = hist[pixelVal] + 1

        return hist

    def adjust_exposure(self, gamma):
        '''Generate a LUT that applies a power-law transform to an image.

        Parameters
        ----------
        gamma : float
            the exponent in the power-law transform; must be a positive value

        Returns
        -------
        numpy.ndarray
            a 256-element LUT that can be provided to ``apply_lut()``

        Raises
        ------
        ValueError
            if ``gamma`` is negative
        '''
        if(gamma < 0.0):
            raise ValueError("Gamma is a negative value")

        lut = np.arange(0, 256, 1, dtype=np.uint8)

        for i in range(len(lut)):
            percentVal = i/255
            newVal = math.pow(percentVal, gamma) * 255

            if(newVal > 255):
                newVal = 255

            if(newVal < 0):
                newVal = 0

            lut[i] = newVal
        print(lut)
        return lut

    def adjust_brightness(self, offset):
        '''Generate a LUT to adjust the image brightness.

        Parameters
        ----------
        offset : int
            the amount to offset brightness values by; this may be negative or
            positive

        Returns
        -------
        numpy.ndarray
            a 256-element LUT that can be provided to ``apply_lut()``
        '''
        lut = np.arange(0, 256, 1)

        for i in range(len(lut)):
            newVal = i + offset
            if(newVal > 255):
                newVal = 255
            if(newVal < 0):
                newVal = 0
            lut[i] = newVal

        return lut.astype(np.uint8)