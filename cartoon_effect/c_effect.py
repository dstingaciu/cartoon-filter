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