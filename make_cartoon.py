import argparse
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from cartoon_effect import CartoonImage
import numpy as np

if __name__ == "__main__":
    # Generate arguments and parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  help="Input file",  type=str, required=True)
    parser.add_argument("--output", help="Output file", type=str, required=True)
    parser.add_argument("--meanXDoG", help="Mean value for Extended Difference of Gaussian", type=float)
    parser.add_argument("--pXDoG", help="Mean value for Extended Difference of Gaussian", type=float)
    parser.add_argument("--threshold", help="Thresholding value for Extended Difference of Gaussian", type=float)
    parser.add_argument("--blurring", help="Gaussian blurring size", type=float)
    parser.add_argument("--levels", help="Numbber of levels for generating level set", type=int)

    args = parser.parse_args()

    # DEFAULT PARAMETER FOR CARTOON VARS
    inputP = ""
    outputP = "output.jpg"
    pXDoG = 16
    levels = 10
    meanXDoG = 1.1
    blurring = 15
    threshold = 0.3


    # Verify variables
    if not args.input or not args.output:
        raise Exception("No input and/or output file provided")

    # Assign argument variables
    inputP = args.input
    outputP = args.output
    
    if(args.pXDoG):
        pXDoG = args.pXDoG
    if(args.levels):
        levels = args.levels
    if(args.meanXDoG):
        meanXDoG = args.meanXDoG
    if(args.blurring):
        blurring = args.blurring
    if(args.threshold):
        threshold = args.threshold
    
    image = imread(inputP)
    

    imgWithEffect = CartoonImage(image, meanXDoG, pXDoG, threshold, blurring, levels).applyEffect()
    
    if(image.ndim == 3):
        rgbSetup = gray2rgb(imgWithEffect)
        
        for row in range(len(rgbSetup)):
            for col in range(len(rgbSetup[row])):
                for colour in range(len(rgbSetup[row][col])):
                    rgbSetup[row][col][colour] = rgbSetup[row][col][colour] + (image[row][col][colour])
        rgbSetup = (rgbSetup/rgbSetup.max()) * 255
        imsave(outputP, rgbSetup)
        
    else:
        outGreyImg = np.copy(image)
        for row in range(len(outGreyImg)):
            for col in range(len(outGreyImg[row])):
                outGreyImg[row][col] = imgWithEffect[row][col] + image[row][col]

        imsave(outputP, outGreyImg)
