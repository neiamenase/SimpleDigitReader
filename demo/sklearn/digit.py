from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import imageio
from skimage import transform
import skimage 
from PIL import Image
import numpy as np


digits = datasets.load_digits()
# print (digits)
features = digits.data 
labels = digits.target

clf = SVC(gamma = 0.001)
clf.fit(features, labels)

#img = misc.imread("testImage.jpg") deprecated
img = imageio.imread("testImage.jpg")
type = img.dtype
#img = misc.imresize(img, (8,8)) deprecated

imgV = Image.fromarray(img, 'RGB')
imgV.save('b4.png')
imgV.show()

img = transform.resize(img, (8,8))

imgV = Image.fromarray(img, 'RGB')
imgV.save('af.png')
imgV.show()

img = img.astype(digits.images.dtype)
#img = misc.bytescale(img, high=16, low=0) deprecated
# Byte scaling means converting the input image to uint8 dtype and scaling the range to (low, high) (default 0-255). 
# If the input image already has dtype uint8, no scaling is done.
if type != 'uint8':
	skimage.img_as_ubyte(img) ## http://scikit-image.org/docs/dev/user_guide/data_types.html
	#ubyte = uint8
	minV = img.min()
	maxV = imag.max()
	for row in img:
		for elem in row:
			elem = ((elem - minV) / (maxV - minV) * (255 - 0) ) + 0

# w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[256, 256] = img
# img = Image.fromarray(img, 'RGB')
# img.save('my.png')
# img.show()


x_test = []

for eachRow in img:
	for eachPixel in eachRow:
		x_test.append(sum(eachPixel)/3.0)



print(clf.predict([x_test]))