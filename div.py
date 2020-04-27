import main as mn
import cv2 as cv
import sys
path = r'/Users/idel_isbaev/PycharmProjects/IAI_ass/images/img_03.png'

input_image = cv.imread(path)[0:int(mn.pixs), 0:int(mn.pixs), :]
if input_image is None:
    sys.exit("Could not read the image.")
output_image = mn.run_algo(input_image)

print(mn.iter)

cv.imshow('iskustvo',input_image)
cv.imshow('iskustvo',output_image)
cv.waitKey(0)
cv.destroyAllWindows()

