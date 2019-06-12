from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,help="path to the video file")
args = vars(ap.parse_args())

cap=cv2.VideoCapture(args["video"])
while(cap.isOpened()):
    ret, frame=cap.read()
    #Coger imagen ponerla en gray y luego añadirle el blur
    image = frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# hacer threshold a la imagen para detectar las zonas de luz
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
# usamos Erode y dilate para eliminar todo el ruido posible de cada frame
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

#usamos los labels
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    for label in np.unique(labels):
        if label == 0:
            continue
            print("No se encuentran contornos")

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > 200:
            mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = contours.sort_contours(cnts)[0]

    for (i, c) in enumerate(cnts):
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        center= (int(cX), int(cY))
        height = np.size(mask, 0)
        width = np.size(mask, 1)
        color1, color2, color3=image[int(cY), int(cX)]
        #utilizo otra capa para añadir los circulos y difuminarlos
        copia = image.copy()
        #pongo un if para qitar todos los circulos grandes y los pequeños
        if(radius<35 and radius>10):
            circulo=cv2.circle(copia, center, int(radius+12),(int(color1),int(color2),int(color3)), 4)
            #valor alpha, indica lo opaco que será el circulo
            alpha=0.5
            cv2.addWeighted(copia,alpha,image,1-alpha,0,image)

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
