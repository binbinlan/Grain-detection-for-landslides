

#---------------legacy functions--------------------------------------------------

# def findfg(borders,img,erode=0, adaptive = False):
#     if erode != 0:
#         borders = cv2.erode(borders,cv2.getStructuringElement(cv2.MORPH_CROSS, (erode,erode)), iterations = 1)
#     distance = cleanmask(cv2.distanceTransform(borders,cv2.DIST_L2,5).astype(np.uint8),img)
#     if adaptive :
#         surefg = adaptiveOtsu(distance,3,0.9)
#     else:
#         surefg = cv2.threshold(distance,distance.min() ,distance.max(), cv2.THRESH_OTSU)[1]
#     surefg = surefg.astype(np.uint8)
#     return surefg


# def firstout(markers,img, basename, export = False, rectangles = True,colorize = True,centroids = True,rectcolor = [0,170,255],centroidcolor = [197,97,255]):
#     if export:
#         csvdata = {"area":[],"Lmax":[],"Lmin":[]}
#     h,w = markers.shape
#     marked = img.copy()
#     marked[markers == -1] = centroidcolor
#     markers = rerangemarkers(markers,h,w)
#     rectangular = img.copy()
#     if colorize:
#         colorful = np.zeros((h,w,3), dtype=np.uint8)
#         counter = 0
#         colorpool = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[0,170,255],[0,255,200],[0,255,106],[0,140,255],[115,0,255],[255,0,166],[145,149,255],[153,255,145]]
#     if centroids:
#         centrimg = img.copy()
#         centroid_list = []
#     for i in range(h):
#         for j in range(w):
#             if markers[i,j] == 255:
#                 temporary = np.zeros((h,w), dtype=np.uint8)
#                 mask = cv2.bitwise_not(cv2.copyMakeBorder(markers.astype(np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=255))
#                 retval, temporary, mask, rect = cv2.floodFill(temporary,mask,[j,i],255,10,10)
#                 subimage = temporary.copy()[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
#                 centroid_loc,centr_lines, arealen, axelens = find_centroid(subimage)
#                 if export:
#                     csvdata["area"].append(arealen)
#                     csvdata["Lmax"].append(axelens["Lmax"])
#                     csvdata["Lmin"].append(axelens["Lmin"])
#                 centroid = [int(centroid_loc[0])+rect[0],int(centroid_loc[1])+rect[1]]
#                 markers = cv2.floodFill(markers,cv2.copyMakeBorder(np.zeros((h,w),dtype=np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=0),[j,i],0,10,10)[1]
#                 if rectangles:
#                     rectangular = cv2.rectangle(rectangular, [rect[0],rect[1]], [rect[0]+rect[2],rect[1]+rect[3]], rectcolor, 1)
#                 if colorize:
#                     colorful[temporary==255] = colorpool[counter]
#                     counter+=1
#                     if counter == len(colorpool):
#                         counter = 0
#                 if centroids:
#                     centrimg = cv2.rectangle(centrimg, [centroid[0]-1,centroid[1]-1], [centroid[0]+1,centroid[1]+1], centroidcolor, 2)
#                     for k in range(len(centr_lines)):
#                         cv2.line(centrimg, np.add(np.array(centr_lines[k][0]),np.array(rect[:2])), np.add(np.array(centr_lines[k][1]),np.array(rect[:2])), centroidcolor, 1)
#     if export:
#        df = pd.DataFrame.from_dict(csvdata)
#        df.to_csv(basename+'.csv', index=False)
#     if rectangles:
#         cv2.imshow('rectangles',rectangular)
#     if colorize:
#         newcolor = cv2.addWeighted(colorful, 0.25, img, 0.75, 0)
#         cv2.imshow('colorful',newcolor.astype(np.uint8))
#     if centroids:
#         cv2.imshow('centroids',centrimg.astype(np.uint8))


#---------------legacy functions--------------------------------------------------







