import numpy as np
import math
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
import cv2
import os
from PIL import Image
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import shutil
from scipy.spatial.distance import pdist
import random as rnd


class obj_exporter:
    def __init__(self,name,pointlist):
        self.name = name
        self.pointlist = pointlist
        self.colors = False
        self.triangle_cutover =5
        self.vertex_cutover = 5

    def export(self):
        self.makelists()
        self.remove_abnormal_vertex()
        self.tri = Delaunay(np.array(self.coo2d))
        self.remove_abnormal_triangles()
        with open(self.name+'.obj', 'w') as f:
            self.write_vertex(f)
            self.write_faces(f)
        print('operation successfull')
        f.close()

    def export_with_textures(self,cameralist):
        photodim =cv2.imread(self.photoname)
        dimensions =[photodim.shape[1], photodim.shape[0]]
        del photodim
        cameraid = -1
        
        for i in range(len(cameralist)):
            if cameralist[i]['name'] == os.path.basename(self.photoname):
                cameraid = cameralist[i]['ID']
        if cameraid == -1:
            print('ERROR!! check the name of the photo')
            return
        self.makelists()
        self.remove_abnormal_vertex()
        self.tri = Delaunay(np.array(self.coo2d))
        self.remove_abnormal_triangles()
        with open(self.name+'.mtl', 'w') as f:
            f.write('newmtl '+str(os.path.basename(self.photoname))+'\n')
            f.write('Ka 1.000 1.000 1.000\n')
            f.write('Kd 1.000 1.000 1.000\n')
            f.write('Ks 0.000 0.000 0.000\n')
            f.write('Ns 10.000\n')
            f.write('map_Ka '+str(os.path.basename(self.photoname))+'\n')
            f.write('map_Kd '+str(os.path.basename(self.photoname))+'\n')
        f.close()
        with open(self.name+'.obj', 'w') as f:
            f.write('mtllib '+str(os.path.basename(self.photoname))+'.mtl\n')
            self.write_vertex(f)
            print('writing texture coordinates...')
            for i in range(len(self.coordlist)):
                for j in self.pointlist[self.goodpoint_index[i]]['cameras']:
                    if int(j['ID'])==int(cameraid):
                        f.write('vt '+str(float(j['xpix'])/dimensions[0])+' '+str(1-float(j['ypix'])/dimensions[1])+' '+'\n')
            f.write('usemtl '+os.path.basename(self.photoname)+'\n')
            self.write_faces(f,textures=True)
        print('operation successfull')
        f.close()

    def makelists(self):
        self.idlist = []
        self.coordlist = []
        self.coo2d = []
        self.colorlist = []
        for i in self.pointlist:
            self.idlist.append(i['ID'])
            self.coordlist.append([float(s) for s in i['xyz']])
            self.coo2d.append(i['xyz'][:2])
            if self.colors:
                self.colorlist.append(i['RGB'])

    def remove_abnormal_vertex(self):
        npointlist = np.array(self.coordlist)
        tree = KDTree(self.coo2d)
        print('removing abnormal vertex...')
        old_lists = [self.coordlist,self.coo2d]
        new_lists = [[],[],[]]
        if self.colors:
            old_lists.append(self.colorlist)
        self.goodpoint_index = []
        for i in range(len(npointlist)):
            _, indices = tree.query(self.coo2d[i], k=16)
            allneighbours = np.array([npointlist[j] for j in indices])
            neighbours = np.delete(allneighbours, np.where((allneighbours == npointlist[i]).all(axis=1)), axis=0)
            a,b,c,d = self.best_fit_plane(neighbours)
            distance = self.point_plane_dist(list(allneighbours),a,b,c,d)
            index_anal = np.where((allneighbours == npointlist[i]).all(axis=1))[0][0]
            for j in range(len(distance)):
                if distance[j]<0:
                    distance[j] = -distance[j]
            anal = distance.pop(index_anal)
            avgdist = sum(distance)/len(distance) 
            if anal< self.vertex_cutover*avgdist:
                self.goodpoint_index.append(i)
                for j in range(len(old_lists)):
                    new_lists[j].append(old_lists[j][i])
        self.coordlist,self.coo2d,self.colorlist = new_lists
    
    def remove_abnormal_triangles(self):
        print('removing abnormal triangles...')
        self.clean_triangles = []
        areas = []
        for i in range(len(self.tri.simplices)):
            points = [self.coordlist[j] for j in self.tri.simplices[i]]
            areas.append(area_of_triangle_3d(points))
        average =sum(areas)/len(areas)
        for i in range(len(self.tri.simplices)):
            if areas[i]/average<=self.triangle_cutover:
                self.clean_triangles.append(self.tri.simplices[i])

    def write_vertex(self,f):
        f.write('o mesh3d\n')
        print('exporting vertex...')
        for i in range(len(self.coordlist)):
            f.write('v')
            for j in range(len(self.coordlist[i])):
                f.write(" {:.6f}".format(float(self.coordlist[i][j])))
            if self.colors:
                for j in range(len(self.coordlist[i])):
                    f.write(' '+str(self.colorlist[i][j]))
                f.write('\n')
            else:
                f.write('\n')

    def write_faces(self,f, textures = False):
        print('exporting faces...')
        f.write('s off\n')
        for i in range(len(self.clean_triangles)):
            f.write('f')
            for j in range(len(self.clean_triangles[i])):
                pointnum = str(self.clean_triangles[i][j]+1)
                if textures:
                    f.write(' '+pointnum+'/'+pointnum)
                else:
                    f.write(' '+pointnum)
            f.write('\n')
    
    def best_fit_plane(self,points):
        centroid = np.mean(points, axis=0)
        cov = np.cov(points.T)
        _, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]
        normal /= np.linalg.norm(normal)
        D = -np.dot(normal, centroid)
        return normal[0], normal[1], normal[2], D
    
    def point_plane_dist(self,points,a,b,c,d):
        distance =[]
        for i in points:
            plane_normal = np.array([a, b, c])
            point_on_plane = np.array([0,0,-d/c])
            distance.append(abs(np.dot(plane_normal, i - point_on_plane)) / np.linalg.norm(plane_normal))
        return distance

def haversine(coord_a, coord_b):
    lat1, lon1, alt1= list(coord_a)
    lat2, lon2, alt2= list(coord_b)
    # Radius of the Earth in meters
    earth_radius = 6371000  # Approximately 6,371 km
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    dalt = alt2 - alt1
    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    # Distance in meters
    distance = earth_radius * c
    # Add altitude difference
    distance = math.sqrt(distance**2 + dalt**2)
    return distance

def gps_to_float(gpscoord):
    gpscoord = list(float(i) for i in gpscoord)
    for figure in [2,1]:
        gpscoord[figure-1]+=gpscoord[figure]/60
    return gpscoord[0]

def scale_factor(cameralist, photofolder, samples = 100):
    exiflist = list_exif(photofolder)
    modelcoordlist = []
    realcoordlist = []
    for nameind in range(len(exiflist['exif'])):
        namephoto = exiflist['exif'][nameind]['photoname']
        for camera in cameralist:
            if camera['name'] == namephoto:
                modelcoordlist.append([float(x) for x in camera['xyz']])
                realdata = exiflist['exif'][nameind]['data'][34853]
                realcoordlist.append([gps_to_float(realdata[2]),gps_to_float(realdata[4]),float(realdata[6])])

    listfactors = []
    for sample in range(samples):
        a = round(rnd.random()*(len(modelcoordlist)-1))
        b = round(rnd.random()*(len(modelcoordlist)-1))
        while b==a:
            b = round(rnd.random()*(len(modelcoordlist)-1))
        
        modeldist = pdist([modelcoordlist[a],modelcoordlist[b]])[0]
        realdist = haversine(realcoordlist[a],realcoordlist[b])
        listfactors.append(realdist/modeldist)
    return float(listfactors[round(samples/2)])

def nvm_to_points(path,photofolder, separator, export_cameras = True):
    with open(path, "r") as file:
        contents = file.read().split('\n')
        pointnum = {'cameras':0,'points':0}
        camerakeys =['ID','unknown','xpix','ypix']
        inilist =[]
        pointlist =[]
        idnum =0
        for i in contents:
            last = i.split(separator)
            if pointnum['cameras']==0 or pointnum['points']==0:
                inilist.append(last)
                afloat = 0.1
                try:
                    afloat = float(last[0])
                except:
                    pass
                if afloat%1==0:
                    if pointnum['cameras']==0:
                        pointnum['cameras']= int(afloat)
                    elif pointnum['points']==0:
                        pointnum['points']= int(afloat)
            else:
                if idnum < pointnum['points']:
                    last[2]=str(-float(last[2])) # invert Z
                    last[1]=str(-float(last[1])) # invert Y
                    apoint = {'ID':idnum,'xyz':last[:3],'RGB':last[3:6],'cameras':[]}
                    for j in range(int(last[6])):
                        acamera ={}
                        for k in range(len(camerakeys)):
                            acamera[camerakeys[k]]=last[7+(j*len(camerakeys))+k]
                        apoint['cameras'].append(acamera)
                    idnum+=1
                    pointlist.append(apoint)
    
    cameralist = []
    if export_cameras:
        record =False
        for i in range(len(contents)):
            newline = contents[i].split(' ')
            if record:
                cameralist.append(newline)
                if newline[0] ==str(pointnum['points']):
                    record = False
            if newline[0] ==str(pointnum['cameras']):
                record = True
        while len(cameralist[-1])<len(cameralist[0]):
            del cameralist[-1]
        for i in range(len(cameralist)):
            cameralist[i][7] = str(-float(cameralist[i][7]))
            cameralist[i][8] = str(-float(cameralist[i][8]))
            apoint ={'ID':i,'name':cameralist[i][0],'rotations(rx,ry,rz)':cameralist[i][3:6],'xyz':cameralist[i][6:9]}
            cameralist[i]=apoint

    scalefactor = scale_factor(cameralist, photofolder)

    for pn in range(len(pointlist)):
        pointlist[pn]['xyz']= [float(dim)*scalefactor for dim in pointlist[pn]['xyz']]
    for cm in range(len(cameralist)):
        cameralist[cm]['xyz']= [float(dim)*scalefactor for dim in cameralist[cm]['xyz']]  
    return pointlist,cameralist,pointnum

def photo_in_model(photoname,alldata):
    # photoname = os.path.basename(photoname)
    found =False
    for i in range(len(alldata[1])):
        if alldata[1][i]['name']== photoname:
            camera_anal= alldata[1][i]
            found = True
    if found:
        return camera_anal
    else:
        return 'camera not found'

def export_singlephoto(photoname,alldata,exportname = 'same',textures=False):
    seenpoint = []
    camera_anal = photo_in_model(os.path.basename(photoname),alldata)
    for i in range(len(alldata[0])):
        liscameraid =[]
        for j in alldata[0][i]['cameras']:
            liscameraid.append(j['ID'])
        # print(camera_anal)
        if str(camera_anal['ID']) in liscameraid:
            seenpoint.append(alldata[0][i])
    meshwriter = obj_exporter(photoname,seenpoint)
    if exportname == 'same':
        meshwriter.name = photoname
    else:
        meshwriter.name =  exportname
    meshwriter.triangle_cutover = 20
    if textures:
        meshwriter.photoname = photoname
        meshwriter.export_with_textures(alldata[1])
    else:
        meshwriter.export()
    return meshwriter

def area_of_triangle_3d(triangle):
    """
    Calculate the area of a triangle in 3D space given three points.
    """
    v1 = np.array(triangle[1]) - np.array(triangle[0])
    v2 = np.array(triangle[2]) - np.array(triangle[0])
    try:
        cross_product = np.cross(v1, v2)
    except:
        print('there is a problem with', v1,v2)
    magnitude = np.linalg.norm(cross_product)
    area = 0.5 * magnitude
    return area

def catalogue_in_model(nvmmodel,discrim_path):
    '''in some cases, Colmap cannot create a single model with all the photos taken, this is perhaps due to the sudden change of direction of the drone.
      this function separates the pictures in a model (nvmmodel) from those that have been discarded from the original folder (discrim_path)'''
    import os
    alldata = nvm_to_points(nvmmodel,' ')
    listimage = os.listdir(discrim_path)
    newfolder = r'C:\Users\lovam\Documents\sklgp_lit\3d_drone\DJI_202304211714_024_onlyglacier'
    newfolder = os.path.join(newfolder,nvmmodel[:-4])
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)
    for i in listimage:
        photoinmod = photo_in_model(i,alldata)
        if photoinmod != 'camera not found':
            oldpath = os.path.join(discrim_path,i)
            newpath = os.path.join(newfolder,i)
            os.rename(oldpath,newpath)

def list_exif(image_path):
    exiflist ={'exif':[],'no exif':[]}
    def read_exif_data(image_path):
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            if exif_data is not None:
                return exif_data
            else:
                return 'null'
        except Exception:
            
            return 'null'

    listimage = os.listdir(image_path)
    for i in listimage:
        exif = read_exif_data(os.path.join(image_path,i))
        if exif != 'null':
            exiflist['exif'].append({'photoname':i,'data':exif})
        else:
            exiflist['no exif'].append(i)
    return exiflist

def flightdata(image_path, mode = 'path'):
    def ordinaltime(timestamp):
        timestamp = timestamp.split(' ')[1]
        raword = timestamp.split(':')
        return int(raword[0])*3600+int(raword[1])*60+int(raword[2])

    def decimal_coord(rawcoord):
        return float(float(rawcoord[0])+(((float(rawcoord[1])*60)+float(rawcoord[2]))/3660))

    exiflist = list_exif(image_path) 
    flypath = []
    for i in range(len(exiflist['exif'])):
        flypath.append({'ordinaltime':0,'latitude':0,'longitude':0,'altitude':0})
        flypath[-1]['ordinaltime'] = ordinaltime(exiflist['exif'][i]['data'][36867])
        flypath[-1]['latitude']= decimal_coord(exiflist['exif'][i]['data'][34853][2])
        flypath[-1]['longitude']= decimal_coord(exiflist['exif'][i]['data'][34853][4])
        flypath[-1]['altitude'] = exiflist['exif'][i]['data'][34853][6]
    if mode =='path':
        return flypath
    elif mode == 'vectors':
        def deltacoord(path, point,coordinate):
            onedeg = 111319.9 #meters
            deltacoord = onedeg* float(path[point][coordinate]-path[point-1][coordinate])
            if coordinate == 'longitude':
                deltacoord *=math.cos(path[point]['latitude'])
            return deltacoord
        
        def unitize_vector(vector):
            norm = np.linalg.norm(vector)  # Calculate the Euclidean norm of the vector
            if norm == 0:  # Avoid division by zero
                return vector
            return vector / norm  # Divide the vector by its norm
    
        def clusterize(vectors):
            from sklearn.cluster import KMeans
            num_clusters = 4 #vectors are 4 because of the exif 274 (orientation) without mirrors
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(vectors)
            cluster_assignments = list(kmeans.labels_)
            vectorgroups = [[],[],[],[]]
            for i in range(len(vectors)):
                vectorgroups[cluster_assignments[i]].append(vectors[i])
            lenghts = []
            for i in range(4):
                lenghts.append(cluster_assignments.count(i))
            maingroup = lenghts.index(max(lenghts))
            mainvector = [0,0]
            for i in range(len(vectors)):
                if cluster_assignments[i]==maingroup:
                    for j in range(len(mainvector)):
                        mainvector[j]+=vectors[i][j]
            return cluster_assignments,maingroup, unitize_vector(mainvector)

        def angle_between_vectors(vectors_a, vectors_b):
            dot_product = vectors_a[0] * vectors_b[0] + vectors_a[1] * vectors_b[1]
            angle_radians = math.acos(dot_product)
            return math.degrees(angle_radians)

        vectorpath = []
        vectorlist = []
        for i in range(1,len(exiflist['exif'])):
            coordvect = unitize_vector([deltacoord(flypath,i,'latitude'),deltacoord(flypath,i,'longitude')])
            vectorlist.append(coordvect)
        vectorlabels,maingroup,mainvector = clusterize(vectorlist)
        
        for i in range(1,len(exiflist['exif'])):
            angle = 0
            if vectorlabels[i-1]!=maingroup:
                angle = round(angle_between_vectors(mainvector,vectorlist[i-1])/90,0)*90
            vectorpath.append({'photoname':exiflist['exif'][i]['photoname'],'ordinaltime':ordinaltime(exiflist['exif'][i]['data'][36867]),'vector':vectorlist[i-1],'group':vectorlabels[i-1], 'rotation':angle })
        return vectorpath
    
def rotate_images(image_path):
    def rotate_one_deprec(oldpath,newpath, angle):
        image = cv2.imread(oldpath)
        height, width = image.shape[:2]
        center_rota = [width/2, height/2]
        if angle in [90,270]:
            maxdim = max(image.shape[:2])
            center_rota[image.shape[:2].index(min(image.shape[:2]))]+= (center_rota[0]-center_rota[1])*math.copysign(1,180-angle)
            rotation_matrix = cv2.getRotationMatrix2D(center_rota, angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, [maxdim,maxdim])
        else:
            rotation_matrix = cv2.getRotationMatrix2D(center_rota, angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        cv2.imwrite(newpath,rotated_image)
        return
    
    def rotate_one(oldpath,newpath, angle):
        image = cv2.imread(oldpath)
        height, width = image.shape[:2]
        canvatype = 'square'
        if angle in [90,270] and height != width:
            maxdim = max(image.shape[:2])
            padding = abs(width - height) // 2
            canvas = np.zeros((maxdim, maxdim, 3), dtype=np.uint8)
            if width > height:
                canvatype = 'landscape'
                canvas[padding:padding+height, :, :] = image
            else:
                canvatype = 'portrait'
                canvas[:, padding:padding+width, :] = image
            del image
            image = canvas
        height, width = image.shape[:2]
        center_rota = [width/2, height/2]
        rotation_matrix = cv2.getRotationMatrix2D(center_rota, angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        if angle in [90,270] and canvatype !='square':
            if canvatype == 'landscape':
                rotated_image = rotated_image[:,padding:width-padding]
            else:
                rotated_image = rotated_image[padding:height-padding,:]
        cv2.imwrite(newpath,rotated_image)
        return
            
    rotations = flightdata(image_path, mode = 'vectors')
    newfolder = os.path.join(image_path,'rotated')
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)
    for i in rotations:
        oldpath = os.path.join(image_path,i['photoname'])
        newpath = os.path.join(newfolder,i['photoname'])
        if not os.path.exists(newpath):
            if i['rotation'] == 0:
                os.rename(oldpath,newpath)
            else:
                rotate_one(oldpath,newpath,i['rotation'])
                # Image.open(oldpath).rotate(i['rotation']).save(newpath)

def yesno_popup(title,text):
    result = messagebox.askyesno(title, text)
    if result:
        return True
    else:
        return False

if __name__ == '__main__':
    print('select the photo to analize')
    pathfile = filedialog.askopenfilename()
    imagename = os.path.basename(pathfile)
    alldata = nvm_to_points('model.nvm', ' ')
    camera_anal = photo_in_model(imagename,alldata)
    if camera_anal!= 'camera not found':
        vegmask = yesno_popup('vegetation mask','do you want to generate a vegetation mask?')
        shutil.copy(pathfile, imagename)
        if vegmask:
            import maskmkr as mmkr
            img = cv2.imread(pathfile)
            landslide_colors,mask_colors = mmkr.selectcolors(img)
            print('generating the mask...')
            mask = mmkr.genmask(img,landslide_colors,mask_colors)
            print('exporting the masked texture...')
            mmkr.mergemask(img,mask,exportname = imagename)
        export_singlephoto(imagename, alldata,textures=True)
    else:
        print('3D model absent')
# meshwriter = obj_exporter('trial',nvm_to_points('valley_complete.nvm', ' ')[0])
# meshwriter.triangle_cutover = 5
# meshwriter.vertex_cutover = 5
# meshwriter.export()
# image_path = r'C:\Users\lovam\Documents\sklgp_lit\3d_drone\DJI_202304211714_024_onlyglacier'
# rotate_images(image_path)