import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from skimage import data
from skimage.measure import label
from skimage.measure import regionprops
#
import statsmodels.api as sm

from scipy import misc
import skimage
import scipy.ndimage as scind
import cellprofiler.object as cpo
from centrosome.cpmorphology import ellipse_from_second_moments_ijv
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.cpmorphology import calculate_extents
from centrosome.cpmorphology import calculate_perimeters
from centrosome.cpmorphology import calculate_solidity
from centrosome.cpmorphology import euler_number
from centrosome.cpmorphology import distance_to_edge
from centrosome.cpmorphology import maximum_position_of_labels
from centrosome.cpmorphology import median_of_labels
from centrosome.cpmorphology import feret_diameter
from centrosome.cpmorphology import convex_hull_ijv
import skimage.measure
import scipy.io as sio
import time
import pdb
import cPickle
import glob
from mlgmm_hy import data2gmmp

from load_tiffile_package import tif2resolution
# plt.rcParams['figure.constrained_layout.use'] = True
def cdfplot(temp):
    ecdf = sm.distributions.ECDF(temp)
    x = np.linspace(min(temp), max(temp), 1000)
    y = ecdf(x)
    return [x, y]
def extract_morphological_features(objects):
    if len(objects.shape) is 2:
        #
        # Do the ellipse-related measurements
        #
        i, j, l = objects.ijv.transpose()
        centers, eccentricity, major_axis_length, minor_axis_length, \
        theta, compactness = \
            ellipse_from_second_moments_ijv(i, j, 1, l, objects.indices, True)
        del i
        del j
        del l

        is_first = False
        if len(objects.indices) == 0:
            nobjects = 0
        else:
            nobjects = np.max(objects.indices)

        marea = objects.areas
        meccentricity = eccentricity
        major_axis_length = major_axis_length
        minor_axis_length = minor_axis_length
        mcompactness = compactness
        morientation = theta * 180 / np.pi

        mcenter_x = np.zeros(nobjects)
        mcenter_y = np.zeros(nobjects)
        mextent = np.zeros(nobjects)
        mperimeters = np.zeros(nobjects)
        msolidity = np.zeros(nobjects)
        meuler = np.zeros(nobjects)
        max_radius = np.zeros(nobjects)
        median_radius = np.zeros(nobjects)
        mean_radius = np.zeros(nobjects)
        min_feret_diameter = np.zeros(nobjects)
        max_feret_diameter = np.zeros(nobjects)
        mformfactor = np.zeros(nobjects)
        # eccentricity, major_axis_length, minor_axis_length,  theta, compactness, formfactor, objects.areas
        zf = {}
        if nobjects > 0:
            chulls, chull_counts = convex_hull_ijv(objects.ijv, objects.indices)
            for labels, indices in objects.get_labels():
                to_indices = indices - 1
                distances = distance_to_edge(labels)
                mcenter_y[to_indices], mcenter_x[to_indices] = \
                    maximum_position_of_labels(distances, labels, indices)
                max_radius[to_indices] = fix(scind.maximum(
                    distances, labels, indices))
                mean_radius[to_indices] = fix(scind.mean(
                    distances, labels, indices))
                median_radius[to_indices] = median_of_labels(
                    distances, labels, indices)
                #
                # The extent (area / bounding box area)
                #
                mextent[to_indices] = calculate_extents(labels, indices)
                #
                # The perimeter distance
                #
                mperimeters[to_indices] = calculate_perimeters(labels, indices)
                #
                # Solidity
                #
                msolidity[to_indices] = calculate_solidity(labels, indices)
                #
                # Euler number
                #
                meuler[to_indices] = euler_number(labels, indices)
                #
            #
            # Form factor
            #
            mformfactor = 4.0 * np.pi * objects.areas / mperimeters ** 2
            #
            # Feret diameter
            #
            min_feret_diameter, max_feret_diameter = \
                feret_diameter(chulls, chull_counts, objects.indices)

        else:
            mformfactor = np.zeros(0)

    else:
        labels = objects.segmented
        props = skimage.measure.regionprops(labels)
        # Area
        areas = [prop.area for prop in props]
        # Extent
        extents = [prop.extent for prop in props]
        # Centers of mass
        import mahotas
        if objects.has_parent_image:
            image = objects.parent_image

            data = image.pixel_data

            spacing = image.spacing
        else:
            data = np.ones_like(labels)

            spacing = (1.0, 1.0, 1.0)

        centers = mahotas.center_of_mass(data, labels=labels)

        if np.any(labels == 0):
            # Remove the 0-label center of mass
            centers = centers[1:]

        center_z, center_x, center_y = centers.transpose()

        # Perimeters
        perimeters = []

        for label in np.unique(labels):
            if label == 0:
                continue

            volume = np.zeros_like(labels, dtype='bool')

            volume[labels == label] = True

            verts, faces, _, _ = skimage.measure.marching_cubes(
                volume,
                spacing=spacing,
                level=0
            )

            perimeters += [skimage.measure.mesh_surface_area(verts, faces)]
    return [marea, meccentricity, major_axis_length, minor_axis_length, mcompactness, morientation, mcenter_x,
            mcenter_y, mextent,
            mperimeters, msolidity, meuler, max_radius, median_radius, mean_radius, min_feret_diameter,
            max_feret_diameter, mformfactor]
def image2feature(segmentedImage, resolutionpx):
    bw_image = misc.imread(segmentedImage) - 1
    # label image regions
    label_image = label(bw_image, neighbors=8)
    ##
    objects = cpo.Objects()
    objects.segmented = label_image
    #
    [marea, meccentricity, major_axis_length, minor_axis_length, mcompactness, morientation, mcenter_x, mcenter_y,
     mextent,
     mperimeters, msolidity, meuler, max_radius, median_radius, mean_radius, min_feret_diameter, max_feret_diameter,
     mformfactor] = extract_morphological_features(objects)
    # filter out the objects by the equvilent diameter: should be between 8 and 70 micrometer.
    index_area = range(len(marea))
    index_area = (np.sqrt(marea / np.pi) * 2 < 70) & (np.sqrt(marea / np.pi) * 2 > 8)
    # there is ONE important output,
    # index_true: index of objects lying in the biological realistic range
    index_true = np.where(index_area)
    index_true = index_true[0]
    #
    if len(index_true) < 2:
        temp_marea = marea[index_area] * resolutionpx ** 2
        #
        temp_meccentricity = meccentricity[index_area]
        #
        temp_major_axis_length = major_axis_length[index_area] * resolutionpx
        #
        temp_minor_axis_length = minor_axis_length[index_area] * resolutionpx
        #
        temp_mcompactness = mcompactness[index_area]
        #
        temp_morientation = morientation[index_area]
        #
        temp_mcenter_x = mcenter_x[index_area]
        #
        temp_mcenter_y = mcenter_y[index_area]
        #
        temp_mextent = mextent[index_area]
        #
        temp_mperimeters = mperimeters[index_area] * resolutionpx
        #
        temp_msolidity = msolidity[index_area]
        #
        temp_meuler = meuler[index_area]
        #
        temp_max_radius = max_radius[index_area] * resolutionpx
        #
        temp_median_radius = median_radius[index_area] * resolutionpx
        #
        temp_mean_radius = mean_radius[index_area] * resolutionpx
        #
        temp_min_feret_diameter = min_feret_diameter[index_area] * resolutionpx
        #
        temp_max_feret_diameter = max_feret_diameter[index_area] * resolutionpx
        #
        temp_mformfactor = mformfactor[index_area]
        #
        x_marea = [];
        y_marea = [];
        x_meccentricity = [];
        y_meccentricity = [];
        x_major_axis_length = [];
        y_major_axis_length = [];
        x_minor_axis_length = [];
        y_minor_axis_length = [];
        x_mcompactness = [];
        y_mcompactness = [];
        x_morientation = [];
        y_morientation = [];
        x_mcenter_x = y_mcenter_x = [];
        x_mcenter_y = y_mcenter_y = x_mextent = y_mextent = x_mperimeters = y_mperimeters = []
        x_msolidity = y_msolidity = x_meuler = y_meuler = x_max_radius = y_max_radius = []
        x_median_radius = y_median_radius = x_mean_radius = y_mean_radius = x_min_feret_diameter = y_min_feret_diameter = []
        x_max_feret_diameter = y_max_feret_diameter = x_mformfactorr = y_mformfactor = [];

    else:
        temp_marea = marea[index_area] * resolutionpx ** 2
        x_marea, y_marea = cdfplot(temp_marea)
        #
        temp_meccentricity = meccentricity[index_area]
        x_meccentricity, y_meccentricity = cdfplot(temp_meccentricity)
        #
        temp_major_axis_length = major_axis_length[index_area] * resolutionpx
        x_major_axis_length, y_major_axis_length = cdfplot(temp_major_axis_length)
        #
        temp_minor_axis_length = minor_axis_length[index_area] * resolutionpx
        x_minor_axis_length, y_minor_axis_length = cdfplot(temp_minor_axis_length)
        #
        temp_mcompactness = mcompactness[index_area]
        x_mcompactness, y_mcompactness = cdfplot(temp_mcompactness)
        #
        temp_morientation = morientation[index_area]
        x_morientation, y_morientation = cdfplot(temp_morientation)
        #
        temp_mcenter_x = mcenter_x[index_area]
        x_mcenter_x, y_mcenter_x = cdfplot(temp_mcenter_x)
        #
        temp_mcenter_y = mcenter_y[index_area]
        x_mcenter_y, y_mcenter_y = cdfplot(temp_mcenter_y)
        #
        temp_mextent = mextent[index_area]
        x_mextent, y_mextent = cdfplot(temp_mextent)
        #
        temp_mperimeters = mperimeters[index_area] * resolutionpx
        x_mperimeters, y_mperimeters = cdfplot(temp_mperimeters)
        #
        temp_msolidity = msolidity[index_area]
        x_msolidity, y_msolidity = cdfplot(temp_msolidity)
        #
        temp_meuler = meuler[index_area]
        x_meuler, y_meuler = cdfplot(temp_meuler)
        #
        temp_max_radius = max_radius[index_area] * resolutionpx
        x_max_radius, y_max_radius = cdfplot(temp_max_radius)
        #
        temp_median_radius = median_radius[index_area] * resolutionpx
        x_median_radius, y_median_radius = cdfplot(temp_median_radius)
        #
        temp_mean_radius = mean_radius[index_area] * resolutionpx
        x_mean_radius, y_mean_radius = cdfplot(temp_mean_radius)
        #
        temp_min_feret_diameter = min_feret_diameter[index_area] * resolutionpx
        x_min_feret_diameter, y_min_feret_diameter = cdfplot(temp_min_feret_diameter)
        #
        temp_max_feret_diameter = max_feret_diameter[index_area] * resolutionpx
        x_max_feret_diameter, y_max_feret_diameter = cdfplot(temp_max_feret_diameter)
        #
        temp_mformfactor = mformfactor[index_area]
        x_mformfactorr, y_mformfactor = cdfplot(temp_mformfactor)
    # first 36 store the x, y axis of the features.
    # the last 18 store the feature themselves
    return [x_marea, y_marea, x_meccentricity, y_meccentricity, x_major_axis_length, y_major_axis_length,
            x_minor_axis_length, y_minor_axis_length, x_mcompactness, y_mcompactness, x_morientation, y_morientation,
            x_mcenter_x, y_mcenter_x, x_mcenter_y, y_mcenter_y,
            x_mextent, y_mextent, x_mperimeters, y_mperimeters, x_msolidity, y_msolidity,
            x_meuler, y_meuler, x_max_radius, y_max_radius, x_median_radius, y_median_radius,
            x_mean_radius, y_mean_radius, x_min_feret_diameter, y_min_feret_diameter,
            x_max_feret_diameter, y_max_feret_diameter,
            x_mformfactorr, y_mformfactor],[
            temp_marea, temp_meccentricity, temp_major_axis_length,
            temp_minor_axis_length, temp_mcompactness, temp_morientation, temp_mcenter_x, temp_mcenter_y, temp_mextent,
            temp_mperimeters, temp_msolidity,
            temp_meuler, temp_max_radius, temp_median_radius, temp_mean_radius, temp_min_feret_diameter,
            temp_max_feret_diameter, temp_mformfactor, index_true]
def imagetoprop(label_image, a_sorted, index_label, yclist):
    #
    #
    # Resolution
    #
    sResolution = 1024
    bmimage_cluster1 = np.zeros([sResolution, sResolution], dtype=np.uint8);
    bmimage_cluster2 = np.zeros([sResolution, sResolution], dtype=np.uint8);
    #
    index_label_cluster1 = index_label[yclist[0:len(index_label)] == a_sorted[0]]
    index_label_cluster2 = index_label[yclist[0:len(index_label)] == a_sorted[1]]
    #
    for labeli in range(0, len(index_label_cluster1)):
        bmimage_cluster1[label_image == (index_label_cluster1[labeli] + 1)] = 1
    #
    for labeli in range(0, len(index_label_cluster2)):
        bmimage_cluster2[label_image == (index_label_cluster2[labeli] + 1)] = 1

    label_image_c1 = label(bmimage_cluster1, neighbors=8)
    tempprop1 = regionprops(label_image_c1)
    label_image_c2 = label(bmimage_cluster2, neighbors=8)
    tempprop2 = regionprops(label_image_c2)

    return tempprop1, tempprop2, bmimage_cluster1, bmimage_cluster2
filenameA= './allGMM_featureM.cp'
with open(filenameA, 'rb') as fp:
    varName = cPickle.load(fp)
    feature_matrix = cPickle.load(fp)
    bestgmm_A = cPickle.load(fp)


newcolors = [[0, 1,1,1],
            [1, 1, 0 ,0],
            [2, 0,0,1]]
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
newcmp = ListedColormap(newcolors)
newcmp2 = LinearSegmentedColormap.from_list('custom cluster',
                                             [(0,    '#ffffff'),
                                              (0.5, '#ff0000'),
                                              (1,    '#0000ff')], N=256)

newRed = LinearSegmentedColormap.from_list('custom red',
                                             [(0,    '#ffffff'),
                                              (1., '#ff0000')], N=256)


newBlue = LinearSegmentedColormap.from_list('custom blue',
                                             [(0,    '#ffffff'),
                                             (1,    '#0000ff')], N=256)


feature_list = np.array([36, 45, 46, 53]) - 36

nFragmentedMitoA =[]
nFusedMitoA = []
nTMitoA = []
aSumFragmentedMitoA = []
aSumFusedMitoA= []

dfWell4training = pd.read_csv('./202004_WellSite4training_updatedAfterSegOnlyKnownTreatment.csv')


'''
'''
for i, filegraySearchi in enumerate(dfWell4training['grayMitoImages']):
    fileList = sorted(glob.glob(filegraySearchi))
    if len(fileList) == 0:
        print('no file')
    else:
        grayMitoImageTmp = fileList[0]
        # grayMitoImageTmpList.append(grayMitoImageTmp)

    print(grayMitoImageTmp)

    resolutionpx  = tif2resolution(grayMitoImageTmp)
    segmentedMitoImage = dfWell4training.iloc[i]['segmentedMitoImages']
    xyfeature, feature_image = image2feature(segmentedMitoImage, resolutionpx)

    bw_image = misc.imread(segmentedMitoImage) - 1
    # label image regions
    label_image = label(bw_image, neighbors=8)
    a = bestgmm_A.means_[:, 0]  # only the area of the clusters
    a_sorted = [b[0] for b in sorted(enumerate(a), key=lambda i: i[1])]

    index_label = feature_image[-1]


    f_temp = []  # np.array([]).reshape(0, 4)
    for featurei in feature_list:
        f_temp.append(feature_image[featurei])
    feature_matrix = np.array(f_temp).T

    yclist = bestgmm_A.predict(feature_matrix)
    tmp = segmentedMitoImage.rsplit('/',1)

    tempprop1, tempprop2, bmimage_cluster1, bmimage_cluster2 = imagetoprop(label_image, a_sorted, index_label, yclist)

    fileCluster = pathCluster + dfWell4training.iloc[i]['DataSetID'] + '_' + dfWell4training.iloc[i]['wellID2'] + '.cp'
    print(fileCluster)
    with open(fileCluster, 'wb') as fp:
        cPickle.dump(['bmimage_cluster1', 'bmimage_cluster2'], fp)
        cPickle.dump(bmimage_cluster1, fp)
        cPickle.dump(bmimage_cluster2, fp)

    nFragmentedMito = np.sum(1-yclist)
    nFusedMito = np.sum(yclist)
    nTMito = nFragmentedMito + nFusedMito

    aFragmentedMito =  [mitoFragmented.area for mitoFragmented in tempprop1]
    aFusedMito =  [mitoFused.area for mitoFused in tempprop2]
    aSumFragmentedMito = np.sum(aFragmentedMito) * resolutionpx**2
    aSumFusedMito = np.sum(aFusedMito) * resolutionpx**2


    nFragmentedMitoA.append(nFragmentedMito)
    nFusedMitoA.append(nFusedMito)
    nTMitoA.append(nTMito)
    aSumFragmentedMitoA.append(aSumFragmentedMito)
    aSumFusedMitoA.append(aSumFusedMito)


    if 0:
        bmimage = bmimage_cluster1 + 2 * bmimage_cluster2

        fig, axes = plt.subplots(2, 3, figsize= (18,12))#,constrained_layout=True)
        # pdb.set_trace()
        imGrayMitoimage = misc.imread(grayMitoImageTmp)
        # grayNucleusImage = dfWell4training.iloc[i]['grayNucleusImages']
        grayNucleusImage = dfWell4training.iloc[i]['grayMitoImages'].replace('c2', 'c1')

        imGrayNucleusimage = misc.imread(grayNucleusImage)

        plt.subplot(2,3,1)
        plt.imshow(imGrayNucleusimage,cmap='Greys')

        plt.subplot(2,3,2)
        plt.imshow(imGrayMitoimage,cmap='Greys')
        plt.subplot(2,3,3)
        plt.imshow(bw_image,cmap='Greys')
        plt.subplot(2,3,4)
        plt.imshow(bmimage_cluster1,cmap=newRed)
        plt.subplot(2,3,5)
        plt.imshow(bmimage_cluster2,cmap=newBlue)

        plt.subplot(2,3,6)
        plt.imshow(bmimage,cmap=newcmp2)

        plt.pause(2.7)
        plt.close(fig)

# plt.show()

dfWell4training['nFragmentedMito'] = nFragmentedMitoA
dfWell4training['nFusedMito'] = nFusedMitoA
dfWell4training['nTMito'] = nTMitoA
dfWell4training['aSumFragmentedMito'] = aSumFragmentedMitoA
dfWell4training['aSumFusedMito'] = aSumFusedMitoA

dfWell4training.to_csv('./trainingDataFrameResult.csv')
pdb.set_trace()

dfnew = dfWell4training[['wellID2', 'DataSetID',
                         'treatment', 'con_uM', 'tp',
                         'nFragmentedMito', 'nFusedMito','nTMito',
                         'aSumFragmentedMito', 'aSumFusedMito'
                         ]]

treatmentL = pd.unique(dfnew['treatment'])
nTreatment = len(treatmentL)
plt.figure(figsize = (24,16))

for i, treatmenti in enumulate(treatmentL):
    tpL = pd.unique(dfnew[dfnew['treatment']==treatmenti]['tp'])

    concL = pd.unique(dfnew[dfnew['treatment']==treatmenti]['conc_uM'])

    plt.subplot(5, 6, i+1)
    plt.title(treatmenti)
    nFragmentedMitoTreatment = []
    nFusedMitoTreatment = []

    for conci in sorted(concL):
        dfTmp = dfnew[(dfnew['treatment'] == treatmenti) &
                      (dfnew['con_uM'] == conci)]
        nFragmentedMitoTreatment = nFragmentedMitoTreatment.append(dfTmp['nFragmentedMito'])
        nFusedMitoTreatment = nFusedMitoTreatment.append(dfTmp['nFusedMito'])
    plt.plot(np.array(sorted(concL)), np.array(nFragmentedMitoTreatment), linestyle = 'none')
    plt.plot(np.array(sorted(concL)), np.array(nFusedMitoTreatment), linestyle = 'none')
plt.show()
