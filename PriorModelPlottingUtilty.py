import ezdxf
import itertools
import k3d
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import meshio
import numpy as np
import pandas as pd
import pynoddy
from scipy.interpolate import griddata
import scipy.interpolate as interp
import sys, os
import subprocess
from shutil import copyfile
import vtk
from vtkplotter import *
import pynoddy.history
import pynoddy.events
import time
from scipy.interpolate import interpn

def get_wellbore_voxels_from_paths2(LithBlock, xi,yi,zi, xlim, ylim, zlim, delV):

    x = np.linspace(xlim[0], xlim[1], np.shape(LithBlock)[0])
    y = np.linspace(ylim[0], ylim[1], np.shape(LithBlock)[1])
    z = np.linspace(zlim[0], zlim[1], np.shape(LithBlock)[2])

    Vi = interpn((x,y,z), LithBlock, np.array([xi,yi,zi]).T, method='nearest')

    return Vi
    
def importModulesAndSetup():
    import pynoddy
    
    ##############
    # Some file fixes
    ##############
    basepynoddyfile = pynoddy.__file__[:-11]+'experiment/__init__.py'
    # Read in the file
    with open(basepynoddyfile, 'r') as file :
      filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('from . import util.sampling as Sample', 'from .util import sampling as Sample')

    # Write the file out again
    with open(basepynoddyfile, 'w') as file:
      file.write(filedata)

    target = pynoddy.__file__[:-11]+'output.py'

    source = 'output.py'

    copyfile(source, target)

    ##############
    # Changing exection permissions
    ##############
    folder = os.getcwd()
    noddyEXE = folder+'/noddy.exe'
    strV = 'chmod 777 '+noddyEXE
    os.system(strV)
    import pynoddy.experiment
    
def CalculateModel(modelfile, output_name, outputoption = 'ALL', cubesize = 250):
    folder = os.getcwd()
    noddyEXE = folder+'/noddy.exe'
    H1 = pynoddy.history.NoddyHistory(modelfile)
    H1.change_cube_size(cubesize)
    H1.write_history(modelfile)
    print(subprocess.Popen([noddyEXE, modelfile, output_name, outputoption], 
                           shell=False, stderr=subprocess.PIPE, 
                           stdout=subprocess.PIPE).stdout.read())
    print('Finished calculating model')
    
def getDXF_parsed_structure(output_name):
    filename = output_name + '.dxf'
#    doc = ezdxf.readfile(filename)
    cell_data = []
    xpoint = []
    ypoint = []
    zpoint = []
    with open(filename) as f:
        cntr=0
        faceCounter=0
        for line in f:
            if(cntr==(7+faceCounter*28)):
                cell_data.append(line)
                faceCounter=faceCounter+1
            elif(cntr==(9+(faceCounter-1)*28)):
                xpoint.append(float(line))
            elif(cntr==(11+(faceCounter-1)*28)):
                ypoint.append(float(line))
            elif(cntr==(13+(faceCounter-1)*28)):
                zpoint.append(float(line))

            elif(cntr==(15+(faceCounter-1)*28)):
                xpoint.append(float(line))
            elif(cntr==(17+(faceCounter-1)*28)):
                ypoint.append(float(line))
            elif(cntr==(19+(faceCounter-1)*28)):
                zpoint.append(float(line))

            elif(cntr==(21+(faceCounter-1)*28)):
                xpoint.append(float(line))
            elif(cntr==(23+(faceCounter-1)*28)):
                ypoint.append(float(line))
            elif(cntr==(25+(faceCounter-1)*28)):
                zpoint.append(float(line))

            cntr=cntr+1

    points = np.column_stack((np.asarray(xpoint, dtype=float),
                             np.asarray(ypoint, dtype=float),
                             np.asarray(zpoint, dtype=float)))
    cell_data.pop()
    cell_data = np.asarray(cell_data, dtype=object)
#    print('Finished reading model')
   
    return points, cell_data, faceCounter

def convertSurfaces2VTK(points, cell_data, faceCounter, outputOption = 1, fileprefix='Surface',  xy_origin=[0,0,0]):
    
    # Choose output option
    num3Dfaces=faceCounter
    print('The number of triangle elements (cells/faces) is: ' + str(num3Dfaces))


    #apply origin transformation
    points[:, 0] = points[:, 0]+xy_origin[0]
    points[:, 1] = points[:, 1]+xy_origin[1]
    points[:, 2] = points[:, 2]+xy_origin[2]
    
    cell_data = pd.Series(cell_data.reshape((-1, )))

    CatCodes = np.zeros((len(cell_data),))
    filterB = (cell_data.str.contains('B')) 
    filterS = (cell_data.str.contains('S')) 

    CatCodes[filterB]= cell_data.loc[filterB].str[:-20].astype('category').cat.codes
    CatCodes[filterS]= -1*(cell_data.loc[filterS].str[:-12].astype('category').cat.codes+1)

    for i in range(1, len(CatCodes)):
        if(CatCodes[i]==0):
            CatCodes[i]=CatCodes[i-1]
            if(CatCodes[i-1]==0):
                CatCodes[i]=CatCodes[np.nonzero(CatCodes)[0][0]]

#    CatCodes[filterB]= 0
#    CatCodes[filterS]= 1

    UniqueCodes = np.unique(CatCodes)
    nSurfaces = len(UniqueCodes)

    if (outputOption==2): ## if you would like a single vtk file
        UniqueCodes = np.unique(CatCodes)
        nCodes = len(UniqueCodes)
        cells = np.zeros((num3Dfaces, 3),dtype ='int')
        i=0
        for f in range(num3Dfaces):
            cells[f,:]= [i, i+1, i+2]
            i=i+3
        meshio.write_points_cells(
            "Model.vtk",
            points,
            cells={'triangle':cells},
            cell_data= {'triangle': {'cat':CatCodes}}   
            )
    else: ## option 1: make a separate file for each surface
        for i in range(nSurfaces):
            filterPoints = CatCodes==UniqueCodes[i]
            nCells = np.sum(filterPoints)
            Cells_i = np.zeros((nCells, 3),dtype ='int')
            cntr = 0
            for j in range(nCells):
                Cells_i[j]=[cntr, cntr+1, cntr+2]
                cntr=cntr+3

            booleanFilter = np.repeat(filterPoints,3)
   
            meshio.write_points_cells(
                fileprefix+str(i)+".vtk",
                points[np.repeat(filterPoints,3), :],
                cells={'triangle':Cells_i}
                )
#            print('surface ' +str(i) + '  code:'+str(UniqueCodes[i]))

#    print('Finished converting dxf to vtk')
    
    
    return nSurfaces, points, CatCodes

def CalculatePlotStructure(H1, output_name, plot, includeGravityCalc=0, cubesize = 250,  xy_origin=[317883,4379646, 1200-4000]):
    
    newmodelfile ='temp_model'
    H1.write_history(newmodelfile)

    #Alter the mesh size if desiring to speed up the process. Recommended size is 100
#    output_name = 'temp_noddy_out'
    
    
    #output options
    #1. BLOCK       
    #GEOPHYSICS   
    #SURFACES
    #BLOCK_GEOPHYS
    #BLOCK_SURFACES
    #TOPOLOGY
    #ANOM_FROM_BLOCK
    #ALL 
    if(includeGravityCalc==0):
        outputoption = 'BLOCK_SURFACES'
    else:
        outputoption = 'ALL'

    start = time.time()
    CalculateModel(newmodelfile, output_name, outputoption, cubesize)
    end = time.time()
    print('Calculation time took '+str(end - start) + ' seconds')

    ## Now need to change the DXF file (mesh format) to VTK. This is slow unfortunately
    start = time.time()
    points, cell_data, faceCounter = getDXF_parsed_structure(output_name)
    end = time.time()
    print('Parsing time took '+str(end - start) + ' seconds')

    ## Make a vtk file for each surface (option 1) or make a single vtk file for all surfaces (option 2)
    outputOption = 1
    fileprefix = 'TempSurface'

    start = time.time()
    nSurfaces, points, CatCodes = convertSurfaces2VTK(points, cell_data, faceCounter, outputOption, fileprefix,  xy_origin=xy_origin)
    
    end = time.time()
    print('Convert 2 VTK time took '+str(end - start) + ' seconds')

    N1 = pynoddy.output.NoddyOutput(output_name)

    lithology = N1.block

    [maxX, maxY, maxZ] = np.max(points, axis=0)
    [minX, minY, minZ] = np.min(points, axis=0)
    minZ = xy_origin[2]
    x = np.linspace(minX, maxX, N1.nx, dtype=np.float32)
    y = np.linspace(minY, maxY, N1.ny, dtype=np.float32)
    z = np.linspace(xy_origin[2], maxZ, N1.nz, dtype=np.float32)
#    z = np.linspace(0, 4000, N1.nz, dtype=np.float32)

    delx = x[1]-x[0]
    dely = y[1]-y[0]
    delz = z[1]-z[0]
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    CoordXYZ = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1),zz.reshape(-1,1)), axis=1)

    Lithology = griddata(CoordXYZ, np.transpose(lithology, axes =(2, 1, 0)).reshape(-1,), (xx, yy, zz), method='nearest')
    
    vol = Volume(Lithology, c='jet', spacing=[delx, dely,delz], origin =[xy_origin[0], xy_origin[1], xy_origin[2]])
    lego = vol.legosurface(-1, 5).opacity(0.15).c('jet')
#    vol = vol.color(['red', 'violet', 'green'])
    plot += lego
#    plot += vol

    colors = pl.cm.jet(np.linspace(0,1,nSurfaces))

    #outputOption = 1

    if(outputOption==1):
        for i in range(nSurfaces):
            filename = 'TempSurface'+str(i)+'.vtk'
            e=load(filename).c(colors[i, 0:3])

            plot += e
    else:
        filename = 'TempModel.vtk'
        e=load(filename)
        plot += e
        
    Data = pd.read_csv('WellNamesPaths.csv')
    
    filterV = (Data['X_m']>minX+0.5*delx) & (Data['Y_m']>minY+0.5*dely) &  (Data['Z_m']>minZ+0.5*delz) & (Data['X_m']<maxX-0.5*delx) & (Data['Y_m']<maxY-0.5*dely) & (Data['Z_m']<maxZ-0.5*delz)
    
    Data = Data[filterV]
    WellboreColors = get_wellbore_voxels_from_paths2(lithology, Data['X_m'], Data['Y_m'], Data['Z_m'], [minX, maxX], [minY, maxY], [minZ, maxZ], [delx, dely, delz])

    
    cm = plt.get_cmap('jet')
    maxL_V = np.max(WellboreColors)
    minL_V = np.min(WellboreColors)
    normL = (WellboreColors-minL_V)/(maxL_V-minL_V)
    RGBA = cm(normL)
    
    pointsPlot = Points(Data[['X_m', 'Y_m', 'Z_m']].values, r=4, c=RGBA)

    # color vertices based on their scalar value with any matplotlib color map
#    points.pointColors(WellboreColors, cmap='coolwarm')
    
    plot += pointsPlot

    return points

def sampleSinglePriorModelParameters(ModelParametersTable, baseInputFile, OutputfileName):
    H1 = pynoddy.history.NoddyHistory(baseInputFile)
    
    Events = ModelParametersTable.columns.tolist()
    Events.remove('Parameter')
    Events.remove('Property')
    nEvents = len(Events)
    
    Properties= pd.unique(ModelParametersTable['Property'].values).tolist()
    nProperties = len(Properties)
    
    for e in range(nEvents):
        for p in range(nProperties):
            event = Events[e]
            prop = Properties[p]
            priorDistType = ModelParametersTable.loc[p*3, event]
            priorparam1 = float(ModelParametersTable.loc[p*3+1, event])
            priorparam2 = float(ModelParametersTable.loc[p*3+2, event])
            # Take care of uniform
            if(priorDistType=='Uniform'):
                paramVal = np.random.uniform(priorparam1,priorparam2,1)
                H1.events[e+2].properties[prop] = paramVal[0]
            
    H1.write_history(OutputfileName)
    return H1

def sampleMultiplePriorModelParameters(ModelParametersTable, baseInputFile, nrealizations, folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    H1 = pynoddy.history.NoddyHistory(baseInputFile)
    
    for r in range(nrealizations):
        OutputfileName = folder+'/realization_'+str(r)
        sampleSinglePriorModelParameters(ModelParametersTable, baseInputFile, OutputfileName)

def calcMultipleHistoryFiles(nrealizations, folder, cubesize = 250, outputoption = 'BLOCK_SURFACES', xy_origin=[317883,4379646, 1200-4000]):

    for r in range(nrealizations):
        inputfile = folder+'/realization_'+str(r)
        output_name = inputfile+'_output'
        CalculateModel(inputfile, output_name, outputoption, cubesize)

        ## Now need to change the DXF file (mesh format) to VTK. This is slow unfortunately
        points, cell_data, faceCounter = getDXF_parsed_structure(output_name)

        ## Make a vtk file for each surface (option 1) or make a single vtk file for all surfaces (option 2)
        outputOption = 1
        nSurfaces, points, CatCodes = convertSurfaces2VTK(points, cell_data, faceCounter, outputOption, fileprefix=inputfile+'_Surface', xy_origin=xy_origin)
        data = np.concatenate((points,np.repeat(CatCodes,3).reshape(-1,1)), axis=1)
        np.savetxt(inputfile+'pointdata.csv', data, delimiter=',')        
