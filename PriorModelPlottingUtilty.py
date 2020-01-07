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
    print('cube size is ' +str(H1.get_cube_size()))
    print(subprocess.Popen([noddyEXE, modelfile, output_name, outputoption], 
                           shell=False, stderr=subprocess.PIPE, 
                           stdout=subprocess.PIPE).stdout.read())
    print('Finished calculating model')
    
def getDXF_parsed_structure(output_name):
    filename = output_name + '.dxf'
    doc = ezdxf.readfile(filename)
    print('Finished reading model')
    return doc

def convertSurfaces2VTK(dxf_parsed, outputOption = 1, fileprefix='Surface'):
    # Choose output option
    msp = dxf_parsed.modelspace()
    faces = msp.query('3DFACE')
    num3Dfaces=len(faces)
    points = np.zeros((num3Dfaces*3, 3))

    cells = np.zeros((num3Dfaces, 3),dtype ='int')
    cell_data=[]

    i=0
    faceCounter=0
    name = []
    for e in faces:
        points[i, 0] = e.dxf.vtx0[0]
        points[i, 1] = e.dxf.vtx0[1]
        points[i, 2] = e.dxf.vtx0[2]

        points[i+1, 0] = e.dxf.vtx1[0]
        points[i+1, 1] = e.dxf.vtx1[1]
        points[i+1, 2] = e.dxf.vtx1[2]

        points[i+2, 0] = e.dxf.vtx2[0]
        points[i+2, 1] = e.dxf.vtx2[1]
        points[i+2, 2] = e.dxf.vtx2[2]

        cells[faceCounter,:]= [i, i+1, i+2]
        cell_data.append(e.dxf.layer)
        i=i+3
        faceCounter=faceCounter+1

    print('The number of triangle elements (cells/faces) is: ' + str(i))

    cell_data = pd.Series(cell_data)

    CatCodes = np.zeros((len(cell_data),))
    filterB = (cell_data.str.contains('B')) 
    filterS = (cell_data.str.contains('S')) 

    CatCodes[filterB]= cell_data.loc[filterB].str[:-16].astype('category').cat.codes
    CatCodes[filterS]= -1*cell_data.loc[filterS].str[:-12].astype('category').cat.codes

    if (outputOption==2): ## if you would like a single vtk file
        UniqueCodes = np.unique(CatCodes)
        nCodes = len(UniqueCodes)

        meshio.write_points_cells(
            "Model.vtk",
            points,
            cells={'triangle':cells},
            cell_data= {'triangle': {'cat':CatCodes}}   
            )
    else: ## option 1: make a separate file for each surface
        UniqueCodes = np.unique(CatCodes)
        nSurfaces = len(UniqueCodes)

        for i in range(nSurfaces):
            filterPoints = CatCodes==UniqueCodes[i]
            nCells = np.sum(filterPoints)
            Cells_i = np.zeros((nCells, 3),dtype ='int')
            cntr = 0
            for j in range(nCells):
                Cells_i[j]=[cntr, cntr+1, cntr+2]
                cntr=cntr+3

            meshio.write_points_cells(
                fileprefix+str(i)+".vtk",
                points[np.repeat(filterPoints,3), :],
                cells={'triangle':Cells_i}
                )

    print('Finished converting dxf to vtk')
    
    return nSurfaces, points

def CalculatePlotStructure(H1, plot, includeGravityCalc=0):
    
    newmodelfile ='temp_model'
    H1.write_history(newmodelfile)

    #Alter the mesh size if desiring to speed up the process. Recommended size is 100
    output_name = 'temp_noddy_out'
    cubesize = 250
    
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
        
    CalculateModel(newmodelfile, output_name, outputoption, cubesize)

    ## Now need to change the DXF file (mesh format) to VTK. This is slow unfortunately
    dxf_parsed = getDXF_parsed_structure(output_name)

    ## Make a vtk file for each surface (option 1) or make a single vtk file for all surfaces (option 2)
    outputOption = 1
    fileprefix = 'TempSurface'
    nSurfaces, points = convertSurfaces2VTK(dxf_parsed, outputOption, fileprefix)
    
    N1 = pynoddy.output.NoddyOutput(output_name)

    lithology = N1.block

    [maxX, maxY, maxZ] = np.max(points, axis=0)

    x = np.linspace(0, maxX, N1.nx, dtype=np.float32)
    y = np.linspace(0, maxY, N1.ny, dtype=np.float32)
    z = np.linspace(0, maxZ, N1.nz, dtype=np.float32)

    delx = x[1]-x[0]
    dely = y[1]-y[0]
    delz = z[1]-z[0]
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    CoordXYZ = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1),zz.reshape(-1,1)), axis=1)

    Lithology = griddata(CoordXYZ, np.transpose(lithology, axes =(2, 1, 0)).reshape(-1,), (xx, yy, zz), method='nearest')
    
    embedWindow('k3d') #you can also choose to change to itkwidgets

    vol = Volume(Lithology, c='jet', spacing=[delx, dely,delz])

    lego = vol.legosurface(-1, 4).opacity(0.15).c('jet')

    plot += lego

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