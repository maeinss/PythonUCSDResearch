# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 23:09:47 2023

@author: roues
"""
import numpy as np
from scipy import ndimage
from Operations import Operations
import Plot_operations
import torchio as tio
import SimpleITK as sitk
#%%

class transformationX:
    
    def __init__(self, matrix):
        """
        Initialize the Sectioning class.

        Parameters
        ----------
        matrix : ndarray
            3D matrix.
        """
        self.matrix = matrix
        
    def nonZeroMatrixPad(self, dose_Matrix, pad = 3):
        """
        Find the minimum non-zero matrix element and pad the matrix in the z-direction.
    
        Parameters:
        ----------
        dose_Matrix : np.ndarray
            3D matrix representing dose values.
        pad : int, optional
            Number of padding elements to add in the z-direction. Default is 3.
    
        Returns:
        -------
        tuple
            A listcontaining two matrices:
            - minimum_matrix: The minimum non-zero matrix element padded in the z-direction.
            - minimum_matrix_dose: The corresponding dose matrix with the same padding as minimum_matrix.
        """
        nonzero_indices = np.nonzero(self.matrix)
        min_indices = [np.min(index) - pad for index in nonzero_indices]
        max_indices = [np.max(index) + 1 + pad for index in nonzero_indices]
        minimum_matrix = self.matrix[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]]
        minimum_matrix_dose = dose_Matrix[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]]
        return [minimum_matrix, minimum_matrix_dose]

    def cart2pol1(self, x, y, x0, y0):
        """
        Convert Cartesian coordinates to polar coordinates.

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.
        x0 : float
            x-coordinate of the center.
        y0 : float
            y-coordinate of the center.

        Returns
        -------
        float
            Angle in degrees.
        """
        if x == 0 and y == 0:
            x = 0.00001
            y = 0.00001
        else:
            x = x
            y = y
        x = x0 - x
        y = y0 - y
        phi = np.arctan2(y, x)
        phi = np.degrees(phi)
        if phi < 0:
            phi = phi + 360
        return phi

    def toPolarMatrix(self):
        """
        Convert the matrix to polar coordinates.

        Returns
        -------
        list
            List containing the polar matrix, original matrix, and center coordinates.
        """
        matrix = np.array(self.matrix)
        polar = np.empty(matrix.shape)
        valuesAbsCen = [i for i in ndimage.measurements.center_of_mass(np.array(matrix))]
        for idx, val in enumerate(valuesAbsCen):
            if np.isnan(val):
                valuesAbsCen[idx] = (matrix.shape[idx] - 1) / 2
        center = [round(i) for i in valuesAbsCen]

        for slices in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                for y in range(matrix.shape[2]):
                    polar[slices, x, y] = self.cart2pol1(x, y, center[1], center[2])
        
        return [polar, matrix, center]
    
#%%torchio transformations
class Torchio:
    "takes a 3D matrix whether in the form of an image or simply a numpy array and performs operations"
    def __init__(self, matrix):
        """
        Initialize the Sectioning class.

        Parameters
        ----------
        matrix : ndarray
            3D matrix.
        """
        self.matrix = matrix 
    
    def resampleX(self, dose_array, transformedSpacing = (1,1,1), setspacing = (1,1,1), image_interpolation = "nearest"):
        """Resamples a 3D dose array to a new voxel size array.

        Parameters
        ----------
        dose_array : np.ndarray
            3D dose array.
        transformedSpacing : tuple of floats, optional
            The new voxel size of the output. Default is (0.5, 0.5, 0.5).
        setSpacing : tuple of floats, optional
            The voxel size in mm to set in the input array. Default is (1, 1, 1).
        image_interpolation: type of interpolation used. Default is "nearest" for nearest neighbors
    
        Returns
        -------
        a set of 3D str array and dose array
        """
        transformation = tio.Resample(transformedSpacing, image_interpolation= image_interpolation)
        img_str_array = sitk.GetImageFromArray(self.matrix)
        dose_str_array = sitk.GetImageFromArray(dose_array)
        img_str_array.SetSpacing((setspacing))
        dose_str_array.SetSpacing((setspacing))

        img_str_array = transformation(img_str_array)
        dose_str_array = transformation(dose_str_array)
        
        newImgArray = sitk.GetArrayFromImage(img_str_array)
        newDoseArray = sitk.GetArrayFromImage(dose_str_array)
        
        return [[newImgArray, newDoseArray],[img_str_array,dose_str_array]]
        
    

#%%
mat = np.random.randint(0,5, (10,10,10))
# init3 = transformationX(mat)
# val = init3.toPolarMatrix() 

# Operations().selectElements(mat, 1)
# #%%
# trialCube = np.zeros((60, 60, 60))
# cube_size = 30 
# x_start = (trialCube.shape[0] - cube_size) // 2
# y_start = (trialCube.shape[1] - cube_size) // 2
# z_start = (trialCube.shape[2] - cube_size) // 2

# trialCube[x_start:x_start+cube_size, y_start:y_start+cube_size, z_start:z_start+cube_size] = 1
# Plot_operations.PltX([trialCube, trialCube]).plot_3d_scatter()

#%%














