# -*- coding: utf-8 -*-
"""
This module contains operations for such as indexing and data manipulating across one or many subsets
Created on Fri Apr 14 11:05:40 2023

@author: Rupesh Ghimire
"""
import numpy as np
from scipy.interpolate import interp1d
import seg_metrics.seg_metrics as sg
from tabulate import tabulate
#%%

class Operations:
    """
    This class contains functions for modifying and arranging data.
    """

    def __init__(self, matrix = np.random.randint(0,5, (10,10,10)), dose = np.random.randint(0,5, (10,10,10))):
        """
        Initialize the Modification class.

        Parameters
        ----------
        matrix : Matrix
            Random matrix.
        """
        self.matrix = matrix
        self.dose = dose

    def chop(self, delta=0.999999, fillvalue=0):
        """
        Takes a matrix of any dimension as input and chops the values to a certain threshold.

        Parameters
        ----------
        delta : float, optional
            The threshold below which elements will be chopped. The default is 0.999999.

        fillvalue : int, optional
            The value to replace the chopped elements with. The default is 0.

        Returns
        -------
        numpy.ndarray or list
            The rounded-off matrix or a list, same as the input.
        """
        modified_matrix = np.ma.masked_inside(self.matrix, -delta, delta).filled(fillvalue)
        return modified_matrix

    def intersection(lists):
        """
        Checks two lists and finds the intersecting elements.

        Parameters
        ----------
        matrix2D : a list of two lists.

        Returns
        -------
        list
            List of intersecting elements.
        """
        if len(lists) == 2:
            set1 = set(map(tuple, lists[0]))
            set2 = set(map(tuple, lists[1]))
            return list(set1 & set2)
        else:
            return "The matrix is not 2D."

    def intersecting_coords(self, indicesx):
        """
        Compare indices obtained from the 3D section to the structure coordinates and map them to the original structure.

        Parameters
        ----------
        indicesx : list
            Lists of indices obtained from the 3D matrix using np.where. [np.where(..), np.where(..)]

        Returns
        -------
        list
            List of arrays containing intersecting coordinates for each set of indices.
        """
        final_array = []
        oneIndices = np.where(self.matrix != 0)
        indicesxT = [np.array(x).T for x in indicesx]
        oneIndicesT=np.array((oneIndices)).T
        
        for idx,val in enumerate(indicesxT):
            intList=Operations.intersection([oneIndicesT,val])
            intList = np.array([list(x) for x in intList]).T
            final_array.append(intList)   
        final_array=[[np.array(j) for j in val] for val in final_array]
        return final_array

    def selectElements(self, myLists,i):
        """
        Select the element at index `i` from each sublist in the given list.
    
        Parameters:
        ----------
        lists : list
            List of sublists.
        i : int
            Index of the element to select from each sublist.
    
        Returns:
        -------
        np.ndarray
            Array containing the selected elements from each sublist. If a sublist has length less than or equal to 1, None is included instead.
    
        Example:
        --------
        Given the input lists = [[1, 2, 3], [4, 5], [6, 7, 8, 9]], and i = 1,
        the function will return np.array([2, 5, 7]).
        If a sublist has length less than or equal to 1, None is included instead, so given the input lists = [[1], [2, 3], [4]], and i = 0,
        the function will return np.array([1, None, 4]).
        """
        result = []
        for sublist in myLists:
            if len(sublist) > 1:
                result.append(sublist[i])
            else:
                result.append(None)
        return np.array(result)
    
    def normalize_list(self,my_list):
        min_val = np.min(my_list)
        max_val = np.max(my_list)
        normalized_list = (my_list - min_val) / (max_val - min_val)
        return normalized_list
    
    def nonZeroMatrix(self, dose_Matrix):
        """
        Extracts the minimum bounding box of non-zero values from a binary array
        and retrieves the corresponding region from a dose matrix.
    
        Args:
            binary_array (ndarray): A 3-dimensional binary array.
            dose_Matrix (ndarray): A 3-dimensional matrix representing doses.
    
        Returns:
            list: A list containing two elements:
                  - minimum_matrix: A submatrix containing the minimum bounding box
                                    of non-zero values from the binary array.
                  - minimum_matrix_dose: A submatrix containing the corresponding
                                         region from the dose matrix.
        """
        nonzero_indices = np.nonzero(self.matrix)
        min_indices = [np.min(index) for index in nonzero_indices]
        max_indices = [np.max(index) + 1 for index in nonzero_indices]
        minimum_matrix = self.matrix[min_indices[0]:max_indices[0],
                                      min_indices[1]:max_indices[1],
                                      min_indices[2]:max_indices[2]]
        minimum_matrix_dose = dose_Matrix[min_indices[0]:max_indices[0],
                                          min_indices[1]:max_indices[1],
                                          min_indices[2]:max_indices[2]]
        return [minimum_matrix, minimum_matrix_dose]

    def perform_operation(self,matrix, constant, operation):
        """
        Performs a given operation on each element of a matrix with a constant number.
    
        Args:
            matrix (ndarray): The input matrix.
            constant (float): The constant value to use in the operation.
            operation (str): The operation to perform on the elements.
                             Possible values: 'divide', 'multiply', 'add', 'subtract'.
    
        Returns:
            ndarray: The resulting matrix after the specified operation.
    
        """
        # Create an empty matrix to store the results
        result = np.empty_like(matrix)
    
        # Iterate through each element of the matrix
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if len(matrix.shape) == 3:
                    for k in range(matrix.shape[2]):
                        # Perform the specified operation on the element
                        if operation == 'divide':
                            result[i, j, k] = matrix[i, j, k] / constant
                        elif operation == 'multiply':
                            result[i, j, k] = matrix[i, j, k] * constant
                        elif operation == 'add':
                            result[i, j, k] = matrix[i, j, k] + constant
                        elif operation == 'subtract':
                            result[i, j, k] = matrix[i, j, k] - constant
                else:
                    # Perform the specified operation on the element
                    if operation == 'divide':
                        result[i, j] = matrix[i, j] / constant
                    elif operation == 'multiply':
                        result[i, j] = matrix[i, j] * constant
                    elif operation == 'add':
                        result[i, j] = matrix[i, j] + constant
                    elif operation == 'subtract':
                        result[i, j] = matrix[i, j] - constant
    
        return result
    
    def addMatrices(self, normalize = False, additionalFxs = 5):
    
        """Adds a list of matrices together.
        
        Parameters
        ----------
        matrix_list : list of np.ndarray
            A list of matrices to be added together.
        
        Returns
        -------
        np.ndarray
            The sum of the matrices in the list.
        """
        
        result = np.zeros_like(self.matrix[0])
        if normalize == True:
            for matrix in self.matrix:
                result += self.perform_operation(matrix,additionalFxs, "divide")
        else:
            for matrix in self.matrix:
                result += matrix
        return result
    
    def intpl(self,val):
        """
        Interpolates the dose value based on a given input value.
    
        Parameters:
            val (float): The input value for which the dose value needs to be interpolated.
    
        Returns:
            float: The interpolated dose value corresponding to the input value.
    
        """
        intfun = interp1d(self.matrix, self.dose ,bounds_error=False)
        return intfun(val)
    
    def constantArray(self,string, n):
      """Repeats a string n times.

      Args:
        string: The string to repeat.
        n: The number of times to repeat the string.

      Returns:
        A list of strings, where each string is the original string repeated n times.
      """
      repeated_strings = []
      for i in range(n):
        repeated_strings.append(string)

      return repeated_strings
  
    def dScore(self , labels = range(1,16) ,prnt =False):
        """Gives the dicescore and other scores like hausdroff ..of two matrices.

        Parameters
        ----------
        labels : int, optional
            The list of integers which are the labels in two matrices, default is range(1,16)
    
        print : Boolean, optional
            whether if you want to print it or not default is False
    
        Returns
        -------
        A list of 4 values [dice, hd, hd95, msd] scores
        """
        
        metrics = sg.write_metrics(labels=labels,  # exclude background if needed
                          gdth_img=self.matrix,
                          pred_img=self.dose,
                          csv_file=None,
                          metrics=["dice",'hd', 'hd95', 'msd'])
        key,values=np.array([list(i.keys()) for i in metrics ]),np.array([list(i.values()) for i in metrics ])
        if prnt ==True:
            print("")
            print(tabulate(values[0].T, headers=key[0],tablefmt='orgtbl'))
        return values[0].T

#%%
# val2=Operations([0.92,3,4,4,5,4,2])
# val2.chop()
# #%%
# mat = np.random.randint(0,5, (10,10,10))
# idx1 = np.where(mat==2)
# idx2 = np.where(mat == 3)
# init1 = Operations(mat)
# init1.intersecting_coords([idx1,idx2])

# val1 = [np.array(idx1).T, np.array(idx2).T]
#%%






