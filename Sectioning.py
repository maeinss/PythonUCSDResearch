# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 22:25:16 2023

@author: roues
"""
import numpy as np
from Operations import Operations
from Transformation import transformationX
import matplotlib.pyplot as plt
#%%
class Sectioning:
    """
    This class deals with horizonatal and vertical sectioning functions. Takes a 3D structure matrix as a constructor varible as self
    
    """
    def __init__(self, matrix):
        """
        Initialize the Find class.

        Parameters
        ----------
        matrix : a three-dimensional (3D) matrix
        """
        self.matrix = matrix
        
    def divide_matrix(self, dose_array, numSections=5, numSectors=3):
        """
        Divide the self matrix into multiple sections based on the number of non-zero elements.
    
        Parameters:
        ----------
        dose_array : np.ndarray
            3D matrix representing dose values.
        num : int, optional
            Number of sections to divide the matrix into. Default is 5.
        numSectors : int, optional
            Number of sectors to modify within each section. Default is 3.
            numSectors isn't generally used unless the function faces singularity. In this case we put some voxes with non-zero values .
    
        Returns:
        -------
        list
            A list containing three elements:
            - fiveMats: List of divided matrix sections.
            - dose_array_sections: List of divided dose arrays corresponding to each section.
            - arrangedIndex: List of indices indicating the arrangement of sections in the original matrix.
        """
        matrix = self.matrix[::-1, :, :]
        total_ones = np.count_nonzero(matrix)
        target_ones_per_section = total_ones // numSections
    
        sections = []
        sliceidxs = []
        current_count = 0
        running_count = 0
        current_section = []
        current_sliceidxSec = []
    
        for i in range(matrix.shape[0]):
            slice_no = matrix.shape[0] - i - 1
            current_slice = matrix[i]
            current_section.append(current_slice)
            current_sliceidxSec.append(slice_no)
            current_count = np.count_nonzero(current_slice)
    
            if (current_count + running_count) > target_ones_per_section:
                sections.append(current_section)
                sliceidxs.append(current_sliceidxSec)
                current_section = []
                current_count = 0
                running_count = 0
                current_sliceidxSec = []
            else:
                running_count = current_count + running_count
                continue
    
        sections.append(current_section)
        sliceidxs.append(current_sliceidxSec)
        sections[::-1]
        arrangedIndex = [x[::-1] for x in sliceidxs[::-1]]
        arrangedIndex = [x for x in arrangedIndex if x != []]
    
        if len(arrangedIndex) < numSections:
            val1 = [arrangedIndex[0][0]]
            del arrangedIndex[0][0]
            arrangedIndex.insert(0, val1)
    
        lastIdx = [1 + item[-1] for item in arrangedIndex][:-1]
        fiveMats = np.split(matrix[::-1, :, :], lastIdx, axis=0)
    
        for idx, val in enumerate(fiveMats):
            if np.count_nonzero(val) == 0:
                shapek = np.array(val).shape
                for i in range(numSectors):
                    fiveMats[idx][-1][round((shapek[1] - 1) / 2) + i][round((shapek[2] - 1) / 2) + i] = 1
    
        return [fiveMats, np.split(dose_array, lastIdx, axis=0), arrangedIndex]
#%
    def threeSections(self, matrix1, dose_array, numSectors=3, numSections = 5, show=False, labelsChange=False, secIdx=0):
        """
        Divide a 3D matrix into multiple sections based on polar angle.
    
        Parameters:
        ----------
        matrix1 : numpy.ndarray
            3D matrix to be divided into sections.
        dose_array : numpy.ndarray
            Dose array corresponding to the matrix.
        numSectors : int, optional
            Number of Sectors to divide the matrix into (default is 3). i.e. vertically
        show : bool, optional
            Flag indicating whether to show the plot of the sections (default is False).
        labelsChange : bool, optional
            Flag indicating whether to change the labels of the sections (default is False). This value 
            is set False unless you are trying to change the labels across all the list of 3D matrices
        secIdx : int, optional
            Index of the section to be modified if labelsChange is True (default is 1). This value 
            is not necessay unless you are trying to change the labels across all the list of 3D matrices
        numSections : int, optional
            Number of Sections to divide the matrix into (default is 3). i.e. horizonatally. This value 
            is not necessay unless you are trying to change the labels across all the list of 3D matrices
    
        Returns:
        -------
        list
            List containing the following elements of the given section:
            - idxsT : list
                List of indices.
            - structure : list
                List of structures values .
            - dose : list
                List of dose values 
            - mask1 : numpy.ndarray
                Masked matrix with section labels.
            - centeRs : list
                List of centers .
        """
        matrix1 = np.array(matrix1)
        matrix = transformationX(matrix1).toPolarMatrix()[0]
        centeRs = transformationX(matrix1).toPolarMatrix()[-1]
        matrix = np.array(matrix)
        dose_array = np.array(dose_array)
        valEnd = []
        valEnd=[]
        indices, subset, subsetM, dose, section  = np.empty([5,numSectors]).tolist()
        val1 = 0;        
        for i in range(numSectors-1):
             for val in np.linspace(0,360,100):
                indices[i] = np.where((matrix >val1) & (matrix <= val))
                subset[i] = matrix[indices[i]]
                subsetM[i] = matrix1[indices[i]]
                dose[i] = dose_array[indices[i]]
                if np.sum(subsetM[i]) >= np.sum(matrix1) / numSectors:
                    section[i] = subsetM[i]
                    val1 = val
                    valEnd.append(val1)
                    break        
        mask = np.zeros_like(matrix, dtype=bool)
        for i in range(numSectors-1):
            mask[indices[i]] = True
        indices[-1]=np.where(mask==False)
        subset[-1]= matrix[~mask]
        section[-1] = np.array(matrix1[~mask])
        dose[-1]=np.array(dose_array[~mask])
        print(*[np.sum(section[i]) for i in range(numSectors)])
    
        idxsT = Operations(matrix1).intersecting_coords(indices)
        mask1 = np.copy(matrix1)
        
        for i in range(numSectors):
            mask1[idxsT[i]]=2*i+1
        if labelsChange == True:
            for i in range(numSectors):
                mask1[idxsT[i]]=i+1+ numSectors*secIdx
        centers=[]
    
        centers = []
        if show == True:
            cm = plt.get_cmap('gist_rainbow')
            color_list = [cm(1. * i / numSectors) for i in range(numSectors)]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i, coord in enumerate(idxsT):
                xs = coord[0]
                ys = coord[1]
                zs = coord[2]
                center = [np.mean(xs), np.mean(ys), np.mean(zs)]
                centers.append(center)
                ax.scatter(xs, ys, zs, color=color_list[i], label=f'section{i+1}', alpha=0.03, s=100)
                ax.scatter(*center, c='black', marker='o', s=100, zorder=10)
        
            ax.legend()
            plt.show()
    
        dose = [dose_array[idxsT[i][0], idxsT[i][1], idxsT[i][2]] for i in range(numSectors)]
        structure = [matrix1[idxsT[i][0], idxsT[i][1], idxsT[i][2]] for i in range(numSectors)]
    
        return [idxsT, structure, dose, mask1, centeRs]


    def divideOperations(self, dose_array3D, numSections=5, numSectors=3, labelsChange=False, show = False):
        """
        Divide the 3D matrix into multiple sections and sectors.
    
        Args:
       dose_array3D: 3D array representing the dose matrix.
       numSections (optional): The number of sections to divide the matrix into. Defaults to 5.
       numSectors (optional): The number of sectors within each section. Defaults to 3.
       labelsChange (optional): Boolean flag indicating whether to change labels. Defaults to False.

       Returns:
           A list containing:
           - fiveMasks: Masks for the divided sections.
           - concatenatedMasks: Concatenated masks of all sections.
           - idxs: Index values from the divided sections.
           - structs: List of Structural elements from the divided sections.
           - dose: List of Dose values from the divided sections.
           - centeRs: Centers of the divided sections.
       """
    
        # Divide the matrix into non-zero sections
        dividedNonZeroMat1 = self.divide_matrix(dose_array3D, numSections = numSections, numSectors= numSectors)
    
        # Perform three-section division within each non-zero section
        divided3Secs = [
            self.threeSections(
                dividedNonZeroMat1[0][i],
                dividedNonZeroMat1[1][i],
                show=show,
                numSectors=numSectors,
                labelsChange=labelsChange,
                secIdx=i
            )
            for i in range(numSections)
        ]
    
        # Select masks and centers from the divided sections
        idxs = Operations().selectElements(divided3Secs, 0)
        structs = Operations().selectElements(divided3Secs, 1)
        dose = Operations().selectElements(divided3Secs, 2)
        fiveMasks = Operations().selectElements(divided3Secs, 3)
        centeRs = Operations().selectElements(divided3Secs, 4)
    
        # Concatenate the masks of all sections
        concatenatedMasks = np.concatenate(tuple(fiveMasks), axis=0)
    
        return [fiveMasks, concatenatedMasks, idxs, structs, dose, centeRs]
#%%
    

#%%
# mat = np.random.randint(0,2, (20,20,20))
# init2 = Sectioning(mat)
# val = init2.divide_matrix(mat)
# valts = init2.threeSections(val[0][2], val[0][2], numSectors = 4, show = False)
# #%%
# valtsLabeled = init2.threeSections(val[0][2], val[0][2], numSectors = 4, numSections=5, labelsChange = True, secIdx=4)
# #%%
# divOp = init2.divideOperations(mat,numSectors = 3, numSections=5,labelsChange=True)
# #%%

# valtsLabeled = init2.threeSections(val[0][2], val[0][2], numSectors = 3, numSections=5, labelsChange = True, secIdx=5)

# np.unique(divOp[1])

# #%%

# np.unique(init2.threeSections(val[0][2], val[0][2], numSectors = 3, numSections=5, labelsChange = True, secIdx=2)[3])






