# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:33:37 2023

@author: rghimire
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# import pickle
# import opns
import pandas as pd
import seaborn as sns
#%%
cm = plt.get_cmap('gist_rainbow')
colorList = [cm(1. * i / 40) for i in range(40)]
class PltX:
    """
    This class deals with plotting of structure and calculating Dose Volume Hisotgram (DVH)
    """
    
    def __init__(self, structure = np.random.randint(0,2, (20,20,20)), dose = np.random.randint(0,2, (20,20,20))):
        """
        Parameters
        ----------
        structure : ndarray or list
            A single 3D matrix or a list of 3D matrices.
        dose : ndarray or list
            A single 3D matrix or a list of 3D matrices.
        color : list of str, optional
            The colors to use for the scatter plots. Default is a list of 100 colors.
        """
        self.structure = structure
        self.dose = dose
        
    def plot_3d_scatter(self, color=colorList,  titles = None):
        """
    
        Returns
        -------
        fig : plt.Figure
            The figure containing the scatter plots.
    
        """
    
        if not isinstance(self.structure, list):
            matrices = [self.structure]  # Convert single matrix to list of matrices
        else:
            matrices = self.structure
        
        num_matrices = len(matrices)
        num_rows = int(np.ceil(np.sqrt(num_matrices)))  # Number of rows in subplot grid
        num_cols = int(np.ceil(num_matrices / num_rows))  # Number of columns in subplot grid
    
        # Create figure and subplot grid
        fig = plt.figure()
        num_matrices = len(matrices)
        num_rows = int(np.ceil(np.sqrt(num_matrices)))
        num_cols = int(np.ceil(num_matrices / num_rows))
        
        for i, matrix in enumerate(matrices):
            if not isinstance(matrix, np.ndarray):
                raise ValueError("Input must be a NumPy ndarray or a list of NumPy ndarrays.")
            
            if matrix.ndim != 3:
                raise ValueError("Input must be a 3D matrix.")
            
            ax = fig.add_subplot(num_rows, num_cols, i + 1, projection='3d')
            x, y, z = matrix.nonzero()
            ax.scatter(x, y, z, zdir='z', c=matrix[x, y, z], s=3, cmap=ListedColormap(color))
            ax.set_xlim([0, matrix.shape[0]])
            ax.set_ylim([0, matrix.shape[1]])
            ax.set_zlim([0, matrix.shape[2]])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(20, 120)
            
            if titles is None:
                ax.set_title(f'Matrix {i + 1}')
            else:
                ax.set_title(titles[i])
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show()
        
      
    def getHistogram(self, data, binSize=np.arange(0, 50.02, 0.01), type1="cumulative", colorIdx=0, show=True):
        """
        Generates data and plots different types of histograms: "cumulative", "differential", and Lance's "lMethod" plot of the given data.
    
        Parameters:
            data (array-like): The input data, a 1D list, for which the histogram or CDF needs to be computed.
            binSize (array-like, optional): The bin edges for the histogram. Default is np.arange(0, 50.02, 0.01).
            type1 (str, optional): The type of plot to generate. Available options: 'cumulative', 'differential', 'lMethod'.
                                  Default is 'cumulative'.
            colorIdx (int, optional): The index of the color to be used for the plot. Default is 0.
            show (bool, optional): Whether to display the plot. If False, the plot is not displayed but generated.
                                  Default is True.
    
        Returns:
            list: A list containing the bin edges and the corresponding values for the histogram or CDF.
        """
    
        bin_edges = binSize
        fig, ax = plt.subplots()
    
        if type1 == "cumulative":
            n, bins = np.histogram(data, bins=bin_edges, density=True)
            cdf = 1 - (np.cumsum(n) * np.diff(bins))
            ax.plot(bins[:-1], cdf, color=colorList[colorIdx])
            data = [bins[:-1], cdf]
        elif type1 == "differential":
            n, bins, patches = ax.hist(data, bins=binSize, density=True, alpha=0)
            ax.plot(bins[:-1], n, color=colorList[colorIdx])
            data = [bins, n]
        elif type1 == "lMethod":
            volume_bin = []
            x_axis = np.arange(0, 50.02, 0.01)
            for i in x_axis:
                bin_height = np.count_nonzero(data > i) / np.count_nonzero(data)
                volume_bin.append(bin_height)
            ax.plot(x_axis, volume_bin, color=colorList[colorIdx])
            data = [x_axis, volume_bin]
    
        ax.grid()
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Volume (%)")
        ax.set_title("DVH")
    
        if show:
            plt.show()
        else:
            plt.close()
    
        return data


    def getDVH(self, binSize = np.arange(0, 50.02, 0.01), names=['S_{}'.format(j) for j in range(1, 100)], type1="multiple", typehist="cumulative", show=True):
        """
        Generates and plots the DVH (Dose-Volume Histogram) based on the given dose and volume data.
    
        Parameters
        ----------
        dose_array : array-like
            The dose data as a list or array.
        dict_volumes : array-like
            The volume data as a list or array.
        names : list, optional
            The names of the DVH curves. Default is ['S_1', 'S_2', ..., 'S_99'].
        type1 : str, optional
            The type of plot to generate. Available options: 'multiple', 'single', 'mat', 'mats'.
            Default is 'multiple'.
        typehist : str, optional
            The type of histogram to use for plotting. Available options: 'cumulative', 'differential', 'lMethod'.
            Default is 'cumulative'.
        show : bool, optional
            Whether to display the plot. If True, the plot is shown. If False, the plot is generated but not displayed.
            Default is True.
    
        Returns
        -------
        list
            A list containing the bin edges and the corresponding values for the histogram or CDF.
    
        Note
        ----
        The method self.getHistogram is called to generate the histogram data.
        """
        dose_array = self.dose
        dict_volumes = self.structure
        data = []
        NUM_COLORS=len(self.dose)
        colorList = [cm(1.*i/NUM_COLORS) for i in range(100)]
        fig, ax = plt.subplots()
        
        if type1 == "multiple" or type1 == "mats":
            for idx, volume in enumerate(dict_volumes):
                volume = np.array(volume)
                dose_R = np.array(dose_array[idx])
                volume_voxels_count = np.count_nonzero(volume)
                
                if volume_voxels_count == 0:
                    continue
                
                masked_dose = dose_R[volume > 0]
                values = self.getHistogram(masked_dose, binSize, type1=typehist, show=False)
                ax.plot(values[0], values[1], label=names[idx], c=colorList[idx])
                data.append(values)

        
        elif type1 == "single" or type1 == "mat":
            volume = np.array(dict_volumes)
            dose_R = np.array(dose_array)
            masked_dose = dose_R[volume > 0]
            values = self.getHistogram(masked_dose, binSize, type1=typehist, show=False)
            data = values
            ax.plot(values[0], values[1])

        
        ax.grid()
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Volume (%)")
        ax.set_title("DVH")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return data
    def create_grouped_boxplot(self,df, grtoMelt, grID, var_name="var", value_name="value", hue = "patient", xName = "var", title = "BowelTrue", xLabel = "var", yLabel = "value",figsize=(10, 6), fontsize = 12):
        """
        Create a grouped boxplot using Seaborn library.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame containing the data.
        grtoMelt : list or str
            The column(s) to be melted.
        grID : list or str
            The column(s) to be used as the grouping identifiers.
        var_name : str, optional
            The name of the variable column in the melted DataFrame. Default is "var".
        value_name : str, optional
            The name of the value column in the melted DataFrame. Default is "value".
        hue : str, optional
            The column name to be used for grouping the boxplots. Default is "patient".
        xName : str, optional
            The label for the x-axis. Default is "var".
        title : str, optional
            The title of the plot. Default is "BowelTrue".
        xLabel : str, optional
            The label for the x-axis. Default is "var".
        yLabel : str, optional
            The label for the y-axis. Default is "value".
        figsize : tuple, optional
            The figure size. Default is (10, 6).
        fontsize : int, optional
            The font size for title and labels. Default is 12.
        
        Returns
        -------
        df : pandas.DataFrame
            The input DataFrame.
        fig : matplotlib.figure.Figure
            The generated figure.
        """
     
         # Melt the DataFrame
        melted_df = pd.melt(df, id_vars=grID, \
                            value_vars=grtoMelt, var_name=var_name, value_name=value_name)
        plt.subplots(figsize=figsize)
        # Create the grouped boxplot
        sns.boxplot(x=melted_df[xName].astype(str), y=melted_df[value_name].astype(float), hue=melted_df[hue].astype(str))

        # Set the title and labels
        plt.title(title, fontsize = fontsize)
        plt.xlabel(xLabel, fontsize = fontsize)
        plt.ylabel(yLabel, fontsize = fontsize)
        # Add gridlines
        plt.grid(True)
   
       # Increase fontsize of tick labels
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
   

        # Show the plot
        plt.show()

        
#%%
# with open(r'E:\work\test\OrganizedSSD\gynR_003point25resampledstranddose.dat', 'rb') as f:
#     strAndDoseR3 = pickle.load(f)
# nonZval = opns.nonZeroMatrixPad(*strAndDoseR3)   
# # PltX(nonZval[1]).plot_3d_scatter()      
# PltX().getHistogram(nonZval[1])
# PltX([nonZval[0],nonZval[0]],  [nonZval[1],nonZval[1]]).getDVH(type1 = "multiple")
# PltX().getDVH(nonZval[1][np.nonzero(nonZval[0])], nonZval[0][np.nonzero(nonZval[0])],type1 = "single")


















