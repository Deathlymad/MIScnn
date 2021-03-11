#==============================================================================#
#  Author:       Philip Meyer                                                  #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
# Internal libraries/scripts
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction
from itertools import chain, combinations
#-----------------------------------------------------#
#             Subfunction class: Mirroring            #
#-----------------------------------------------------#
""" A Mirroring Subfunction class which can be used for creating multiple instances mirrored around the various axis.

Methods:
    __init__                Object creation function
    preprocessing:          Clipping the imaging data
    postprocessing:         Do nothing
"""
class Mirroring(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """
        Parameter:
            axisList (bool list):       A list of axies that should be mirrored. the image has a higher dimension than the provided data the axises are assumed to not be mirrored.
                                        If the axisList is longer than the image dimension it just stops early.
            combineMirroring (boolean): Should any combination of axis mirrorings be generated. this could avoid multiple mirroring steps.
        Return:
            None
    """
    def __init__(self, axisList, combineMirroring = True):
        self.axisList = axisList
        self.combineMirroring = combineMirroring
        comb = len([a for a in axisList if a])
        if (comb > 3 and combineMirroring):
            """
            This warning is meant to avoid accidental excessive Sample generation. Theoretically it would allow to also mirror the colors of an RGB image which would not make sense but it would double the instances.
            """
            print("WARNING: You are applying mirroring to more axises than spacial dimensions while also allow any combinations of mirrorings. This may be very memory consuming.")
            print("Mirroring will, in this case, generate " + str(2 ** comb) + " instances for each sample.") #the grwoth of mirroring for each axis with combination is 2^n

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True):
        # Access image
        image = sample.img_data
        #create result list and add sample
        results = []
        mirrorAxises = [i for i, val in enumerate(self.axisList) if val] 
        if not self.combineMirroring:
            results.append(sample)
            for a in mirrorAxises:
                newSample = sample.copy()
                newSample.img_data = np.flip(image, a)
                newSample.add_extended_data({"mirroring":a})
                results.append(newSample)
        else:
            tmp = chain.from_iterable(combinations(mirrorAxises, r) for r in range(len(mirrorAxises)+1))
            tmp = list(tmp)
            for axisTuple in tmp:
                newSample = sample.copy()
                newSample.img_data = np.flip(image, axisTuple)
                newSample.add_extended_data({"mirroring":list(axisTuple)})
                results.append(newSample)
        # Update the sample with the normalized image
        return results
    
    
    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, sample, prediction):
        mirrorData = sample.get_extended_data()
        if ("mirroring" in mirrorData.keys()):
            return np.flip(prediction, tuple(mirrorData["mirroring"])) #revert mirroring based on existant data. CRUCIALLY THIS DOES NOT MERGE THE PREDICTIONS.
        else:
            return prediction
