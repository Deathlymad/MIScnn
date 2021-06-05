#==============================================================================#
#  Author:       Philip Meyer                                                #
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
from miscnn.utils.patch_operations import slice_matrix, concat_matrices, pad_patch, crop_patch

#-----------------------------------------------------#
#             Subfunction class: Clipping             #
#-----------------------------------------------------#
""" The patch creation function. It splits an image into subimages that are reassembeled afterwards.

Methods:
    __init__                Object creation function
    preprocessing:          Split the sample image into an array of subimages. That leads to a shape change from (x, y, z, c) to (batch_size, x, y, z, c)
    postprocessing:         merge the patches back together into one image.
"""
class SampleSplitting(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, three_dim=True, method="fullimage", patch_shape=None, patch_overlap=(0,0,0), skip_blanks=False, skip_class=0):
        # Exception: Analysis parameter check
        analysis_types = ["patchwise-crop", "patchwise-grid", "fullimage"]
        if not isinstance(method, str) or method not in analysis_types:
            raise ValueError('Non existent analysis type in preprocessing.')
        # Exception: Patch-shape parameter check
        if (method == "patchwise-crop" or method == "patchwise-grid") and \
            not isinstance(patch_shape, tuple):
            raise ValueError("Missing or wrong patch shape parameter for " + \
                             "patchwise analysis.")
        
        self.three_dim=three_dim
        self.method = method
        
        #TODO check if the following is relevant
        
        self.patch_shape = patch_shape
        self.patchwise_overlap = patch_overlap              # In patchwise_analysis, an overlap can be defined between adjuncted patches.
        self.patchwise_skip_blanks = skip_blanks            # In patchwise_analysis, patches, containing only the background annotation,
                                                            # can be skipped with this option. This result into only
                                                            # training on relevant patches and ignore patches without any information.
        self.patchwise_skip_class = skip_class              # Class, which will be skipped if patchwise_skip_blanks is True
    
    def set_data_aug(self, data_aug):
        self.data_aug = data_aug
    
    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True):
        # Run Fullimage analysis
        if self.method == "fullimage":
            self.analysis_fullimage(sample, training)
        # Run patchwise cropping analysis
        elif self.method == "patchwise-crop" and training:
            self.analysis_patchwise_crop(sample)
        # Run patchwise grid analysis
        else:
            if not training:
                sample.extended["patch_orig_shape"] = sample.img_data.shape
            self.analysis_patchwise_grid(sample, training)

    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, sample, prediction):
        if self.method == "patchwise-crop" or \
            self.method == "patchwise-grid":
            # Check if patch was padded
            if "patch_orig_slice" in sample.extended.keys():
                prediction = crop_patch(prediction, sample.extended["patch_orig_slice"])
            # Load cached shape & Concatenate patches into original shape
            seg_shape = sample.extended["patch_orig_shape"]
            prediction = concat_matrices(patches=prediction,
                                    image_size=seg_shape,
                                    window=self.patch_shape,
                                    overlap=self.patchwise_overlap,
                                    three_dim=self.three_dim)
        # For fullimages remove the batch axis
        else : prediction = np.squeeze(prediction, axis=0)
    
    
    #---------------------------------------------#
    #           Patch-wise grid Analysis          #
    #---------------------------------------------#
    def analysis_patchwise_grid(self, sample, training):
        # Slice image into patches
        patches_img = slice_matrix(sample.img_data, self.patch_shape,
                                   self.patchwise_overlap,
                                   self.three_dim)
        if training:
            # Slice segmentation into patches
            patches_seg = slice_matrix(sample.seg_data, self.patch_shape,
                                       self.patchwise_overlap,
                                       self.three_dim)
        else : patches_seg = None
        # Skip blank patches (only background)
        if training and self.patchwise_skip_blanks:
            # Iterate over each patch
            for i in reversed(range(0, len(patches_seg))):
                # IF patch DON'T contain any non background class -> remove it
                if not np.any(patches_seg[i][...,self.patchwise_skip_class] != 1):
                    del patches_img[i]
                    del patches_seg[i]
        # Concatenate a list of patches into a single numpy array
        img_data = np.stack(patches_img, axis=0)
        if training : seg_data = np.stack(patches_seg, axis=0)
        # Pad patches if necessary
        if img_data.shape[1:-1] != self.patch_shape and training:
            img_data = pad_patch(img_data, self.patch_shape,return_slicer=False)
            seg_data = pad_patch(seg_data, self.patch_shape,return_slicer=False)
        elif img_data.shape[1:-1] != self.patch_shape and not training:
            img_data, slicer = pad_patch(img_data, self.patch_shape,
                                         return_slicer=True)
            sample.extended["patch_orig_slice"] = slicer
        # Run data augmentation
        if (self.data_aug is not None) and training:
            img_data, seg_data = self.data_aug.run(img_data, seg_data)
        elif (self.data_aug is not None) and not training:
            img_data = self.data_aug.run_infaug(img_data)

        sample.img_data = img_data
        sample.seg_data = seg_data
    
    #---------------------------------------------#
    #           Patch-wise crop Analysis          #
    #---------------------------------------------#
    def analysis_patchwise_crop(self, sample):
        # If skipping blank patches is active
        if self.patchwise_skip_blanks:
            # Slice image and segmentation into patches
            patches_img = slice_matrix(sample.img_data, self.patch_shape,
                                       self.patchwise_overlap,
                                       self.three_dim)
            patches_seg = slice_matrix(sample.seg_data, self.patch_shape,
                                       self.patchwise_overlap,
                                       self.three_dim)
            # Skip blank patches (only background)
            for i in reversed(range(0, len(patches_seg))):
                # IF patch DON'T contain any non background class -> remove it
                if not np.any(patches_seg[i][...,self.patchwise_skip_class] != 1):
                    del patches_img[i]
                    del patches_seg[i]
            # Select a random patch
            pointer = np.random.randint(0, len(patches_img))
            img = patches_img[pointer]
            seg = patches_seg[pointer]
            # Expand image dimension to simulate a batch with one image
            img_data = np.expand_dims(img, axis=0)
            seg_data = np.expand_dims(seg, axis=0)
            # Pad patches if necessary
            if img_data.shape[1:-1] != self.patch_shape:
                img_data = pad_patch(img_data, self.patch_shape,
                                     return_slicer=False)
                seg_data = pad_patch(seg_data, self.patch_shape,
                                     return_slicer=False)
            # Run data augmentation
            if (self.data_aug is not None):
                img_data, seg_data = self.data_aug.run(img_data,
                                                                seg_data)
        # If skipping blank is not active -> random crop
        else:
            # Access image and segmentation data
            img = sample.img_data
            seg = sample.seg_data
            # If no data augmentation should be performed
            # -> create Data Augmentation instance without augmentation methods
            if (self.data_aug is None):
                cropping_data_aug = Data_Augmentation(cycles=1,
                                            scaling=False, rotations=False,
                                            elastic_deform=False, mirror=False,
                                            brightness=False, contrast=False,
                                            gamma=False, gaussian_noise=False)
            else : cropping_data_aug = self.data_aug
            # Configure the Data Augmentation instance to cropping
            cropping_data_aug.cropping = True
            cropping_data_aug.cropping_patch_shape = self.patch_shape
            # Expand image dimension to simulate a batch with one image
            img_data = np.expand_dims(img, axis=0)
            seg_data = np.expand_dims(seg, axis=0)
            # Run data augmentation and cropping
            img_data, seg_data = cropping_data_aug.run(img_data, seg_data)
        
        sample.img_data = img_data
        sample.seg_data = seg_data

    #---------------------------------------------#
    #             Full-Image Analysis             #
    #---------------------------------------------#
    def analysis_fullimage(self, sample, training):
        # Access image and segmentation data
        img = sample.img_data
        if training : seg = sample.seg_data
        # Expand image dimension to simulate a batch with one image
        img_data = np.expand_dims(img, axis=0)
        if training : seg_data = np.expand_dims(seg, axis=0)
        # Run data augmentation
        if (self.data_aug is not None) and training:
            img_data, seg_data = self.data_aug.run(img_data, seg_data)
        elif (self.data_aug is not None) and not training:
            img_data = self.data_aug.run_infaug(img_data)
        sample.img_data = img_data
        sample.seg_data = seg_data
