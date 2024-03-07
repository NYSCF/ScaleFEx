import data_query
import yaml,os,pickle
import parallelize_computation
import pandas as pd
import numpy as np
from datetime import datetime
import Load_preprocess_images.image_preprocessing_functions
import Quality_control_HCI.compute_global_values
import cv2


class Screen_Compute: #come up with a better name

    def __init__(self,yaml_path='parameters.yaml'):
 
        with open(yaml_path) as f:
            self.parameters=yaml.load(f.read(),Loader=yaml.CLoader)
        f.close()


        if self.parameters['AWS']['use_AWS'] in['no','N','NO','n','Nope']:
            print('Local computation')
            self.computation='local'
                       
        
        elif self.parameters['AWS']['use_AWS'] in ['yes','Y','YES','yes','y']:
            self.computation='AWS'
            print(' AWS computation')   

        else:
            print(self.parameters['AWS']['use_AWS'], ' is not an accepted character. Please specify Yes or No')

        self.data_retrieve=import_module(self.parameters[self.computation]['query_function'])
    
        print("retrieving files from ", (self.parameters['location_parameters']['exp_folder']))
        files=self.data_retrieve.query_data(self.parameters['location_parameters']['exp_folder'])
### FFC
        if self.parameters['FFC'] is True:
            if not os.path.exists(self.parameters['location_parameters']['saving_folder']+self.parameters['location_parameters']['experiment_name']+'_FFC.p'):
                print('Flat Field correction image not found in ' + self.parameters['location_parameters']['saving_folder'],
                      ' Generating FFC now')
                self.flat_field_correction={}
                
                    
                self.flat_field_correction=self.data_retrieve.flat_field_correction_on_data(files,self.parameters['type_specific']['channel'],n_images=20)

                pickle.dump(self.flat_field_correction, open(self.parameters['location_parameters']['saving_folder'] +
                        self.parameters['location_parameters']['experiment_name']+'_FFC.p', "wb"))
            else:
                print('Flat Field correction image found in ' + self.parameters['location_parameters']['saving_folder'],
                      ' Loding FFC')
                self.flat_field_correction = pickle.load(open(self.parameters['location_parameters']['saving_folder'] +
                        self.parameters['location_parameters']['experiment_name']+'_FFC.p', "rb"))
        else:
            for channel in self.parameters['type_specific']['channel']: 
                self.flat_field_correction[channel]=1

        for plate in self.parameters['location_parameters']['plates']:
            self.start_computation(plate,files)

### Start computation
            
    def start_computation(self,plate,files):
        #TB fixed
        
        # self.plate=plate
        task_files=files.loc[files.plate==plate]
        wells, task_fields = self.data_retrieve.make_well_and_field_list(task_files)
        if not os.path.exists(self.parameters['location_parameters']['saving_folder']+self.parameters['vector_type']):
            os.makedirs(self.parameters['location_parameters']['saving_folder']+self.parameters['vector_type'])

        csv_file = self.parameters['location_parameters']['saving_folder']+self.parameters['vector_type']+self.parameters['location_parameters']['experiment_name']+'_'+str(plate)+'Embeddings.csv'
        if self.parameters['QC']==True:
            csv_fileQC = self.parameters['location_parameters']['saving_folder']+'QC_analysis/'+self.parameters['location_parameters']['experiment_name']+'_'+str(plate)+'QC.csv'
            if not os.path.exists(self.parameters['location_parameters']['saving_folder']+'QC_analysis'):
                os.makedirs(self.parameters['location_parameters']['saving_folder']+'QC_analysis')

        wells=self.data_retrieve.check_if_file_exists(csv_file,wells,task_fields[-1])

        if wells[0] == 'Over':
            print('plate ', plate, 'is done')
            return
            
        def compute_vector(well):
            ''' Function that imports the images and extracts the location of cells'''
                #TB fixed 
            
            print(well, plate, datetime.now())
            for site in task_fields:
    
                print(site, well, plate, datetime.now())
                np_images, original_images = Load_preprocess_images.image_preprocessing_functions.load_and_preprocess(task_files,
                                    self.parameters['type_specific']['channel'],well,site,self.parameters['type_specific']['zstack'],self.data_retrieve,
                                    self.parameters['type_specific']['img_size'],self.flat_field_correction,
                                    self.parameters['downsampling'],return_original=self.parameters['QC'])
                try:
                    print(original_images.shape)
                except:
                    print('Images corrupted')

                if np_images is not None:
                    if self.parameters['segmentation']['csv_coordinates']=='':
                        center_of_mass=self.segment_crop_images(original_images[0])
                    else:
                        locations=pd.read_csv(self.parameters['segmentation']['csv_coordinates'],index_col=0)
                        locations['plate']=locations['plate'].astype(str)
                        locations=locations.loc[(locations.well==well)&(locations.site==site)&(locations.plate==plate)]
                        center_of_mass=np.asarray(locations[['coordX','coordY']])
                        

                    print(center_of_mass)

                    if self.parameters['QC']==True:
                        indQC=0
                        if self.parameters['type_specific']['compute_live_cells'] is False:
                            live_cells=len(center_of_mass)
                        else:
                            print('to be implemented')
                        QC_vector,indQC = Quality_control_HCI.compute_global_values.calculateQC(len(center_of_mass),live_cells,
                                            self.parameters['location_parameters']['experiment_name'],original_images,well,plate,site,self.parameters['type_specific']['channel'],
                                            indQC,self.parameters['type_specific']['neurite_tracing'])
                        if not os.path.exists(csv_fileQC):
                            QC_vector.to_csv(csv_fileQC,header=True)
                        else:
                            QC_vector.to_csv(csv_fileQC,mode='a',header=False)

        if self.computation=='local':
            if self.parameters['local']['parallel'] is True:
                function = compute_vector
                parallelize_computation.parallelize_local(wells,function)
            else:
                for well in wells:
                    compute_vector(well)
        elif self.computation=='AWS':
            print('Gab to finish :) ')

    def segment_crop_images(self,img_nuc):

        # extraction of the location of the cells
    
        nls=import_module(self.parameters['segmentation']['segmenting_function'])
        center_of_mass = nls.retrieve_coordinates(nls.compute_DNA_mask(img_nuc),
                    cell_size_min=self.parameters['segmentation']['min_cell_size']*self.parameters['downsampling'],
                    cell_size_max=self.parameters['segmentation']['max_cell_size']/self.parameters['downsampling'])
        try:
            center_of_mass
        except NameError:
            center_of_mass = []
            print('No Cells detected')
        

        return center_of_mass
                        


def import_module(module_name):
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        print(f"Module '{module_name}' not found.")
        return None
    
if __name__ == "__main__":
	Screen_Compute()
