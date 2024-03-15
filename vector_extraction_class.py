import yaml,os,pickle
import parallelize_computation
import pandas as pd
import numpy as np
from datetime import datetime
import Load_preprocess_images.image_preprocessing_functions
import Quality_control_HCI.compute_global_values
import Embeddings_extraction_from_image.batch_compute_embeddings
import ScaleFEx_from_crop.compute_ScaleFEx
import time
from scipy.spatial import KDTree



class Screen_Compute: #come up with a better name
    """
    Class representing the computation of screen data.

    Methods:
        __init__(yaml_path='parameters.yaml'): 
            Initializes the Screen_Compute object with parameters from a YAML file.
    """
    def __init__(self, yaml_path='parameters.yaml'):
        """
        Initializes the Screen_Compute object with parameters from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file containing parameters. Default is 'parameters.yaml'.
        """

        # Read the yaml file
        with open(yaml_path, 'rb') as f:
            self.parameters = yaml.load(f.read(), Loader=yaml.CLoader)
       
        self.visualization = self.parameters['segmentation']['visualization']
        # Determine the type of computation to be used
        self.computation = 'local' if self.parameters['AWS']['use_AWS'] in [
            'no', 'N', 'NO', 'n', 'Nope'] else 'AWS'

        # Import the data retrieval function
        self.data_retrieve = import_module(self.parameters[self.computation]['query_function'])

        # Print the experiment folder
        print("retrieving files from ", (self.parameters['location_parameters']['exp_folder']))

        # Get the files
        files = self.data_retrieve.query_data(self.parameters['location_parameters']['exp_folder'],plate_type= self.parameters['location_parameters']['plate_type'])

        # Perform Flat Field Correction (FFC)
        self.flat_field_correction = {}
        if self.parameters['FFC'] is True:
            if not os.path.exists(self.parameters['location_parameters']['saving_folder'] +
                                   self.parameters['location_parameters']['experiment_name'] + '_FFC.p'):
                print('Flat Field correction image not found in ' +
                      self.parameters['location_parameters']['saving_folder'],
                      ' Generating FFC now')
                self.flat_field_correction = self.data_retrieve.flat_field_correction_on_data(
                    files, self.parameters['type_specific']['channel'], n_images=20)
                pickle.dump(self.flat_field_correction,
                            open(self.parameters['location_parameters']['saving_folder'] +
                                 self.parameters['location_parameters']['experiment_name'] + '_FFC.p', "wb"))

            else:
                print('Flat Field correction image found in ' +
                      self.parameters['location_parameters']['saving_folder'],
                      ' Loding FFC')
                self.flat_field_correction = pickle.load(
                    open(self.parameters['location_parameters']['saving_folder'] +
                         self.parameters['location_parameters']['experiment_name'] + '_FFC.p', "rb"))
        else:
            for channel in self.parameters['type_specific']['channel']:
                self.flat_field_correction[channel] = 1

        # Loop over plates and start computation
        for plate in self.parameters['location_parameters']['plates']:
            self.start_computation(plate, files)

### Start computation
            
    def start_computation(self,plate,files):
        
        # self.plate=plate
        
        task_files=files.loc[files.plate==plate]
        wells, task_fields = self.data_retrieve.make_well_and_field_list(task_files)

        if not os.path.exists(self.parameters['location_parameters']['saving_folder']+self.parameters['vector_type']):
            os.makedirs(self.parameters['location_parameters']['saving_folder']+self.parameters['vector_type'])

        csv_file = self.parameters['location_parameters']['saving_folder']+self.parameters['vector_type']+'/'+self.parameters['location_parameters']['experiment_name']+'_'+plate+'_'+self.parameters['vector_type']+'.csv'
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

            
            print(well, plate, datetime.now())
            for site in task_fields:
    
                print(site, well, plate, datetime.now())
                stime=time.perf_counter()
                np_images, original_images = Load_preprocess_images.image_preprocessing_functions.load_and_preprocess(task_files,
                                    self.parameters['type_specific']['channel'],well,site,self.parameters['type_specific']['zstack'],self.data_retrieve,
                                    self.parameters['type_specific']['img_size'],self.flat_field_correction,
                                    self.parameters['downsampling'])#,return_original=self.parameters['QC'])
                try:
                    original_images.shape
                except NameError:
                    print('Images corrupted')
                print('images load and process time ',time.perf_counter()-stime)

                if np_images is not None:
                    # stime = time.perf_counter()
                    if self.parameters['segmentation']['csv_coordinates']=='':
                        center_of_mass=self.segment_crop_images(original_images[0])
                        center_of_mass=[row + [n] for n,row in enumerate(center_of_mass)]
                        if self.parameters['type_specific']['compute_live_cells'] is False:
                            live_cells=len(center_of_mass)
                        else:
                            print('to be implemented')
                    else:
                        locations=pd.read_csv(self.parameters['segmentation']['csv_coordinates'],index_col=0)
                        locations['plate']=locations['plate'].astype(str)
                        locations=locations.loc[(locations.well==well)&(locations.site==site)&(locations.plate==plate)]
                        center_of_mass=np.asarray(locations[['coordX','coordY','cell_id']])

                        if self.parameters['type_specific']['compute_live_cells'] is False:
                            live_cells=len(center_of_mass)
                            
                    # print('coordinates time ',time.perf_counter()-stime)    
                        
                    #print(center_of_mass)
                    # stime = time.perf_counter()
                    if self.parameters['QC']==True:
                        indQC=0

                        QC_vector,indQC = Quality_control_HCI.compute_global_values.calculateQC(len(center_of_mass),live_cells,
                                            self.parameters['location_parameters']['experiment_name'],original_images,well,plate,site,self.parameters['type_specific']['channel'],
                                            indQC,self.parameters['type_specific']['neurite_tracing'])
                        if not os.path.exists(csv_fileQC):
                            QC_vector.to_csv(csv_fileQC,header=True)
                        else:
                            QC_vector.to_csv(csv_fileQC,mode='a',header=False)
                    # print('QC time ',time.perf_counter()-stime)

                    if self.parameters['tile_computation'] is True:
                        ind=0
                        vector=pd.DataFrame(np.asarray([plate,well,site]).reshape(1,3),columns=['plate','well','site'],index=[ind])
                        vector=pd.concat([vector,Embeddings_extraction_from_image.batch_compute_embeddings.Compute_embeddings(np_images,ind,self.parameters['type_specific']['channel'],
                                                                                            self.parameters["device"],weights=self.parameters['weights_location']).embeddings],axis=1)
                        if not os.path.exists(csv_file[:-4]+'Tile.csv'):
                            vector.to_csv(csv_file[:-4]+'Tile.csv',header=True)
                        else:
                            vector.to_csv(csv_file[:-4]+'Tile.csv',mode='a',header=False)
              
                    for x,y,n in center_of_mass:
                       
                        # stime = time.perf_counter()
                        crop=np_images[:,int(float(x)-self.parameters['type_specific']['ROI']):int(float(x)+self.parameters['type_specific']['ROI']),
                                           int(float(y)-self.parameters['type_specific']['ROI']):int(float(y)+self.parameters['type_specific']['ROI']),:]
                        # if ((x-self.parameters['type_specific']['ROI']<0) or (x-self.parameters['type_specific']['ROI']>self.parameters['location_parameters']['image_size'][0]) or
                        #     (y-self.parameters['type_specific']['ROI']<0) or (y-self.parameters['type_specific']['ROI']>self.parameters['location_parameters']['image_size'][1])):
                        if crop.shape != (len(self.parameters['type_specific']['channel']),self.parameters['type_specific']['ROI']*2,self.parameters['type_specific']['ROI']*2,1):
                            print(crop.shape, "cell on the border")
                            continue
                        else:
                            ind=0
                            vector=pd.DataFrame(np.asarray([plate,well,site,x,y,n]).reshape(1,6),columns=['plate','well','site','coordX','coordY','cell_id'],index=[ind])

                        
                            if self.parameters['location_parameters']['coordinates_csv']=='':
                             
                                tree = KDTree([row[:2] for row in center_of_mass])

                                # Query the nearest distance and the index of the nearest point
                                distance, _ = tree.query([x,y], k=2)    
                                vector['distance']=distance[1] 

                            
                            # print(crop.shape)

                            if 'mbed' in self.parameters['vector_type']:

                                vector=pd.concat([vector,Embeddings_extraction_from_image.batch_compute_embeddings.Compute_embeddings(crop,0,self.parameters['type_specific']['channel'],
                                                                                                self.parameters["device"],weights=self.parameters['weights_location']).embeddings],axis=1)
                                
                                if not os.path.exists(csv_file):
                                    vector.to_csv(csv_file,header=True)
                                else:
                                    vector.to_csv(csv_file,mode='a',header=False)
                                print('embedding_computatiomn time ',time.perf_counter()-stime)
                            
                            elif ('cale' in self.parameters['vector_type']) or ('FEx' in self.parameters['vector_type']) or ('fex' in self.parameters['vector_type']):
                                
                                vector=pd.concat([vector,ScaleFEx_from_crop.compute_ScaleFEx.ScaleFEx(crop, channel=self.parameters['type_specific']['channel'],
                                                    mito_ch=self.parameters['type_specific']['Mito_channel'], rna_ch=self.parameters['type_specific']['RNA_channel'],
                                                    neuritis_ch=self.parameters['type_specific']['neurite_tracing'],downsampling=self.parameters['downsampling'], 
                                                    visualization=self.parameters['visualize_masks'], roi=int(self.parameters['type_specific']['ROI'])
                                                    ).single_cell_vector.loc[1,0]],axis=1)
                                #print(vector)
                                if not os.path.exists(csv_file):
                                    vector.to_csv(csv_file,header=True)
                                else:
                                    vector.to_csv(csv_file,mode='a',header=False)

                                # print('vector_computatiomn time ',time.perf_counter()-stime)

                            else:
                                print(' Not a valid vector type entry')
                            

        if self.computation=='local':
            if self.parameters['local']['parallel'] is True:
                function = compute_vector
                parallelize_computation.parallelize_local(wells,function)
            else:
                for well in wells:
                    stime=time.perf_counter()
                    compute_vector(well)
                    print('well time ',time.perf_counter()-stime)
        elif self.computation=='AWS':
            print('Gab to finish :) ')

    def segment_crop_images(self,img_nuc):

        # extraction of the location of the cells
    
        nls=import_module(self.parameters['segmentation']['segmenting_function'])
        
            
        img_mask=nls.compute_DNA_mask(img_nuc)
        center_of_mass = nls.retrieve_coordinates(img_mask,
                    cell_size_min=self.parameters['segmentation']['min_cell_size']*self.parameters['downsampling'],
                    cell_size_max=self.parameters['segmentation']['max_cell_size']/self.parameters['downsampling'])
        if self.visualization is True:
            self.show_image(img_nuc,img_mask)

        try:
            center_of_mass
            print('N of cells found: ',len(center_of_mass))
        except NameError:
            center_of_mass = []
            print('No Cells detected')
        

        return center_of_mass

    def show_image(self,img,nuc):
        import matplotlib.pyplot as plt
        _,ax=plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(img)
        ax[1].imshow(nuc)
        plt.show()                    


def import_module(module_name):
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        print(f"Module '{module_name}' not found.")
        return None
    
# total_time=time.perf_counter()

# if __name__ == "__main__":
    
# 	Screen_Compute()

# print('total time: ',time.perf_counter()-total_time)

import cProfile
import pstats

def main():
    Screen_Compute()

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    main()  # Your main execution block

    pr.disable()
    with open("profile_results.txt", "w") as f:  # Choose a file path/name
        ps = pstats.Stats(pr, stream=f)
        ps.sort_stats("cumulative")  # Sorting by cumulative time
        ps.print_stats()
