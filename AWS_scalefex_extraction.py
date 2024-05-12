import yaml,os,pickle
import parallelize_computation
import pandas as pd
import numpy as np
from datetime import datetime
import data_query.query_functions_AWS as dq
import time
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


ROOT_DIR = '/'.join(__file__.split('/')[:-1])

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
            global ScaleFEx_from_crop
            import ScaleFEx_from_crop.compute_ScaleFEx
        if self.parameters['QC']==True:
            global Quality_control_HCI 
            import Quality_control_HCI.compute_global_values
        
        files = dq.query_data(self.parameters['fname_pattern'],plate_identifier = self.parameters['plate_identifier'],
                            delimiters = self.parameters['fname_delimiters'],exts=self.parameters['file_extensions'],
                            experiment_name = self.parameters['experiment_name'],plates=self.parameters['plates'], 
                            s3_bucket = self.parameters['s3_bucket'])

        print(files.head())
        print(files)
        
        self.vec_dir = os.path.join(self.parameters['vector_type'])
        if not os.path.exists(self.vec_dir):
            os.makedirs(self.vec_dir)  
        ffc_file = os.path.join(self.vec_dir,self.parameters['experiment_name'] + '_FFC.p')
        self.flat_field_correction = {}
        if self.parameters['FFC'] is True and os.path.exists(ffc_file):
                print(ffc_file + ' Found generating FFC now')
        else:
            for channel in self.parameters['channel']:
                self.flat_field_correction[channel] = 1

        plate = self.parameters['plates'][0]
        
        if self.parameters['QC']==True:
            self.csv_fileQC = os.path.join(self.vec_dir,self.parameters['experiment_name']+'_'+str(plate)+'_'+'QC'
                                    +'_'+str(self.parameters['subset_index'])+'.csv') 
        self.start_computation(plate, files)

### Start computation
            
    def start_computation(self,plate,files):
        
        task_files = dq.filter_task_files(files,self.parameters['subset_index'], self.parameters['nb_subsets']) 

        if os.path.exists(self.parameters['csv_coordinates']):
            self.locations=pd.read_csv(self.parameters['csv_coordinates'])
            self.locations=self.locations.loc[self.locations.plate.astype(str)==str(plate)]
            self.locations = dq.filter_coord(self.locations, task_files)
            
            wells=np.unique(self.locations.well)
        wells=np.unique(task_files.well)

        def compute_vector(well):
            ''' Function that imports the images and extracts the location of cells'''
            
            print(well, plate, datetime.now())
            csv_file = os.path.join(self.vec_dir,self.parameters['experiment_name']+'_'+plate
                                +'_SF_'+str(self.parameters['subset_index'])+'A'+'.csv')
            
            sites = np.unique(files.site)
            sites.sort()

            for site in sites:
                print(site, well, plate, datetime.now())
                np_images, original_images = dq.load_and_preprocess(task_files,
                                    self.parameters['channel'],well,site,self.parameters['zstack'],
                                    self.parameters['image_size'],self.flat_field_correction,
                                    self.parameters['downsampling'],return_original=self.parameters['QC'],
                                    s3_bucket = self.parameters['s3_bucket'])
                
                try:
                    original_images.shape
                except NameError:
                    print('Images corrupted')
                if np_images is not None:
                    if self.parameters['csv_coordinates']=='':
                        center_of_mass=self.segment_crop_images(np_images[0,:,:,0])
                        center_of_mass=[list(row) + [n] for n,row in enumerate(center_of_mass)]
                        if self.parameters['compute_live_cells'] is False:
                            live_cells=len(center_of_mass)
                        else:
                            print('to be implemented')
                    else:
                        
                        locations=self.locations
                        locations=locations.loc[(locations.well==well)&(locations.site==site)]
                        center_of_mass=np.asarray(locations[['coordX','coordY','cell_id']])
                        
                        if self.parameters['compute_live_cells'] is False:
                            live_cells=len(center_of_mass)
                            
                    if self.parameters['QC']==True:
                        indQC=0
                        QC_vector,indQC = Quality_control_HCI.compute_global_values.calculateQC(len(center_of_mass),live_cells,
                                            self.parameters['vector_type'],original_images,well,plate,site,self.parameters['channel'],
                                            indQC,self.parameters['neurite_tracing'])
                        self.csv_fileQC = dq.save_csv_file(QC_vector,self.csv_fileQC,self.parameters['max_file_size'])
              
                    for x,y,n in center_of_mass:
                        crop=np_images[:,int(float(x)-self.parameters['ROI']):int(float(x)+self.parameters['ROI']),
                                           int(float(y)-self.parameters['ROI']):int(float(y)+self.parameters['ROI']),:]
                        if crop.shape != (len(self.parameters['channel']),self.parameters['ROI']*2,self.parameters['ROI']*2,1):
                            continue
                        else:
                            ind=0
                            vector=pd.DataFrame(np.asarray([plate,well,site,x,y,n]).reshape(1,6),columns=['plate','well','site','coordX','coordY','cell_id'],index=[ind])
                            if self.parameters['csv_coordinates']=='':
                                tree = KDTree([row[:2] for row in center_of_mass])
                                # Query the nearest distance and the index of the nearest point
                                distance, _ = tree.query([x,y], k=2)    
                                vector['distance']=distance[1] 
                            else:
                                vector['distance']=locations.loc[(locations.coordX==x)&(locations.coordY==y),'distance'].values[0]

                            try:
                                scalefex = ScaleFEx_from_crop.compute_ScaleFEx.ScaleFEx(
                                    crop,
                                    channel=self.parameters['channel'],
                                    mito_ch=self.parameters['Mito_channel'],
                                    rna_ch=self.parameters['RNA_channel'],
                                    neuritis_ch=self.parameters['neurite_tracing'],
                                    downsampling=self.parameters['downsampling'],
                                    visualization=self.parameters['visualize_masks'],
                                    roi=int(self.parameters['ROI'])
                                ).single_cell_vector

                                if isinstance(scalefex, pd.DataFrame):
                                    vector = pd.concat([vector, scalefex], axis=1)
                                    csv_file = dq.save_csv_file(vector, csv_file, self.parameters['max_file_size'])
                            except Exception as e:
                                print("An error occurred during ScaleFEx computation:", e)
        
        function = compute_vector
        parallelize_computation.parallelize_local(wells,function,self.parameters['n_of_workers'],mode = 'dev')
            
        print('All processes have completed their tasks.')
        
        # dq.push_all_files(self.parameters['s3_bucket'],self.parameters['experiment_name'],
        #                                         self.parameters['plate'],self.parameters['subset_index'],self.vec_dir)
        
        # dq.terminate_current_instance(self.parameters['s3_bucket'],self.parameters['experiment_name'],
        #                                                     self.parameters['plate'],self.parameters['subset_index'])
    

    def segment_crop_images(self,img_nuc):

        # extraction of the location of the cells
        nls=import_module(self.parameters['segmenting_function'])
        
        if self.parameters['AI_cell_segmentation'] is False:    
            img_mask=nls.compute_DNA_mask(img_nuc)
            center_of_mass = nls.retrieve_coordinates(img_mask,
                        cell_size_min=self.parameters['min_cell_size']*self.parameters['downsampling'],
                        cell_size_max=self.parameters['max_cell_size']/self.parameters['downsampling'])
        else:
            if img_nuc.max() > 1:
                img_nuc = img_nuc/img_nuc.max()
            img_mask,center_of_mass = self.mrcnn.generate_masks(img_nuc,ds_size=(540,540),score_thresh=0.8,
                                                                min_area_thresh=self.parameters['min_cell_size'],
                                                                max_area_thresh=self.parameters['max_cell_size'],
                                                                ROI=self.parameters['ROI'],
                                                                remove_edges=False,try_quadrants=True)
            if center_of_mass is None:
                center_of_mass = []
        if self.parameters['visualization'] is True:
            self.show_image(img_nuc,img_mask)

        try:
            center_of_mass
            print('N of cells found: ',len(center_of_mass))
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
