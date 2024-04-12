import yaml,os,pickle
import parallelize_computation
import pandas as pd
import numpy as np
from datetime import datetime
import Quality_control_HCI.compute_global_values
import Embeddings_extraction_from_image.batch_compute_embeddings
import ScaleFEx_from_crop.compute_ScaleFEx
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

        # Set frequently used parameters as attributes
        self.SAVE_DIR = self.parameters['location_parameters']['saving_folder']
        self.EXP_NAME = self.parameters['location_parameters']['experiment_name']
        self.VEC_TYPE = self.parameters['vector_type']
        self.loc_params = self.parameters['location_parameters']
        self.seg_params = self.parameters['segmentation']
        self.visualization = self.parameters['segmentation']['visualization']

        # Determine the type of computation to be used
        self.computation = 'local' if self.parameters['AWS']['use_AWS'] in [
            'no', 'N', 'NO', 'n', 'Nope'] else 'AWS'

        # Import the data retrieval function
        self.data_retrieve = import_module(self.parameters[self.computation]['query_function'])

        # Print the experiment folder
        print("retrieving files from ", (self.loc_params['exp_folder']))

        # Get the files
        files = self.data_retrieve.query_data(self.loc_params['exp_folder'],plate_type= self.loc_params['plate_type'],
                                                      pattern=self.loc_params['fname_pattern'],delimiters=self.loc_params['fname_delimiters'],
                                                      exts=self.loc_params['file_extensions'])
        # files = self.data_retrieve.query_data_outdated(self.loc_params['exp_folder'],plate_type= self.loc_params['plate_type'])
        print(files.head())
        
        # Perform Flat Field Correction (FFC)
        self.flat_field_correction = {}
        if self.parameters['FFC'] is True:
            ffc_file = os.path.join(self.SAVE_DIR,self.EXP_NAME + '_FFC.p')
            if not os.path.exists(ffc_file):
                print('Flat Field correction image not found in ' +
                      self.SAVE_DIR,
                      ' Generating FFC now')
                
                self.flat_field_correction = self.data_retrieve.flat_field_correction_on_data(
                    files, self.parameters['type_specific']['channel'], n_images=self.parameters['FFC_n_images'])
                
                pickle.dump(self.flat_field_correction, open(ffc_file, "wb"))

            else:
                print('Flat Field correction image found in ' +
                      self.SAVE_DIR,' Loading FFC')
                self.flat_field_correction = pickle.load(open(ffc_file, "rb"))
        else:
            for channel in self.parameters['type_specific']['channel']:
                self.flat_field_correction[channel] = 1
        if self.seg_params['MaskRCNN_cell_segmentation'] is True:
            from MaskRCNN_Deployment.segmentation_mrcnn import MaskRCNN
            mrcnn_weights = os.path.join(ROOT_DIR,'MaskRCNN_Deployment/weights/maskrcnn_weights.pt')
            self.mrcnn = MaskRCNN(weights=mrcnn_weights,use_cpu=False,gpu_id=self.seg_params['gpu_mrcnn'])

        # Loop over plates and start computation
        if self.loc_params['plates'][0] == 'all':
            plate_list = np.unique(files.plate)
        else:
            plate_list = self.loc_params['plates']
        print('Computing plates: ', plate_list)
        for plate in plate_list:
            self.start_computation(plate, files)

### Start computation
            
    def start_computation(self,plate,files):
        
        # self.plate=plate
        
        task_files=files.loc[files.plate==plate]
        wells, task_fields = self.data_retrieve.make_well_and_field_list(task_files)
        vec_dir = os.path.join(self.SAVE_DIR,self.VEC_TYPE)
        if not os.path.exists(vec_dir):
            os.makedirs(vec_dir)
        csv_file = os.path.join(vec_dir,self.EXP_NAME+'_'+str(plate)+'_'+self.VEC_TYPE+'.csv')

        # Create QC directory
        if self.parameters['QC']==True:
            qc_dir = os.path.join(self.SAVE_DIR,'QC_analysis')
            csv_fileQC = os.path.join(qc_dir,self.EXP_NAME+'_'+str(plate)+'QC.csv')
            if not os.path.exists(qc_dir):
                os.makedirs(qc_dir)

        if self.seg_params['csv_coordinates'] == '':
            wells=self.data_retrieve.check_if_file_exists(csv_file,wells,task_fields)
            if wells[0] == 'Over':
                print('plate ', plate, 'is done')
                return
        else:
            self.locations=self.data_retrieve.check_if_file_exists(csv_file,wells,task_fields,self.seg_params['csv_coordinates'],plate=plate)
            wells=np.unique(self.locations.well)
            
        def compute_vector(well):
            ''' Function that imports the images and extracts the location of cells'''

            
            print(well, plate, datetime.now())
            for site in task_fields:
    
                print(site, well, plate, datetime.now())
                #stime=time.perf_counter()
                np_images, original_images = self.data_retrieve.load_and_preprocess(task_files,
                                    self.parameters['type_specific']['channel'],well,site,self.parameters['type_specific']['zstack'],self.data_retrieve,
                                    self.parameters['type_specific']['img_size'],self.flat_field_correction,
                                    self.parameters['downsampling'],return_original=self.parameters['QC'])
                try:
                    original_images.shape
                except NameError:
                    print('Images corrupted')
                #print('images load and process time ',time.perf_counter()-stime)

                if np_images is not None:
                    # stime = time.perf_counter()
                    if self.seg_params['csv_coordinates']=='':
                        center_of_mass=self.segment_crop_images(np_images[0,:,:,0])
                        center_of_mass=[list(row) + [n] for n,row in enumerate(center_of_mass)]
                        if self.parameters['type_specific']['compute_live_cells'] is False:
                            live_cells=len(center_of_mass)
                        else:
                            print('to be implemented')
                    else:
                        
                        locations=self.locations
                        locations=locations.loc[(locations.well==well)&(locations.site==site)&(locations.plate.astype(str)==str(plate))]
                        center_of_mass=np.asarray(locations[['coordX','coordY','cell_id']])
                        
                        if self.parameters['type_specific']['compute_live_cells'] is False:
                            live_cells=len(center_of_mass)
                            
                    # print('coordinates time ',time.perf_counter()-stime)    
                        
                    #print(center_of_mass)
                    # stime = time.perf_counter()
                    if self.parameters['QC']==True:
                        indQC=0

                        QC_vector,indQC = Quality_control_HCI.compute_global_values.calculateQC(len(center_of_mass),live_cells,
                                            self.EXP_NAME,original_images,well,plate,site,self.parameters['type_specific']['channel'],
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
                        tile_csv = csv_file[:-4]+'Tile.csv'
                        if not os.path.exists(tile_csv):
                            vector.to_csv(tile_csv,header=True)
                        else:
                            vector.to_csv(tile_csv,mode='a',header=False)
              
                    for x,y,n in center_of_mass:
                       
                        # stime = time.perf_counter()
                        crop=np_images[:,int(float(x)-self.parameters['type_specific']['ROI']):int(float(x)+self.parameters['type_specific']['ROI']),
                                           int(float(y)-self.parameters['type_specific']['ROI']):int(float(y)+self.parameters['type_specific']['ROI']),:]
                        # if ((x-self.parameters['type_specific']['ROI']<0) or (x-self.parameters['type_specific']['ROI']>self.loc_params['image_size'][0]) or
                        #     (y-self.parameters['type_specific']['ROI']<0) or (y-self.parameters['type_specific']['ROI']>self.loc_params['image_size'][1])):
                        if crop.shape != (len(self.parameters['type_specific']['channel']),self.parameters['type_specific']['ROI']*2,self.parameters['type_specific']['ROI']*2,1):
                            print(crop.shape, "cell on the border")
                            continue
                        else:
                            if self.parameters['visualize_crops']==True:
                                plt.imshow(crop[0])
                                plt.show()

                            ind=0
                            vector=pd.DataFrame(np.asarray([plate,well,site,x,y,n]).reshape(1,6),columns=['plate','well','site','coordX','coordY','cell_id'],index=[ind])

                        
                            if self.seg_params['csv_coordinates']=='':
                             
                                tree = KDTree([row[:2] for row in center_of_mass])

                                # Query the nearest distance and the index of the nearest point
                                distance, _ = tree.query([x,y], k=2)    
                                vector['distance']=distance[1] 
                            else:
                                vector['distance']=locations.loc[(locations.coordX==x)&(locations.coordY==y),'distance'].values[0]

                            
                            # print(crop.shape)

                            if 'mbed' in self.VEC_TYPE:

                                vector=pd.concat([vector,Embeddings_extraction_from_image.batch_compute_embeddings.Compute_embeddings(crop,0,self.parameters['type_specific']['channel'],
                                                                                                self.parameters["device"],weights=self.parameters['weights_location']).embeddings],axis=1)
                                
                                if not os.path.exists(csv_file):
                                    vector.to_csv(csv_file,header=True)
                                else:
                                    vector.to_csv(csv_file,mode='a',header=False)
                                print('embedding_computation time ',time.perf_counter()-stime)
                            
                            elif ('cale' in self.VEC_TYPE) or ('FEx' in self.VEC_TYPE) or ('fex' in self.VEC_TYPE):
                                scalefex=ScaleFEx_from_crop.compute_ScaleFEx.ScaleFEx(crop, channel=self.parameters['type_specific']['channel'],
                                                    mito_ch=self.parameters['type_specific']['Mito_channel'], rna_ch=self.parameters['type_specific']['RNA_channel'],
                                                    neuritis_ch=self.parameters['type_specific']['neurite_tracing'],downsampling=self.parameters['downsampling'],
                                                    visualization=self.parameters['visualize_masks'], roi=int(self.parameters['type_specific']['ROI'])
                                                    ).single_cell_vector
                                if isinstance(scalefex, pd.DataFrame):
                                    vector=pd.concat([vector,scalefex],axis=1)
                                    #print(vector)
                                    if not os.path.exists(csv_file):
                                        vector.to_csv(csv_file,header=True)
                                    else:
                                        vector.to_csv(csv_file,mode='a',header=False)

                                    # print('vector_computatiomn time ',time.perf_counter()-stime)

                            else:
                                print('Not a valid vector type entry')
                            

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
    
        nls=import_module(self.seg_params['segmenting_function'])
        
        if self.seg_params['MaskRCNN_cell_segmentation'] is False:    
            img_mask=nls.compute_DNA_mask(img_nuc)
            center_of_mass = nls.retrieve_coordinates(img_mask,
                        cell_size_min=self.seg_params['min_cell_size']*self.parameters['downsampling'],
                        cell_size_max=self.seg_params['max_cell_size']/self.parameters['downsampling'])
        else:
            if img_nuc.max() > 1:
                img_nuc = img_nuc/img_nuc.max()
            img_mask,center_of_mass = self.mrcnn.generate_masks(img_nuc,ds_size=(540,540),score_thresh=0.8,
                                                                area_thresh=self.seg_params['min_cell_size'],
                                                                ROI=self.parameters['type_specific']['ROI'],
                                                                remove_edges=False,try_quadrants=True)
            if center_of_mass is None:
                center_of_mass = []
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
