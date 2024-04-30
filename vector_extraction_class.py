import yaml,os,pickle
import parallelize_computation
import pandas as pd
import numpy as np
from datetime import datetime
import Quality_control_HCI.compute_global_values
import Embeddings_extraction_from_image.batch_compute_embeddings
import data_query.query_functions_local
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

        self.saving_folder = self.parameters['saving_folder']

        files = data_query.query_functions_local.query_data(self.parameters['exp_folder'], plate_type = self.parameters['plate_type'],
                                            pattern=self.parameters['fname_pattern'],delimiters = self.parameters['fname_delimiters'],
                                            exts=self.parameters['file_extensions'],resource = self.parameters['resource'], 
                                            experiment_name = self.parameters['experiment_name'],plates=self.parameters['plates'], 
                                            s3_bucket = self.parameters['s3_bucket'])
        print(files.head())
        
        # Perform Flat Field Correction (FFC)
        self.flat_field_correction = {}
        if self.parameters['FFC'] is True:
            ffc_file = os.path.join(self.saving_folder,self.parameters['experiment_name'] + '_FFC.p')
            if not os.path.exists(ffc_file):
                print(ffc_file + ' Not found generating FFC now')

                if self.parameters['resource'] == 'AWS':
                    self.flat_field_correction = data_query.query_functions_local.flat_field_correction_AWS(files,ffc_file,
                    self.parameters['s3_bucket'],self.parameters['channel'],self.parameters['experiment_name'],bf_channel=self.parameters['bf_channel'],
                    n_images=self.parameters['FFC_n_images'])
                    
                else :
                    self.flat_field_correction = data_query.query_functions_local.flat_field_correction_on_data(
                    files, self.parameters['channel'], bf_channel=self.parameters['bf_channel'],n_images=self.parameters['FFC_n_images'])

                    pickle.dump(self.flat_field_correction, open(ffc_file, "wb"))
            else:
                print(ffc_file + ' Found, loading FFC')
                self.flat_field_correction = pickle.load(open(ffc_file, "rb"))
        else:
            for channel in self.parameters['channel']:
                self.flat_field_correction[channel] = 1
        # Initialize segmentation weights if using AI models
        if self.parameters['AI_cell_segmentation'] is True:
            if self.parameters['segmenting_function'] == 'MaskRCNN_Deployment.segmentation_mrcnn':
                from MaskRCNN_Deployment.segmentation_mrcnn import MaskRCNN
                mrcnn_weights = os.path.join(ROOT_DIR,'MaskRCNN_Deployment/weights/maskrcnn_weights.pt')
                mrcnn_weights = '/home/biancamigliori/Documents/random_projection/maskrcnn_weights.pt'
                self.mrcnn = MaskRCNN(weights=mrcnn_weights,use_cpu=self.parameters['use_cpu_segmentation'],gpu_id=self.parameters['gpu_AI'])
            elif self.parameters['segmenting_function'] == 'ADDIEs':
                print("For Addie to implement")


        # Loop over plates and start computation
        if self.parameters['plates'] != 'all' and isinstance(self.parameters['plates'],list):
            plate_list = np.unique(files.plate)
        else:
            plate_list = self.parameters['plates']

        print('Computing plates: ', plate_list)
        
        if 'mbed' in self.parameters['vector_type']:
            import Embeddings_extraction_from_image.inception_set
            
            self.parser = Embeddings_extraction_from_image.inception_set.place_holder(channel=self.parameters['channel'],
                                                                                      device=self.parameters['device'],weights=self.parameters['weights_location'])
            
        for plate in plate_list:
            self.start_computation(plate, files)

### Start computation
            
    def start_computation(self,plate,files):
        
        task_files=files.loc[files.plate==plate]
        wells, task_fields = data_query.query_functions_local.make_well_and_field_list(task_files,self.parameters['resource']
                                                                                       ,self.parameters['subset'])
        vec_dir = os.path.join(self.saving_folder,self.parameters['vector_type'])
        if not os.path.exists(vec_dir):
            os.makedirs(vec_dir)
        csv_file = os.path.join(vec_dir,self.parameters['experiment_name']+'_'+str(plate)+'_'+self.parameters['vector_type']+'.csv')
        # QC
        if self.parameters['QC']==True:
            qc_dir = os.path.join(self.saving_folder,'QC_analysis')
            csv_fileQC = os.path.join(qc_dir,self.parameters['experiment_name']+'_'+str(plate)+'QC.csv')
            if not os.path.exists(qc_dir):
                os.makedirs(qc_dir)
        if self.parameters['save_coordinates'] == True:
            csv_file_coordinates = os.path.join(self.saving_folder,self.parameters['experiment_name'] + '_coordinates_'+str(plate)+'.csv')

        if self.parameters['csv_coordinates'] == '':
            wells=data_query.query_functions_local.check_if_file_exists(csv_file,wells,task_fields)
            if wells[0] == 'Over':
                print('plate ', plate, 'is done')
                return
        else:
            self.locations=data_query.query_functions_local.check_if_file_exists(csv_file,wells,task_fields,
                                                                                 self.parameters['csv_coordinates'],plate=plate)
            wells=np.unique(self.locations.well)
            
        def compute_vector(well):
            ''' Function that imports the images and extracts the location of cells'''

            print(well, plate, datetime.now())
            for site in task_fields:
    
                print(site, well, plate, datetime.now())
                #stime=time.perf_counter()
                np_images, original_images = data_query.query_functions_local.load_and_preprocess(task_files,
                                    self.parameters['channel'],well,site,self.parameters['zstack'],self.parameters['resource'],
                                    self.parameters['image_size'],self.flat_field_correction,
                                    self.parameters['downsampling'],return_original=self.parameters['QC'],
                                    s3_bucket = self.parameters['s3_bucket'])
                try:
                    original_images.shape
                except NameError:
                    print('Images corrupted')
                #print('images load and process time ',time.perf_counter()-stime)
                if np_images is not None:
                    # stime = time.perf_counter()
                    if self.parameters['csv_coordinates']=='':
                        center_of_mass=self.segment_crop_images(np_images[0,:,:,0])
                        center_of_mass=[list(row) + [n] for n,row in enumerate(center_of_mass)]
                        if self.parameters['compute_live_cells'] is False:
                            live_cells=len(center_of_mass)
                        else:
                            print('to be implemented')
                    else:
                        
                        locations=self.locations
                        locations=locations.loc[(locations.well==well)&(locations.site==site)&(locations.plate.astype(str)==str(plate))]
                        center_of_mass=np.asarray(locations[['coordX','coordY','cell_id']])
                        
                        if self.parameters['compute_live_cells'] is False:
                            live_cells=len(center_of_mass)
                            
                    # stime = time.perf_counter()
                    if self.parameters['QC']==True:
                        indQC=0

                        QC_vector,indQC = Quality_control_HCI.compute_global_values.calculateQC(len(center_of_mass),live_cells,
                                            self.parameters['vector_type'],original_images,well,plate,site,self.parameters['channel'],
                                            indQC,self.parameters['neurite_tracing'])
                        if not os.path.exists(csv_fileQC):
                            QC_vector.to_csv(csv_fileQC,header=True)
                        else:
                            QC_vector.to_csv(csv_fileQC,mode='a',header=False)
                    # print('QC time ',time.perf_counter()-stime)
                    # if self.parameters['vector_type'] == 'embeddings':



                    if self.parameters['tile_computation'] is True:
                        ind=0
                        vector=pd.DataFrame(np.asarray([plate,well,site]).reshape(1,3),columns=['plate','well','site'],index=[ind])
                        vector=pd.concat([vector,Embeddings_extraction_from_image.batch_compute_embeddings.Compute_embeddings(self.parser,np_images,ind,self.parameters['channel'],
                                                                                            ).embeddings],axis=1)
                        tile_csv = csv_file[:-4]+'Tile.csv'
                        if not os.path.exists(tile_csv):
                            vector.to_csv(tile_csv,header=True)
                        else:
                            vector.to_csv(tile_csv,mode='a',header=False)
              
                    for x,y,n in center_of_mass:
                       
                        # stime = time.perf_counter()
                        crop=np_images[:,int(float(x)-self.parameters['ROI']):int(float(x)+self.parameters['ROI']),
                                           int(float(y)-self.parameters['ROI']):int(float(y)+self.parameters['ROI']),:]
                        # if ((x-self.parameters['ROI']<0) or (x-self.parameters['ROI']>self.parameters['image_size'][0]) or
                        #     (y-self.parameters['ROI']<0) or (y-self.parameters['ROI']>self.parameters['image_size'][1])):
                        if crop.shape != (len(self.parameters['channel']),self.parameters['ROI']*2,self.parameters['ROI']*2,1):
                            print(crop.shape, "cell on the border")
                            continue
                        else:
                            if self.parameters['visualize_crops']==True:
                                plt.imshow(crop[0])
                                plt.show()
                            ind=0
                            vector=pd.DataFrame(np.asarray([plate,well,site,x,y,n]).reshape(1,6),columns=['plate','well','site','coordX','coordY','cell_id'],index=[ind])
                            if self.parameters['csv_coordinates']=='':
                                tree = KDTree([row[:2] for row in center_of_mass])

                                # Query the nearest distance and the index of the nearest point
                                distance, _ = tree.query([x,y], k=2)    
                                vector['distance']=distance[1] 
                            else:
                                vector['distance']=locations.loc[(locations.coordX==x)&(locations.coordY==y),'distance'].values[0]
                            
                            if self.parameters['save_coordinates']==True:
                                if not os.path.exists(csv_file_coordinates):
                                    vector.to_csv(csv_file_coordinates,header=True)
                                else:
                                    vector.to_csv(csv_file_coordinates,mode='a',header=False)

                            if 'mbed' in self.parameters['vector_type']:

                                vector=pd.concat([vector,Embeddings_extraction_from_image.batch_compute_embeddings.Compute_embeddings(self.parser,
                                                                        crop,0,self.parameters['channel']).embeddings],axis=1)
                                
                                if not os.path.exists(csv_file):
                                    vector.to_csv(csv_file,header=True)
                                else:
                                    vector.to_csv(csv_file,mode='a',header=False)
                                #print('embedding_computation time ',time.perf_counter()-stime)
                            
                            elif ('cal' in self.parameters['vector_type']) :                                
                                scalefex=ScaleFEx_from_crop.compute_ScaleFEx.ScaleFEx(crop, channel=self.parameters['channel'],
                                                    mito_ch=self.parameters['Mito_channel'], rna_ch=self.parameters['RNA_channel'],
                                                    neuritis_ch=self.parameters['neurite_tracing'],downsampling=self.parameters['downsampling'],
                                                    visualization=self.parameters['visualize_masks'], roi=int(self.parameters['ROI'])
                                                    ).single_cell_vector
                                if isinstance(scalefex, pd.DataFrame):
                                    vector=pd.concat([vector,scalefex],axis=1)
                                    if not os.path.exists(csv_file):
                                        vector.to_csv(csv_file,header=True)
                                    else:
                                        vector.to_csv(csv_file,mode='a',header=False)
                                    # print('vector_computatiomn time ',time.perf_counter()-stime)
                            else:
                                print('Not a valid vector type entry')
                            
        if self.parameters['n_of_workers'] != 1:
            function = compute_vector
            parallelize_computation.parallelize_local(wells,function,self.parameters['n_of_workers'],mode = 'dev')
        else:
            for well in wells:
                stime=time.perf_counter()
                compute_vector(well)
                print('well time ',time.perf_counter()-stime)

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
