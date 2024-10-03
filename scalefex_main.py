import yaml,os,pickle
import scalefex_utils
import pandas as pd
import numpy as np
from datetime import datetime
import data_query.query_functions_local
import time
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import ScaleFEx_from_crop.compute_ScaleFEx
import shutil
import argparse
import cv2 as cv
# ignore performance warnings about dataframe fragmentation
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
ROOT_DIR = '/'.join(__file__.split('/')[:-1])

class Process_HighContentImaging_screen: 
    """
    Class designed for processing high-content screening data, including feature extraction and quality control.

    Args:
        yaml_path (str): Path to the YAML configuration file containing processing parameters.


    """
    def __init__(self, yaml_path='parameters.yaml'):
        """
        Initializes the Screen_Compute object with parameters from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file containing parameters. Default is 'parameters.yaml'.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file '{yaml_path}' not found.")
        self.yaml_path = yaml_path
        # Read the yaml file
        with open(yaml_path, 'rb') as f:
            self.parameters = yaml.load(f.read(), Loader=yaml.CLoader)
        self.PARAMS_VALID = scalefex_utils.check_YAML_parameter_validity(yaml_path)
        self.saving_folder = self.parameters['saving_folder']
        self.csv_file = self.parameters['csv_coordinates']
        self.sites_computed_file = os.path.join(self.saving_folder,self.parameters['experiment_name'] + '_sites-computed.csv')

    def run(self):

        start_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        
        
        files = data_query.query_functions_local.query_data(self.parameters['exp_folder'], self.parameters['pattern'],plate_identifiers=self.parameters['plate_identifiers'],
                                                            exts=self.parameters['exts'], plates=self.parameters['plates'],)
        if self.parameters['overwrite'] or not os.path.exists(self.sites_computed_file):
            pd.DataFrame(columns=['plate','well','site','subset','file_path','total_count','computed_count','on_edge_count','fail_count',
                                  'computed_ids','on_edge_ids','fail_ids']).to_csv(self.sites_computed_file,index=False)
        
        # Perform Flat Field Correction (FFC)
        self.flat_field_correction = {}
        if self.parameters['FFC'] is True:
            ffc_file = os.path.join(self.saving_folder,self.parameters['experiment_name'] + '_FFC.p')
            if not os.path.exists(ffc_file):
                print(ffc_file + ' Not found generating FFC now') 
                self.flat_field_correction = data_query.query_functions_local.flat_field_correction_on_data(
                files, self.parameters['channel'],n_images=self.parameters['FFC_n_images'])
                pickle.dump(self.flat_field_correction, open(ffc_file, "wb"))
            else:
                print(ffc_file + ' Found, loading FFC')
                self.flat_field_correction = pickle.load(open(ffc_file, "rb"))
            self.parameters['image_size'] = self.flat_field_correction[self.parameters['channel'][0]].shape
        else:
            for channel in self.parameters['channel']:
                self.flat_field_correction[channel] = 1
            image_size = cv.imread(files.iloc[0]['file_path'],cv.IMREAD_GRAYSCALE).shape
            self.parameters['image_size'] = image_size

        # Loop over plates and start computation

        plate_list = sorted(files.plate.unique().tolist())
        print('Computing plates: ', plate_list)
        
        vec_dir = os.path.join(self.saving_folder,self.parameters['vector_type'])

        if not os.path.exists(vec_dir):
            os.makedirs(vec_dir)

        # save time-stamped parameters file
        shutil.copy2(self.yaml_path,os.path.join(self.saving_folder,self.parameters['experiment_name'] + f'_{start_time}_parameters.yaml'))

        # check which sites haven't computed (if continuing from a previous run)
        if not self.parameters['overwrite'] and os.path.exists(self.sites_computed_file):
            computed_df = pd.read_csv(self.sites_computed_file,converters={'plate':str,'well':str,'site':str,
                                                                               'computed_ids':str,'on_edge_ids':str,'fail_ids':str})
            if len(computed_df)>0:
                files_minus_computed = files.merge(computed_df[['plate','well','site']],on=['plate','well','site'],how='outer',indicator=True)
                files = files_minus_computed[files_minus_computed['_merge']=='left_only'].reset_index(drop=True)
        if len(files) == 0:
            print('All files already processed, set "overwrite: True" to overwrite')
            print('Terminating ScaleFEx...')
            return
        plates_finished = 0

        for plate in plate_list:
             # QC
            if self.parameters['QC']==True:
                qc_dir = os.path.join(self.saving_folder,'QC_analysis')
                self.csv_fileQC = os.path.join(qc_dir,self.parameters['experiment_name']+'_'+str(plate)+'QC.csv')
                if not os.path.exists(qc_dir):
                    os.makedirs(qc_dir)
            plate_coords_csv = os.path.join(self.saving_folder,self.parameters['experiment_name'] + '_coordinates_'+str(plate)+'.csv')
            # delete previous coords files if overwriting
            if self.parameters['overwrite'] and os.path.exists(plate_coords_csv):
                os.remove(plate_coords_csv)

            if self.parameters['save_coordinates'] == True:
                self.csv_file_coordinates = plate_coords_csv

            is_finished = self.start_computation(plate, files)

            if is_finished == True:
                plates_finished += 1
            

        if plates_finished == len(plate_list) and plates_finished>0:
            if self.parameters['save_coordinates'] == True:
                # concatenate coordinates files into time-stamped file
                coordinate_csvs = [os.path.join(self.saving_folder,self.parameters['experiment_name'] + '_coordinates_'+str(plate)+'.csv') for plate in plate_list
                                   if os.path.exists(os.path.join(self.saving_folder,self.parameters['experiment_name'] + '_coordinates_'+str(plate)+'.csv'))]
                if len(coordinate_csvs) == 0:
                    print('No coordinates to concatenate')
                    return
                all_coords = pd.concat([pd.read_csv(file) for file in coordinate_csvs],ignore_index=True).reset_index(drop=True)
                write_mode = 'w'
                if not self.parameters['overwrite']:
                    write_mode = 'a'
                all_coords.to_csv(os.path.join(self.saving_folder,self.parameters['experiment_name'] + '_coordinates.csv'),index=False,mode=write_mode)
                # remove plate specific coordinate files
                _ = [os.remove(file) for file in coordinate_csvs]
            
            

### Start computation
            
    def start_computation(self,plate,files):
        if self.parameters['QC']==True:
            import Quality_control_HCI.compute_global_values
        task_files=files.loc[files.plate==plate]
        vec_dir = os.path.join(self.saving_folder,self.parameters['vector_type'])
        if not os.path.exists(vec_dir):
            os.makedirs(vec_dir)
        
        sites_computed_df = pd.read_csv(self.sites_computed_file,converters={'plate':str,'well':str,'site':str,
                                                                               'computed_ids':str,'on_edge_ids':str,'fail_ids':str})
        wells, sites = data_query.query_functions_local.make_well_and_field_list(task_files)

        if os.path.exists(self.parameters['csv_coordinates']):
            self.locations=pd.read_csv(self.parameters['csv_coordinates'])
            self.locations=self.locations.loc[self.locations.plate.astype(str)==str(plate)]
            wells=np.unique(self.locations.well)
        self.csv_file = ''
        if self.parameters['csv_coordinates'] == '':
            self.csv_file,wells=data_query.query_functions_local.check_if_file_exists(self.csv_file,wells,sites)
        if wells[0] == 'Over':
            print('plate ', plate, 'is done')
            return

        def compute_vector(well):
            ''' Function that imports the images and extracts the location of cells'''
            print(well, plate, datetime.now())
            sites = sorted(task_files[task_files['well']==well]['site'].unique().tolist())
            self.csv_file = os.path.join(vec_dir,self.parameters['experiment_name']+'_'+str(plate)+'_'+self.parameters['vector_type']+'.csv')
            
            for site in sites:

                print(site, well, plate, datetime.now())

                np_images, original_images, current_file = data_query.query_functions_local.load_and_preprocess(task_files,
                                    self.parameters['channel'],well,site,self.parameters['zstack'],
                                    self.parameters['image_size'],self.flat_field_correction,
                                    self.parameters['downsampling'],return_original=self.parameters['QC'])
                try:
                    original_images.shape
                except NameError:
                    print('Images corrupted')

                if np_images is not None:

                    if self.parameters['csv_coordinates']=='':
                        center_of_mass=self.segment_crop_images(original_images[0,:,:])
                        center_of_mass=np.array([list(row/self.parameters['downsampling']) + [n] for n,row in enumerate(center_of_mass)])
                      
                    else:
                        
                        locations=self.locations
                        locations=locations.loc[(locations.well==well)&(locations.site==site)&(locations.plate.astype(str)==str(plate))]
                        center_of_mass=np.asarray(locations[['coordX','coordY','cell_id']])


                    if self.parameters['QC']==True:
                        indQC=0

                        QC_vector,indQC = Quality_control_HCI.compute_global_values.calculateQC(tot_n=len(center_of_mass),
                                            experiment_name=self.parameters['vector_type'],img_raw=original_images,
                                            well=well,plate=plate,site=site,channel=self.parameters['channel'],
                                            indQC=indQC)
                        QC_vector['file_path'] = current_file
                        self.csv_fileQC = self.save_csv_file(QC_vector,self.csv_fileQC)

                    is_computed = (np.ones(len(center_of_mass))*-1).astype(int)
                    for index,(x,y,n) in enumerate(center_of_mass):
       
                        crop=np_images[:,int(float(x)-self.parameters['ROI']):int(float(x)+self.parameters['ROI']),
                                           int(float(y)-self.parameters['ROI']):int(float(y)+self.parameters['ROI']),:]
                
                        if crop.shape != (len(self.parameters['channel']),self.parameters['ROI']*2,self.parameters['ROI']*2,1):
                            print(crop.shape, "cell on the border")
                            is_computed[index] = 0
                            continue
                        else:
                            if self.parameters['visualize_crops']==True:
                                _,axes = plt.subplots(nrows=1,ncols=len(crop),figsize=(len(crop)*3,3))
                                for i,ax in enumerate(axes.flat):
                                    ax.imshow(crop[i],cmap='gray')
                                    ax.axis('off')
                                    ax.set_title(self.parameters['channel'][i])
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
                                self.csv_file_coordinates = self.save_csv_file(vector,self.csv_file_coordinates)

                            if 'scal' in self.parameters['vector_type']:
                                try:
                                    scalefex = ScaleFEx_from_crop.compute_ScaleFEx.ScaleFEx(
                                        crop,
                                        channel=self.parameters['channel'],
                                        mito_ch=self.parameters['Mito_channel'],
                                        rna_ch=self.parameters['RNA_channel'],
                                        downsampling=self.parameters['downsampling'],
                                        visualization=self.parameters['visualize_masks'],
                                        roi=int(self.parameters['ROI'])
                                    ).single_cell_vector

                                    if isinstance(scalefex, pd.DataFrame):
                                        vector = pd.concat([vector, scalefex], axis=1)
                                        vector['file_path'] = current_file
                                        vector['ROI_size'] = self.parameters['ROI']
                                        vector['channel_order'] = str(self.parameters['channel'])
                                        vector['downsampling'] = self.parameters['downsampling']
                                        self.csv_file = self.save_csv_file(vector, self.csv_file)
                                        # if self.parameters['write_computed_sites']:
                                        is_computed[index] = 1

                                except Exception as e:
                                    print("An error occurred during ScaleFEx computation:", e)
                            else:
                                print('Not a valid vector type entry')
                    
                    # tracking cells computed/skipped/failed
                    if len(center_of_mass) ==0:
                        center_of_mass = np.array([]).reshape((0,3))
                    computed_ids = tuple(np.asarray(center_of_mass)[np.nonzero(is_computed==1)[0],2].astype(int))
                    on_edge_ids = tuple(np.asarray(center_of_mass)[np.nonzero(is_computed==0)[0],2].astype(int))
                    failed_ids = tuple(np.asarray(center_of_mass)[np.nonzero(is_computed==-1)[0],2].astype(int))
                    file_path = files[(files['plate']==plate)&(files['well']==well)&(files['site']==site)&
                                    (files['channel']==self.parameters['channel'][0])]['file_path'].iloc[0]
                    compute_vec = [[plate,well,site,0,file_path,
                                    len(center_of_mass),np.count_nonzero(is_computed==1),np.count_nonzero(is_computed==0),
                                    np.count_nonzero(is_computed==-1),str(computed_ids),str(on_edge_ids),str(failed_ids)]]
                    site_row = pd.DataFrame(data=compute_vec,columns=sites_computed_df.columns)
                    
                    site_row.to_csv(self.sites_computed_file,mode='a',header=False,index=False)

        if self.parameters['n_of_workers'] != 1:
            function = compute_vector
            scalefex_utils.parallelize(wells,function,self.parameters['n_of_workers'],mode = 'prod')
        else:
            for well in wells:
                stime=time.perf_counter()
                compute_vector(well)
                print('well time ',time.perf_counter()-stime)
            
        print('All processes have completed their tasks.')
        return True
    def save_csv_file(self,vector,csv_file):
        '''
        Save the vector in a csv file'''
        if not os.path.exists(csv_file):
            vector.to_csv(csv_file,header=True,index=False)
        else:
            if os.stat(csv_file).st_size < self.parameters['max_file_size']*10**6:
                vector.to_csv(csv_file,mode='a',header=False,index=False)
            else:
                try: 
                    int(csv_file[-6:-4])

                    csv_file=csv_file[:-6]+str(int(csv_file[-6:-4])+1).zfill(2)+'.csv'
                except ValueError:
                    csv_file=csv_file[:-4]+str(1).zfill(2)+'.csv'
                
                vector.to_csv(csv_file,mode='a',header=True,index=False)
        return csv_file

    def segment_crop_images(self,img_nuc):
        '''
        Extracting coordinates of cells using segmenting function specified in YAML file
        '''
        # extraction of the location of the cells
        nls=scalefex_utils.import_module(self.parameters['segmenting_function'])
        
        
        img_mask=nls.compute_DNA_mask(img_nuc)
        center_of_mass = nls.retrieve_coordinates(img_mask,
                    cell_size_min=self.parameters['min_cell_size']*self.parameters['downsampling'],
                        cell_size_max=self.parameters['max_cell_size']/self.parameters['downsampling'])

        if self.parameters['visualization'] is True:
            self.show_image(img_nuc,img_mask)

        try:
            #center_of_mass
            print('N of cells found: ',len(center_of_mass))
        except NameError:
            center_of_mass = []
            print('No Cells detected')
        
        return center_of_mass

    def show_image(self,img,nuc):
        import matplotlib.pyplot as plt
        _,ax=plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(img.squeeze(),cmap='gray')
        ax[0].axis('off')
        ax[0].set_title('Image')
        if nuc is not None:
            ax[1].imshow(nuc.squeeze(),cmap='tab20')
            ax[1].set_title('Mask')
        ax[1].axis('off')
        plt.show()                    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parameters", type=str, default='parameters.yaml', 
                        required=False, help="Path to the parameters file")
    args = parser.parse_args()

    pipeline = Process_HighContentImaging_screen(yaml_path=args.parameters)
    # check parameters are valid
    if pipeline.PARAMS_VALID is False:
        run = scalefex_utils.query_yes_no("Some parameters are not valid. Do you want to continue?",default='no')
        if run is False:
            return False
    # check if files exist:
    if pipeline.parameters['overwrite'] is True:
        run = scalefex_utils.query_yes_no(f"You have set \"overwrite\" to True.\
                                          \nThis will overwrite existing files in the '{pipeline.parameters['experiment_name']}' directory.\
                                          \nAre you sure you want to continue?")
        if run is False:
            return False
    # run pipeline
    print('\n\nScaleFEx pipeline starting...')
    pipeline.run()

if __name__ == "__main__":
    main()  
    
