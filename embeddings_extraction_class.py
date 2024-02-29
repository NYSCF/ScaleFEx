import data_query
import yaml,os




class compute_embeddings:

    def __init__(self,yaml_path='parameters.yaml'):

        with open(yaml_path) as f:
            self.parameters=yaml.load(f.read(),Loader=yaml.CLoader)
        f.close()


        if self.parameters['AWS']['use_AWS'] in['no','N','NO','n','Nope']:
            print('Local computation')
            computation='local'
            
        
        elif self.parameters['AWS']['use_AWS'] in ['yes','Y','YES','yes','y']:
            computation='AWS'
            print(' AWS computation')

        self.data_retrieve=import_module(self.parameters['local']['query_function'])   
        self.data_retrieve=import_module(self.parameters['local']['query_function'])   

        else:
            print(self.parameters['AWS']['use_AWS'], ' is not an accepted character. Please specify Yes or No')

        self.data_retrieve=import_module(self.parameters[computation]['query_function'])

        print("retrieving files from ", (self.parameters['location_parameters']['exp_folder']))
        files=self.data_retrieve.query_data(self.parameters['location_parameters']['exp_folder'])

        if self.parameters.FFC:
            if not os.path.exists(self.parameters['location_parameters']['saving_folder']+self.parameters['location_parameters']['experiment_name']+'_FFC.p'):
                print('Flat Field correction image not found in ' + self.parameters['location_parameters']['saving_folder'],
                      ' Generating FFC now')
                self.flat_field_correction=data_query.

        for plate in self.parameters['location_parameters']['Plates']:
            print(plate)
            self.cascade_functions(plate,files.loc[files.plate==plate])
        





    def cascade_functions(self,plate,files):
        
        
        print(files)
        
        # Wells,fields=utils.make_well_and_field_list(files)
        # csv_file=self.saving_folder+'Embedds/'+self.experiment_name+'_'+str(plate)+'Embeddings.csv'
        # if self.QC==True:
        #     csv_file_QC=self.saving_folder+'/QC_Results/'+self.experiment_name+'_'+str(plate)+'.csv'
        # else:
        #     csv_file_QC=None
        # flag,ind1,indSS,indQC,Wells,Site_ex,flag2=utils.check_if_file_exists(csv_file,csv_file_QC,Wells,fields[-1])
        # #self.calculate_QC(Parameters,files,csv_file,plate,Wells,fields,flag,ind)
        # self.embedding_extraction_Phenix(files,plate,Wells,Site_ex,flag2,fields,csv_file,csv_file_QC,flag,ind1,indSS,indQC)



def import_module(module_name):
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        print(f"Module '{module_name}' not found.")
        return None
    


