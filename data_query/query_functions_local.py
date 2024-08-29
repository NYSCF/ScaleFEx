''' Functions for data querying and handling in local machines'''
import glob, os, cv2 ,re
import numpy as np
import pandas as pd
import itertools


def pattern_to_regex(pattern, ext):
    '''
    Converts plaintext pattern to glob search path
    Helper function for query_data()
    '''
    compiled_pattern = re.compile(r'\<([A-Za-z0-9]+)(?:\(([0-9]+)\))?\>')
    replacements = {'ext': ext}
    compiled_pattern = compiled_pattern.sub(lambda match: replacements.get(match.group(1), '*'), pattern)
    compiled_pattern = re.sub('\\*+','*',compiled_pattern) # removes redundant asterisks
    return compiled_pattern

def delimiters_from_pattern(pattern):
    '''
    Extract delimiters from pattern
    (delimiters are the substrings not bounded by < > in the pattern)
    '''
    split_pattern = re.split(r'(<|>)', pattern)
    inside_el = False
    ids_to_remove = [i for i, el in enumerate(split_pattern) if (inside_el := (el == '<')) or (inside_el and (el == '>'))]
    delimiters = [s for i, s in enumerate(split_pattern) if i not in ids_to_remove]
    return list(filter(None, delimiters))


def extract_metadata_from_filename(pattern, filename):
    '''
    Extracts metadata from a filename based on the pattern
    '''
    pattern_components = re.findall(r'\<([A-Za-z0-9]+)(?:\(([0-9]+)\))?\>', pattern)
    metadata = {}
    delimiters = delimiters_from_pattern(pattern.split('/')[-1])
    
    # Calculate delimiter positions in the filename
    delim_ids = [[np.arange(m.start(0), m.end(0)).tolist() for m in re.finditer(re.escape(delimiter), filename)]
                         for delimiter in delimiters]
    delim_ids = list(itertools.chain.from_iterable(delim_ids)) + [[len(filename)]]
    delim_ids = sorted(list(set(itertools.chain.from_iterable(delim_ids))))

    start_index = 0
    for col_name, substr_length in pattern_components:
        substr_length = int(substr_length) if substr_length else None
        on_delimiter = True

        while on_delimiter:
            for d, delimiter_index in enumerate(delim_ids):
                if delimiter_index == start_index:
                    start_index = delimiter_index + 1
                    delim_ids.pop(d)
                    break
                else:
                    on_delimiter = False

        if substr_length is None:
            substr_length = min([i if i > start_index else start_index for i in delim_ids]) - start_index

        # Extract metadata
        end_index = start_index + substr_length
        metadata[col_name.lower()] = filename[start_index:end_index]

        # Update start index for the next field
        start_index = end_index

    return metadata

def query_data(exp_folder,pattern,plate_identifiers=('_CCU384_','_'),exts=('tiff',), plates=('101',)):
    ''' 
    Queries the data from the folders and extracts wells, sites and channels. 
    This is the main function to be changed if the user's has the files 
    arranged in a different way. The output is a dataframe that contains plate, well, 
    site, channel, file name and file path of each image 
    
    Arguments:
        exp_folder: string
            Parent directory where

        pattern: string
            A string specifying locations of metadata as well as length of substrings
            Separate metadata fields are specified with <column_name>, optionally including substring length <column_name(substr_length)>
            Length should be specified if there is no character separating the fields.
            Example: 'Images/<Well(6)><Site(3)><Plane(3)>-<Channel(3)>.<ext>' for phenix data that looks like
                     '/<path_to_network_drive>/<Run>/<plate_scan_subdir>/Images/r03c03f01p01-ch1sk1fk1fl1.tiff'
        
        plate_identifiers: string
            substring or regtex pattern that needs to be in the plate subdirectory name (direct subdirectory of exp_folder) right before the plate number

        exts: tuple of strings
            list of file extensions to search for (e.g., tiff, png, jpg)

        delimiters: tuple of char/str
            list of characters to split filename on
            Default: (' ','-','_')

    Return:
        files_df: pd.DataFrame
            tabular data with metadata fields divided into columns, one row per file
    '''

    print("retrieving files from ", (exp_folder))
    # getting plates that match plate_identifiers
    
    plate_identifiers = list(plate_identifiers)
    if plate_identifiers[0] in ('','/'):
        plate_identifiers[0] = '^'
    if plate_identifiers[1] == ('','/'):
        plate_identifiers[1] = '$'
    plate_identifiers = tuple(plate_identifiers)
    # match plate subdir 
    plate_subdir_pattern = re.compile(f'{plate_identifiers[0]}.*?{plate_identifiers[1]}')
    plate_subdirs = sorted([p for p in os.listdir(exp_folder) if (re.search(plate_subdir_pattern,p) is not None) and
                                                                (os.path.isdir(os.path.join(exp_folder,p)))])
    if plates != 'all' and isinstance(plates,list):
        plates = list(np.asarray(plates).astype(str))
        plate_subdirs = [str(plate_subdir) for plate_subdir in plate_subdirs if re.search('|'.join([plate_identifiers[0]+p+plate_identifiers[1] for p in plates]),plate_subdir) is not None]
        plate_substr = [tuple(p for p in plates if re.search(p,plate_subdir)) for plate_subdir in plate_subdirs]
    else:
        plate_substr = [(p,) for p in plate_subdirs]
        
    print(f'{len(plate_subdirs)} plate(s) found!')
    files_df = pd.DataFrame(columns=['file_path','plate'])
    for i,plate_subdir in enumerate(plate_subdirs):
        plate_pattern = os.path.join(exp_folder,plate_subdir,pattern)
        for _,ext in enumerate(exts):
            compiled_pattern = pattern_to_regex(plate_pattern,ext)
            print(f'finding files matching pattern: {compiled_pattern}')
            files = glob.glob(compiled_pattern)
            files.sort()
            plate_df = pd.DataFrame(files,columns=['file_path'])
            
            plate_df['plate'] = str(plate_substr[i][0])
            plate_df['plate_folder'] = plate_subdir
            files_df = pd.concat([files_df,plate_df],ignore_index=True)

        
    files_df['filename'] = [i.split('/')[-1] for i in files_df.file_path]

    # extracting metadata based on pattern
    files_df['filename_metadata'] = files_df['file_path'].apply(lambda x: extract_metadata_from_filename(plate_pattern, os.path.basename(x)))

    files_df = files_df.join(pd.DataFrame(files_df['filename_metadata'].to_list()))

    # can change this if needed
    default_cols = ['plate','well','site','channel','plane','filename','plate_folder','file_path']
    default_vals = {'plate':'plate01','well':'well01','site':'site01','channel':'channel01','plane':'plane01','plate_folder':''}

    for col, default_val in default_vals.items():
        if col not in files_df.columns:
            
            files_df[col] = str(default_val)
    
    files_df = files_df[default_cols]

    return files_df.convert_dtypes()

def make_well_and_field_list(files):
    ''' inspects the image file name and extracts the unique fields and wells to loop over'''
    wells = np.unique(files.well.astype(str))
    wells.sort()
    fields = np.unique(files.site.astype(str))
    fields.sort()
    return wells, fields

def check_if_file_exists(csv_file, wells, fields,coordinates=None,plate=''):
    ''' Checks if a file for the plate and experiment exists. if it does, if checks what is 
        the last well and field that was calculated. If it equals the last available well and field,
        it considers the computation over, otherwise it extracts where is stops and takes over 
        from there 
        '''
    # check if CSV file exists and non-empty
    if os.path.exists(csv_file) and not os.stat(csv_file).st_size == 0:
        fixed_feature_vector=pd.DataFrame()
        file = None
        for file in glob.glob(csv_file[:-4]+'*'):
            fixed_feature_vector = pd.concat([fixed_feature_vector, pd.read_csv(file, usecols=['well','site','cell_id'])],axis=0,ignore_index=True)


        if len(wells)*len(fields)==len(fixed_feature_vector.drop_duplicates(subset=['well','site'])):
            return file,['Over']
        
        if coordinates is None:
            fixed_feature_vector=fixed_feature_vector.drop_duplicates(subset=['well','site'])
            wells_to_remove=[]
            for well in np.unique(fixed_feature_vector.well):
                if len(np.unique(fixed_feature_vector.loc[fixed_feature_vector.well==well,'site']))==len(fields):
                    wells_to_remove.append(well)
            wells=[i for i in wells if i not in wells_to_remove]       
            return  file,wells

        else:
            coords_df =pd.read_csv(coordinates,index_col=0)
            coords_df =coords_df.loc[coords_df.plate.astype(str)==str(plate)]
            print(coords_df)
            vectors_with_coords_df =pd.merge(fixed_feature_vector,coords_df, on=['well','site','cell_id'],how = 'right',indicator=True)
            print(vectors_with_coords_df)
            coords_df =vectors_with_coords_df[vectors_with_coords_df['_merge']=='right_only'].drop('_merge',axis=1)

            return  file,coords_df

    else:
        if coordinates is None:
            return csv_file,wells
        else:
            coordinates=pd.read_csv(coordinates)
            coordinates=coordinates.loc[coordinates.plate.astype(str)==str(plate)]
        return csv_file,coordinates
    

def load_image(file_path):
    ''' image loader'''
    im = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return im

def flat_field_correction_on_data(files, channel, bf_channel='', n_images=300):
    ''' Calculates the background trend of the entire experiment to be used for flat field correction'''

    flat_field_correction = {}
    n_images = np.min([n_images, np.floor(len(files)/len(channel)).astype(int)])
    for ch in channel:
        B = files.loc[files.channel == ch].sample(n_images)
        img = load_image(B.iloc[0].file_path)
        for j in range(1, n_images):
            img = np.stack([load_image(B.iloc[j].file_path), img], axis=2)

            if ch == bf_channel:
                img = np.mean(img, axis=2)
            else:
                img = np.min(img, axis=2)
            img[img==0]=1

            flat_field_correction[ch] = img
    return flat_field_correction


def process_zstack(image_fnames):
    ''' Computes the stack's max projection from the image name'''
    img = []
    for name in image_fnames:
        img.append(load_image(name))
    img = np.max(np.asarray(img), axis=0)

    return img


def load_and_preprocess(task_files,channels,well,site,zstack,img_size,flat_field_correction,
                        downsampling,return_original=False):
    np_images = []
    original_images= []
    for ch in channels:
        image_fnames = task_files.loc[(task_files.well == well) & (
                task_files.site == site) & (task_files.channel == ch), 'file_path'].values
        
        if zstack is not True:
            img = load_image(image_fnames[0])
        else:
            img = process_zstack(image_fnames)
            
        # Check that the image is of the right format
        if (img is not None) and (img.shape[0] == img_size[0]) and (img.shape[1] == img_size[1]):

            if return_original is True:
                original_images.append(img)

            img = img/(flat_field_correction[ch] * 1e-8)
            if downsampling!=1:
                img,img_size=scale_images(downsampling, img, img_size)

            img = (img/(np.max(img))) * 255
            np_images.append(img.astype('uint8'))
            
        else:
            print('Img corrupted at: ',image_fnames[0])
            return None, None, image_fnames[0]
        
    np_images = np.array(np_images)
    np_images = np.expand_dims(np_images, axis=3)

    return np_images, np.array(original_images), image_fnames[0]


def scale_images(downsampling,img, img_size):
    """
    A function to scale images down using a specified downsampling factor.

    Parameters:
    - downsampling: The factor by which to downsample the image.
    - img: The image to be downscaled.
    - img_size: The size of the original image.

    Returns:
    - img: The downscaled image.
    - img_size: The size of the downscaled image.
    """

    img_size=[int(img_size[0]/downsampling),int(img_size[1]/downsampling)]
    img = cv2.resize(img, img_size)

    return img, img_size

def write_to_csv(df,file):
    flag = not os.path.exists(file) or os.stat(file).st_size == 0
    df.to_csv(f'{file}'[:-4]+'.csv',mode='a',header=flag)
