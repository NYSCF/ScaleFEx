import multiprocessing as mp
from multiprocessing import Manager, Semaphore, Process, Lock, current_process, Queue
from datetime import datetime
import yaml
import os
import sys

class parallelize:
    """
    A class to parallelize tasks using multiprocessing.
    
    Attributes:
        function (callable): The function to apply to each task.
        task_queue (multiprocessing.Queue): Queue to hold tasks for processing.
        mode (str): Operation mode ('dev' or 'prod') for the class behavior.
    
    Methods:
        __init__(tasks, function, max_processes=1, mode='dev'): Initializes the ParallelizeTasks object.
        add_task_to_queue(task): Adds a task to the multiprocessing queue.
        process_worker(semaphore): Worker for processing tasks in 'dev' mode.
        process_worker_prod(semaphore, process_counter, task_count, task_queue, timeout=10000): Worker for processing tasks with timeout in 'prod' mode.
    """
    def __init__(self,tasks,function,max_processes=1,mode='dev'):
        """
        Initialize the ParallelizeTasks with a set of tasks, a target function, and the maximum number of processes.
        
        Parameters:
            tasks (list): List of tasks to be processed.
            function (callable): Function to apply to each task.
            max_processes (int): Maximum number of parallel processes.
            mode (str): Operating mode, 'dev' for development and 'prod' for production.
        """
        if mode =='dev':
            self.function=function
            # Create a multiprocessing Manager Queue to hold the tasks
            self.task_queue = mp.Manager().Queue() 

            # self.start_processing(parllel_process,max_processes) 
                    # Iterate over wells to generate tasks
            for par in tasks:
                task = par # Create the task using well information
                self.add_task_to_queue(task)
            # Create a Semaphore with the maximum number of allowed processes
            process_semaphore = mp.Semaphore(max_processes)

            # Start the worker processes
            processes = []
            for _ in range(max_processes):
                p = mp.Process(target=self.process_worker, args=(process_semaphore,))
                processes.append(p)
                p.start()

            # Wait for all processes to complete
            for p in processes:
                p.join()
                p.close()  # Close the process (release resources)

        if mode =='prod':
            start1 = datetime.now()
            print('multiprocessing starting',datetime.now())
            self.function=function

            #Parallelization setup
            manager = Manager() # For sharing data across processes
            process_counter = manager.Value('i', 0) 
            task_count = manager.Value('i', 0)
            task_count_lock = Lock() # Lock to synchronize access to task_count
            task_queue = manager.Queue()  # Queue for holding tasks to be processed

            # Add tasks to the queue
            with task_count_lock: # Ensure thread-safe increment of task_count
                task_count.value = len(tasks)
                for well in tasks:
                    task_queue.put(well)

            # Create a Semaphore and start worker processes with a timeout
            process_semaphore = Semaphore(max_processes) # This allows for sequential access to tasks
            processes = []
            for _ in range(max_processes):
                p = Process(target=self.process_worker_prod, args=(process_semaphore, process_counter, task_count, task_queue))
                processes.append(p)
                p.start()

            # Wait for all processes to complete
            for p in processes:
                p.join() # Wait for process to finish
                p.close()

            print('All processes have completed their tasks.')
            print('Lenght =', datetime.now() - start1 )


    # Function to add tasks to the queue
    def add_task_to_queue(self,task):
        self.task_queue.put(task)
     
    def process_worker(self,semaphore):
        while True:
            semaphore.acquire()  # Acquire a permit from the semaphore
            if self.task_queue.empty():
                semaphore.release()
                break  # If the queue is empty, break the loop
            task = self.task_queue.get()
            self.function(task) 
            semaphore.release()  # Release the permit

    def process_worker_prod(self, semaphore, process_counter, task_count, task_queue, timeout=10000):
        """
        Process tasks from the queue in development mode. This worker acquires a semaphore,
        processes a task, and releases the semaphore.
        
        Parameters:
            semaphore (multiprocessing.Semaphore): Semaphore to control access to task queue.
        """

        def task_wrapper(queue, task, *args):
            """
            Wrapper function to execute a task and put the result in a queue.
            """
            try:
                result = args[0].function(task)
                queue.put(("Success", result))
            except Exception as e:
                queue.put(("Error", str(e)))

        process_name = current_process().name
        while True:
            semaphore.acquire() # Acquire a semaphore before checking the task queue 
            if task_queue.empty(): # Check if there are no more tasks to process
                semaphore.release()
                break
            task = task_queue.get() # Get a task from the queue
            semaphore.release() # Release the semaphore after getting a task


            # Create a new process for each task with its own timeout
            queue = Queue()
            p = Process(target=task_wrapper, args=(queue, task, self))
            p.start()
            p.join(timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                print(f"Timeout reached for task {task} in process {process_name}. Terminating...")
            else:
                status, message = queue.get()  # Retrieve result or error message
                if status == "Error":
                    print(f"Error processing task {task} in process {process_name}: {message}")
                else:
                    print(f"Task {task} completed successfully in process {process_name}.")

            task_count.value -= 1

        process_counter.value += 1

        print(f"{process_name} completed its tasks or timed out.")

def check_YAML_parameter_validity(yaml_path):
    
    with open(yaml_path, 'rb') as f:
        parameters = yaml.load(f.read(), Loader=yaml.CLoader)
    print(f"Checking parameters in {yaml_path}...")
    
    REQUIRED_PARAMS = ('vector_type','resource','n_of_workers','exp_folder','experiment_name','saving_folder','plates',
                       'max_file_size','plate_identifiers','pattern','exts','image_size','channel',
                       'zstack','ROI','downsampling','QC','FFC','FFC_n_images','csv_coordinates','segmenting_function',
                       'save_coordinates','min_cell_size','max_cell_size','visualization','RNA_channel','Mito_channel',
                       'neurite_tracing','visualize_masks','visualize_crops')
    PASS_CHECK = True
    channels_valid = True
    if not set(REQUIRED_PARAMS).issubset(parameters.keys()):
        print("Missing required general parameters:")
        for param in REQUIRED_PARAMS:
            if param not in parameters:
                print(f"- {param}")
                PASS_CHECK = False
    REQUIRED_PARAMS_AWS = ('s3_bucket','nb_subsets','subset_index','region','instance_type','amazon_image_id','ScaleFExSubnetA',
                           'ScaleFExSubnetB','ScaleFExSubnetC','security_group_id')
    
    if 'resource' in parameters and parameters['resource'] == 'AWS':
        if not set(REQUIRED_PARAMS_AWS).issubset(parameters.keys()):
            print("Missing required AWS parameters:")
            for param in REQUIRED_PARAMS_AWS:
                if param not in parameters:
                    print(f"- {param}")
                    if param == 'channel':
                        channels_valid = False
                    PASS_CHECK = False
    ALL_PARAMS = set(REQUIRED_PARAMS_AWS).union(set(REQUIRED_PARAMS))
    extra_params = set(parameters.keys()) - ALL_PARAMS
    if len(extra_params) > 0:
        print(f"Extra parameters found: {extra_params}")
        print('Please remove them from the YAML file.')
        PASS_CHECK = False

    # LOTS OF PARAMETER DTYPE VALIDATION
    str_params = ('experiment_name','pattern',)
    bool_params = ('zstack','QC','FFC','save_coordinates','visualization','visualize_masks','visualize_crops')
    dir_params = ('exp_folder','saving_folder')
    empty_or_file_params = ('csv_coordinates',)
    list_of_str_params = ('plate_identifiers','exts','channel','plates')
    valid_channel_params = ('RNA_channel','Mito_channel','neurite_tracing')
    int_params = ('n_of_workers','max_file_size','ROI','FFC_n_images','min_cell_size','max_cell_size','nb_subsets','subset_index')
    float_params = ('downsampling',)
    module_params = ('segmenting_function',)
    list_of_int_params = ('image_size',)
    if parameters['vector_type'] not in ('scalefex', 'coordinates',''):
        print(f"- Parameter vector_type ({parameters['vector_type']}) must be in ['scalefex', 'coordinates', ''].")
        PASS_CHECK = False
    if parameters['resource'] not in ('local','AWS'):
        print(f"- Parameter resource ({parameters['resource']}) must be in ['local', 'AWS'].")
        PASS_CHECK = False
    for param in str_params:
        if param in parameters and not isinstance(parameters[param], str):
            print(f"- Parameter {param} ({parameters[param]}) must be a string.")
            PASS_CHECK = False
    for param in bool_params:
        if param in parameters and not isinstance(parameters[param], bool):
            print(f"- Parameter {param} ({parameters[param]}) must be a boolean.")
            PASS_CHECK = False
    for param in dir_params:
        if param in parameters and (not isinstance(parameters[param], str) and not os.path.isdir(parameters[param])):
            print(f"- Parameter {param} ({parameters[param]}) must be a string and a valid directory.")
            PASS_CHECK = False
    for param in empty_or_file_params:
        if param in parameters and (not isinstance(parameters[param], str) and (not parameters[param] and not os.path.isfile(parameters[param]))):
            print(f"- Parameter {param} ({parameters[param]}) must be an empty string or a valid file path.")
            print(len(parameters[param]))
            PASS_CHECK = False
    for param in list_of_str_params:
        if param == 'plates' and isinstance(parameters[param], list):
            parameters[param] = [str(p) for p in parameters[param]]
        if param in parameters and (not isinstance(parameters[param], list) or not all(isinstance(x, str) for x in parameters[param])):
            print(f"- Parameter {param} ({parameters[param]}) must be a list of strings.")
            if param == 'channel':
                channels_valid = False
            PASS_CHECK = False
    if channels_valid:
        for param in valid_channel_params:
            if param in parameters and (not isinstance(parameters[param], str) or not parameters[param] in parameters['channel']+['']):
                print(f"- Parameter {param} ({parameters[param]}) must be a string in list of channels {parameters['channel']}.")
                PASS_CHECK = False
    for param in int_params:
        if param in parameters and (not isinstance(parameters[param], int) or parameters[param] < 0):
            print(f"- Parameter {param} ({parameters[param]}) must be a positive integer.")
            PASS_CHECK = False
    for param in float_params:
        if param in parameters and (not isinstance(parameters[param], (int,float)) or parameters[param] < 0):
            print(f"- Parameter {param} ({parameters[param]}) must be a positive float.")
            PASS_CHECK = False
    for param in list_of_int_params:
        if param in parameters and (not isinstance(parameters[param], list) or not all(isinstance(x, int) for x in parameters[param])\
                                                                            or not all(x > 0 for x in parameters[param])):
            print(f"- Parameter {param} ({parameters[param]}) must be a list of positive integers.")
            PASS_CHECK = False
    for param in module_params:
        if param in parameters and (not isinstance(parameters[param], str)):
            print(f"- Parameter {param} ({parameters[param]}) must be a string and valid module name.")
            PASS_CHECK = False
        else:
            import_module(parameters[param])
            if parameters[param] not in sys.modules:
                print(f"- Parameter {param} ({parameters[param]}) must be a valid module name.")
                PASS_CHECK = False
    
    if PASS_CHECK:
        print("All parameters are valid!")
    return PASS_CHECK

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def import_module(module_name):
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        print(f"Module '{module_name}' not found.")
        return None