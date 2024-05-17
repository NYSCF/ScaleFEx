import multiprocessing as mp
from multiprocessing import Manager, Semaphore, Process, Value, Lock, current_process, Queue
from datetime import datetime

class parallelize:
     #Parallelesation
    def __init__(self,tasks,function,max_processes=1,mode='dev'):
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

            # utils.push_all_files_embeddings(self.bucket,self.experiment_name,self.plate,self.subset)
            # utils.terminate_current_instance(self.bucket,self.experiment_name,self.plate)

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
        Worker function to process each task with a timeout.
        This function is designed to be run in a separate process for each worker.
        It processes tasks from a shared queue, with each task being processed with a specified timeout.

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