import multiprocessing as mp

class parallelize_local:
     #Parallelesation
    def __init__(self,parllel_process,function,max_processes=3):

        self.function=function
        # Create a multiprocessing Manager Queue to hold the tasks
        self.task_queue = mp.Manager().Queue() 

        self.start_processing(parllel_process,max_processes) 

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

    def start_processing(self,parllel_process,max_processes):
        # Iterate over wells to generate tasks
        for par in parllel_process:
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