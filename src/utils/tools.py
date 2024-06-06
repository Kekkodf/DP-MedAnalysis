import logging
import os

def createLogger() -> object:
    '''
    # createLogger()
    creates the logger object and returns it
    Runtime logs can be found in the path specified
    '''
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    log_dir = './logs'
    log_file = os.path.join(log_dir, 'logger.log')
    
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Remove the log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Create an empty log file
    with open(log_file, 'w'):
        pass
    
    # Create and configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create a file handler for logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    return logger
