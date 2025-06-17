"""Logging utilities for DAFTAR-ML."""

import os
import logging
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(output_dir=None, verbose=False):
    """Set up logging configuration.
    
    Args:
        output_dir: Optional directory to write log file to
        verbose: Whether to enable verbose console output
    """
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Configure Optuna logger to use the same level
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.setLevel(logging.INFO)
    optuna_logger.propagate = True
    
    # Clear existing handlers
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    if optuna_logger.handlers:
        for handler in optuna_logger.handlers[:]:
            optuna_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if output_dir:
        try:
            output_dir = os.path.abspath(output_dir)
            
            # Create log file path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(output_dir, f'DAFTAR-ML_{timestamp}.log')
            
            # Save original stdout and stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            # Set up filtering for console output only if verbose is False
            if not verbose:
                # Patterns to keep in console output when not verbose
                important_patterns = [
                    r'Starting DAFTAR-ML pipeline',
                    r'Loading data',
                    r'Processing fold',
                    r'Completed fold',
                    r'Saving results',
                    r'Analysis complete',
                    r'Pipeline completed',
                    r'DAFTAR-ML analysis completed',
                    r'Results saved to',
                    r'To reproduce this run in the future',
                    r'daftar --input',
                ]
                important_regex = re.compile('|'.join(important_patterns))
                
                # Create a filtered version for console output
                class FilteredConsoleOutput:
                    def __init__(self, original_stream):
                        self.original_stream = original_stream
                        self.buffer = ""
                    
                    def write(self, message):
                        # For console output, filter based on patterns
                        self.buffer += message
                        if '\n' in self.buffer:
                            lines = self.buffer.split('\n')
                            self.buffer = lines.pop()  # Keep last incomplete line in buffer
                            
                            for line in lines:
                                # Skip Optuna timestamped messages with trial information
                                if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - Trial \d+ finished with value:', line.strip()):
                                    continue
                                # Skip all other Optuna verbose output lines that start with [I ]
                                elif line.strip().startswith('[I '):
                                    continue
                                # Check if this line contains important information from our patterns
                                elif important_regex.search(line) or line.strip() == "":
                                    # Pass through to console
                                    self.original_stream.write(line + '\n')
                                # Show all other lines that don't match the filtered patterns
                                else:
                                    # Pass through to console
                                    self.original_stream.write(line + '\n')
                    
                    def flush(self):
                        self.original_stream.flush()
                    
                    def close(self):
                        pass
                
                # Create a class that will always capture the full verbose output to the log file
                class LogAllOutput:
                    def __init__(self, original_stream, log_file_path, filtered_stream=None):
                        self.original_stream = original_stream
                        self.log_file_path = log_file_path
                        self.filtered_stream = filtered_stream
                        # Open log file in append mode and keep it open
                        self.log_file = open(log_file_path, 'a')
                    
                    def write(self, message):
                        # Write to the console (filtered or not)
                        if self.filtered_stream:
                            self.filtered_stream.write(message)
                        else:
                            self.original_stream.write(message)
                        
                        # Always write to the log file exactly as if verbose was True
                        if message:
                            self.log_file.write(message)
                            self.log_file.flush()
                    
                    def flush(self):
                        if self.filtered_stream:
                            self.filtered_stream.flush()
                        else:
                            self.original_stream.flush()
                        self.log_file.flush()
                
                # Set up filtered console streams
                filtered_stdout = FilteredConsoleOutput(original_stdout)
                filtered_stderr = FilteredConsoleOutput(original_stderr)
                
                # Override stdout and stderr to capture all output to log file but filter console
                sys.stdout = LogAllOutput(original_stdout, log_file, filtered_stdout)
                sys.stderr = LogAllOutput(original_stderr, log_file, filtered_stderr)
            else:
                # Verbose mode: Console gets everything, and log file gets the exact same
                class LogVerboseOutput:
                    def __init__(self, original_stream, log_file_path):
                        self.original_stream = original_stream
                        # Open log file in append mode and keep it open
                        self.log_file = open(log_file_path, 'a')
                    
                    def write(self, message):
                        # Write to the console directly
                        self.original_stream.write(message)
                        
                        # Always write to the log file exactly what goes to console
                        if message:
                            self.log_file.write(message)
                            self.log_file.flush()
                    
                    def flush(self):
                        self.original_stream.flush()
                        self.log_file.flush()
                
                # Override stdout and stderr to capture all output both to console and log file
                sys.stdout = LogVerboseOutput(original_stdout, log_file)
                sys.stderr = LogVerboseOutput(original_stderr, log_file)
            
            # Add console handler for structured logging
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # Log file is already capturing all output via stdout/stderr redirection
            
        except Exception as e:
            # In case of failure, at least add a simple console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.warning(f"Failed to set up file logging: {e}")
    else:
        # No output directory, just add a console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
