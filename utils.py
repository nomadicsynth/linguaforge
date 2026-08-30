import os


# Print the local rank
is_main_process = os.environ.get("LOCAL_RANK", 0) == "0" or os.environ.get("LOCAL_RANK", -1) == -1


# Function that prints to the console only if the process is the main process
def print_if_main_process(*args, **kwargs):
    # global is_main_process
    if is_main_process:
        print(*args, **kwargs)


