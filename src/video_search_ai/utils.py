import os  
 

VID_EXTENSIONS = ["mp4", "mov", "avi"]

def video_file_format_check(filename : str):
    format = filename.split(".")[-1]
    if format not in VID_EXTENSIONS:
        raise TypeError("Invalid video format for search")



