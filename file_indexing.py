import os

def file_index(f_path):
    l_filename=[]
    # for filenames in os.listdir(str(f_path)):
    #     if filenames.endswith(".fast5"):
    #         l_filename.append(f_path+filenames.replace('\n',''))

    for dirpath, dirs, files in os.walk(f_path):
        for filename in files:
            if filename.endswith(".fast5"):
                # print(dirpath)
                l_filename.append(os.path.join(dirpath,filename))
    return l_filename