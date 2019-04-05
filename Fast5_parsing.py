import h5py

def fast5_parse(f5_file):
    r=f5_file
    with h5py.File(r,'r') as hdf: 
        # asd=hdf.get('Raw/Reads/Read_11/')
        # print(asd.attrs['start_time'])

        #### Extract signal

        raw_data=list(hdf['/Raw/Reads/'].values())[0]
        raw_signal=raw_data['Signal'].value
        ### Extract events
        events_data=hdf.get('/Analyses/Basecall_1D_000/BaseCalled_template/Events/')
        events=events_data.value

        ### Extract start time
        start_time=hdf.get('Raw/Reads/')
        sas=''.join(list(start_time.keys()))

        start_t=hdf.get('Raw/Reads/'+sas+'/')
        start_t=start_t.attrs['start_time']
        # print(sas)
        ### Extract duration
        Du_time=hdf.get('Raw/Reads/'+sas+'/')
        Du_time=Du_time.attrs['duration']
        ### Extract Fastq
        Fastq_data=hdf.get('/Analyses/Basecall_1D_000/BaseCalled_template/Fastq/')
        summary_data=hdf.get('/Analyses/Basecall_1D_000/Summary/basecall_1d_template/')
        # mFastq_data=hdf.get('/Analyses/Basecall_1D_000/BaseCalled_template/mFastq/')
        # # print(Fastq_data.value)
        # print(mFastq_data.value)
        ### Extract frequency
        c_freq = hdf.get('/UniqueGlobalKey/context_tags/')
        c_freq = (c_freq.attrs['sample_frequency']).decode('utf-8')
        raw_fastq=(Fastq_data.value).decode('utf-8')
        fastq_decoded=raw_fastq.split('\n')

    return(events, fastq_decoded, raw_signal)
