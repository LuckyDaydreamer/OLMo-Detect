import argparse
from collections import defaultdict
import glob
import gzip
import json
import multiprocessing as mp
import numpy as np
import os
import resource
import shutil
import sys
import time
from tqdm import tqdm

HACK = 100000

tokenizer = None
token_dtype = None
version = None


def process_tokenized_files(args):
    """
    Processes a sequence of pre-tokenized .npy files by reading them as raw 
    binary buffers, which emulates the low-level data access method used 
    by OLMo's official code.
    """
    
    # ... (initializations and skip check remain the same) ...

    ds_paths = [os.path.join(args.save_dir, f'tokenized.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    od_paths = [os.path.join(args.save_dir, f'offset.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    
    if all([os.path.exists(ds_path) for ds_path in ds_paths]):
         if all([os.path.getsize(ds_path) > 0 for ds_path in ds_paths]):
            print('Step 1 (process_tokenized_files): Skipped. All sharded files already exist and are non-empty.')
            return

    print('Step 1 (process_tokenized_files): Starting ...')

    input_files = list(sorted(glob.glob(os.path.join(args.data_dir, '**', '*.npy'), recursive=True)))
    if not input_files:
        raise FileNotFoundError(f"No .npy files found in the data directory: {args.data_dir}")

    ds_fouts = [open(ds_path, 'wb') for ds_path in ds_paths]
    od_fouts = [open(od_path, 'wb') for od_path in od_paths]
    ods = [0 for _ in od_fouts] 
    
    print(f"Found {len(input_files)} total .npy files. Sharding into {args.shards} output files.")

    # 4. Process each input .npy file
    for file_index, input_path in enumerate(tqdm(input_files, desc="Processing .npy files")):
        
        final_shard_idx = file_index % args.shards
        
        if final_shard_idx % args.workers == args.worker_id:
            
            worker_shard_index = (final_shard_idx - args.worker_id) // args.workers
            ds_fout = ds_fouts[worker_shard_index]
            od_fout = od_fouts[worker_shard_index]
            
            # --- Load and Write Data (Raw Binary Loading) ---
            
            # 1. Read the raw byte content of the .npy file
            try:
                with open(input_path, 'rb') as f:
                    # Read all bytes into memory. This is unavoidable for `np.frombuffer` on the whole file.
                    # NOTE: This assumes the entire file is just token data bytes *without* the NumPy header.
                    # If the NumPy header is present, it will be treated as token data, which might be okay.
                    raw_bytes = f.read() 
                
                # 2. Convert the raw bytes into a NumPy array of the correct token type
                # The assumption is that raw_bytes contains token IDs of type args.token_dtype
                token_array = np.frombuffer(raw_bytes, dtype=token_dtype)

            except Exception as e:
                print(f"Error loading {input_path}: Failed to read and convert raw bytes. Error: {e}. Skipping.")
                continue

            # 3. Handle a potential NumPy header offset (OLMo likely skips this, but we must account for it)
            # If the file *does* contain the standard 128-byte NumPy header, we should adjust.
            # This is complex and usually requires using np.load(mmap_mode) to get the offset,
            # but since OLMo avoids that, we will proceed assuming raw bytes is the way.
            
            if token_array.size == 0:
                 print(f"Warning: {input_path} resulted in an empty token array. Skipping.")
                 continue

            # --- Continue with writing logic ---

            # Write the offset of the new document
            od_fout.write(np.array([ods[worker_shard_index]], dtype=np.uint64).view(np.uint8).tobytes())
            
            # Write the document separator
            ds_fout.write(args.doc_sep)
            
            # Write the raw token bytes directly
            # We use the raw bytes read earlier for efficiency
            ds_fout.write(raw_bytes)
            
            # Update the total size of the output shard
            ods[worker_shard_index] += len(args.doc_sep) + len(raw_bytes)

            del raw_bytes 
            del token_array

    # 5. Close all file handles
    for f in ds_fouts + od_fouts:
        f.close()
        
    print('Step 1 (process_tokenized_files): Done.')
    

def build_sa(args):

    ds_paths = [os.path.join(args.save_dir, f'tokenized.{i}') for i in range(args.worker_id, args.shards, args.workers)]

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    for t, ds_path in enumerate(ds_paths):
        print(f'Shard {t} / {len(ds_paths)}', flush=True)

        sa_path = ds_path.replace('tokenized', 'table')
        if os.path.exists(sa_path):
            print(f'Step 2 (build_sa): Skipped. File already exists.', flush=True)
            continue

        if not os.path.exists(ds_path):
            print(f'Step 2 (build_sa): Skipped. Tokenized file {ds_path} does not exist.', flush=True)
            continue
            
        ds_size = os.path.getsize(ds_path)
        if ds_size == 0:
            print(f'Step 2 (build_sa): Skipped. Tokenized file {ds_path} is empty (size=0).', flush=True)
            continue

        print('Step 2 (build_sa): Starting ...', flush=True)
        start_time_all = time.time()

        # -------- Step 2.1 (make-part) -------- #

        print(f'\tStep 2.1 (make-part): Starting ...', flush=True)
        start_time = time.time()

        ds_size = os.path.getsize(ds_path)
        ratio = int(np.ceil(np.log2(ds_size) / 8))
        mem_bytes = args.mem * 1024**3
        num_job_batches = 1
        while num_job_batches * (mem_bytes // (12 if args.token_width == 1 else 8)) < ds_size:
            num_job_batches *= 2
        parallel_jobs = args.cpus
        total_jobs = num_job_batches * parallel_jobs
        print(f'Using {num_job_batches} batches of {parallel_jobs} jobs each, for a total of {total_jobs} jobs.', flush=True)

        S = ds_size // total_jobs
        # Make sure that parts contain whole tokens
        if S % args.token_width != 0:
            S += args.token_width - S % args.token_width

        parts_dir = os.path.join(args.temp_dir, f'parts-{args.worker_id}')
        shutil.rmtree(parts_dir, ignore_errors=True)
        os.makedirs(parts_dir)

        for batch_start in tqdm(list(range(0, total_jobs, parallel_jobs))):
            batch_end = min(batch_start+parallel_jobs, total_jobs)
            batch_ranges = []
            for i in range(batch_start, batch_end):
                s, e = i*S, min((i+1)*S+HACK, ds_size)
                batch_ranges.append((s, e))
            pipes = []
            for (s, e) in batch_ranges:
                pipes.append(os.popen(f'./rust_indexing make-part --data-file {ds_path} --parts-dir {parts_dir} --start-byte {s} --end-byte {e} --ratio {ratio} --token-width {args.token_width}'))
            [pipe.read() for pipe in pipes]
            if any([pipe.close() is not None for pipe in pipes]):
                print('\tStep 2.1 (make-part): Something went wrong', flush=True)
                exit(1)

        end_time = time.time()
        print(f'\tStep 2.1 (make-part): Done. Took {end_time-start_time:.2f} seconds', flush=True)

        # -------- Step 2.2 (merge) -------- #

        print(f'\tStep 2.2 (merge): Starting ...', flush=True)
        start_time = time.time()

        merged_dir = os.path.join(args.temp_dir, f'merged-{args.worker_id}')
        shutil.rmtree(merged_dir, ignore_errors=True)
        os.makedirs(merged_dir)

        pipe = os.popen(f'./rust_indexing merge --data-file {ds_path} --parts-dir {parts_dir} --merged-dir {merged_dir} --num-threads {args.cpus} --hacksize {HACK} --ratio {ratio} --token-width {args.token_width}')
        pipe.read()
        if pipe.close() is not None:
            print('\tStep 2.2 (merge): Something went wrong', flush=True)
            exit(1)

        shutil.rmtree(parts_dir)

        end_time = time.time()
        print(f'\tStep 2.2 (merge): Done. Took {end_time-start_time:.2f} seconds', flush=True)

        # -------- Step 2.3 (concat) -------- #

        print(f'\tStep 2.3 (concat): Starting ...', flush=True)
        start_time = time.time()

        pipe = os.popen(f'./rust_indexing concat --data-file {ds_path} --merged-dir {merged_dir} --merged-file {sa_path} --num-threads {args.cpus} --ratio {ratio} --token-width {args.token_width}')
        pipe.read()
        if pipe.close() is not None:
            print('\tStep 2.3 (concat): Something went wrong', flush=True)
            exit(1)

        shutil.rmtree(merged_dir)

        end_time = time.time()
        print(f'\tStep 2.3 (concat): Done. Took {end_time-start_time:.2f} seconds', flush=True)

        end_time_all = time.time()
        print(f'Step 2 (build_sa): Done. Took {end_time_all-start_time_all:.2f} seconds', flush=True)


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the raw text corpus. Must be absolute path.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the pre-tokenized .npy files (e.g., part-*-*.npy). Must be absolute path.')
    parser.add_argument('--temp_dir', type=str, default=None, help='Directory where temporary indexing files are stored. Must be absolute path.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory where the final index files are stored. Must be absolute path.')
    parser.add_argument('--version', type=int, default=4, choices=[4, 5], help='Version of the index.')
    parser.add_argument('--tokenizer', type=str, default=None, choices=[None, 'gpt2', 'llama', 'olmo', 'olmo-2-0425-1b', 'dolma2'])
    parser.add_argument('--token_dtype', type=str, default='u16', choices=['u8', 'u16', 'u32'], help='Data type for tokens.')
    parser.add_argument('--add_metadata', default=False, action='store_true', help='Whether to store document metadata in the index.')
    parser.add_argument('--add_unigram', default=False, action='store_true', help='Whether to precompute unigram counts.')
    parser.add_argument('--shards', type=int, default=1, help='Number of shards to split the index into.')
    parser.add_argument('--workers', type=int, default=1, help='Total number of workers. Must be a divisor of shards.')
    parser.add_argument('--worker_id', type=int, default=0, help='The worker ID of this process. Must be in range [0, workers).')
    parser.add_argument('--batch_size', type=int, default=65536, help='Batch size for tokenization.')
    parser.add_argument('--cpus', type=int, default=mp.cpu_count(), help='Number of CPU cores available to the program.')
    parser.add_argument('--mem', type=int, required=True, help='Amount of memory in GiB available to the program.')
    parser.add_argument('--ulimit', type=int, default=102400, help='Maximum number of open files allowed.')
    args = parser.parse_args()

    if args.temp_dir is None:
        args.temp_dir = args.save_dir
    args.data_dir = args.data_dir.rstrip('/')
    args.temp_dir = args.temp_dir.rstrip('/')
    args.save_dir = args.save_dir.rstrip('/')

    assert args.batch_size > 0
    assert args.cpus > 0
    assert args.shards > 0
    assert args.workers > 0
    assert 0 <= args.worker_id < args.workers
    assert args.shards % args.workers == 0

    global token_dtype, version
    if args.token_dtype == 'u8':
        token_dtype = np.uint8
        args.token_width = 1
        args.doc_sep = b'\xff'
    elif args.token_dtype == 'u16':
        token_dtype = np.uint16
        args.token_width = 2
        args.doc_sep = b'\xff\xff'
    elif args.token_dtype == 'u32':
        token_dtype = np.uint32
        args.token_width = 4
        args.doc_sep = b'\xff\xff\xff\xff'
    else:
        raise ValueError(f'Unknown token_dtype: {args.token_dtype}')
    version = args.version

    assert os.path.exists(args.data_dir)
    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    assert sys.byteorder == 'little'
    resource.setrlimit(resource.RLIMIT_NOFILE, (args.ulimit, args.ulimit))

    process_tokenized_files(args)
    build_sa(args)


if __name__ == '__main__':
    main()
