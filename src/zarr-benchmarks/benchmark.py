import zarr
from zarr.util import all_equal
import numpy as np
import timeit
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory
from itertools import product
import click
from typing import Literal, Union, Optional
from numcodecs.abc import Codec
import numcodecs
DTYPE = 'uint8'

def timed_all_equal(size, dtype):
    data = np.random.randint(0,1, dtype=dtype, size=size)
    return timeit.timeit(lambda: all_equal(0, data), number=10)
        
def timed_array_write(write_empty_chunks: bool,
                      write_value: Union[int, Literal['random']],
                      chunk_size: int,
                      num_chunks: int,
                      compressor: Optional[Codec],
                      repeat=128):

    shape = (chunk_size, num_chunks)
    if write_value == 'random':
        write_value = np.random.randint(0, 255, size=shape, dtype=DTYPE)
    else:
        write_value = np.zeros(shape, dtype=DTYPE) + write_value
    
    with TemporaryDirectory() as store:
        arr = zarr.open(zarr.NestedDirectoryStore(store), 
                        dtype=DTYPE, 
                        shape=shape, 
                        chunks=(shape[0], 1), 
                        write_empty_chunks=write_empty_chunks,
                        compressor=compressor,
                        fill_value=0)
        # initialize the chunks
        arr[:] = 255
        result = timeit.repeat(lambda: arr.set_basic_selection(slice(None), write_value), repeat=repeat, number=1)
    return result

def all_equal_plot():
    sizes = (2 ** k for k in range(20, 30))
    all_equal_result = {size: timed_all_equal(size, DTYPE) for size in sizes}
    as_mb = [str(k / 2**20) for k in all_equal_result.keys()]
    fig_equal, axs_equal = plt.subplots(dpi=200)
    axs_equal.semilogy(as_mb, all_equal_result.values(), marker='o')
    axs_equal.set_ylabel('Time (seconds)')
    axs_equal.set_xlabel('Chunk size (MB)')
    axs_equal.set_title('Chunk emptiness check runtime vs chunk size')
    return fig_equal, axs_equal

def array_write_plot():
    chunk_size = 2 ** 26
    num_chunks = 1
    write_value = 'random'
    compressor=numcodecs.GZip()
    compressor=None
    conditions = tuple(product((True, False), ((0, write_value))))
    write_result = {opts: timed_array_write(*opts, chunk_size=chunk_size, num_chunks=num_chunks, compressor=compressor) for opts in conditions}        
    # normalize to baseline
    baseline = np.median(write_result[(True, 0)])
    fig_write, axs_write = plt.subplots(dpi=200, figsize=(6,5))
    cmap = plt.get_cmap('tab10')
    for idx, result in enumerate(write_result.values()):
        if idx % 2 == 0: 
            marker = 'o'
            label = 'array[:] = array.fill_value'
        else: 
            marker = '^'
            label = 'array[:] = 1'
        color = cmap(idx // 2)
        axs_write.scatter(idx // 2 + np.random.uniform(*np.array([-1,1]) / 4, size=len(result)), 
                          result / baseline, 
                          edgecolors=color,
                          linewidths=1, 
                          facecolors='none',
                          alpha=1.0, 
                          marker=marker,
                          label=label)
    axs_write.set_xticks(range(len(conditions)//2))
    xticklabels = [f'write_empty_chunks={a}\n({b} behavior)' for a,b in ((True, 'Old'), (False, 'New'))]
    axs_write.set_xticklabels(xticklabels)
    axs_write.set_title(f'Time to fill a {chunk_size / 2**20} MB chunk. (Normalized, lower is better)' )
    axs_write.set_ylabel('Normalized duration')
    # null artists for legends
    a = axs_write.scatter([],[],marker='^', color='gray', label=f'Fill with {write_value}', facecolors='none')
    b = axs_write.scatter([],[],marker='o', color='gray', label='Fill with array.fill_value', facecolors='none')
    axs_write.legend(handles=[a,b], loc='center')
    return fig_write, axs_write


def array_write_plot2():
    chunk_size = 2 ** 26
    num_chunks = 1
    write_value = 'random'
    compressor=None
    conditions = tuple(product((True, False), ((0, write_value))))
    write_result = {opts: timed_array_write(*opts, chunk_size=chunk_size, num_chunks=num_chunks, compressor=compressor) for opts in conditions}        
    # normalize to duration baseline
    baseline = np.median(write_result[(True, 0)])
    fig_write, axs_write = plt.subplots(dpi=200, figsize=(6,6), nrows=2)
    [axs.grid(True) for axs in axs_write]
    results_agg = np.array(tuple(write_result.values())) / baseline
    bins = np.linspace(results_agg.min(), results_agg.max(), 50, endpoint=True)
    for idx, result in enumerate(write_result.values()):
        axs_idx = idx // 2
        condition_idx = idx % 2 == 0
        if condition_idx:
            label = 'Fill with array.fill_value'
        else:
            label = f'Fill with {write_value}'
        axs_write[axs_idx].hist(result/ baseline, bins=bins, histtype='stepfilled', alpha=.5, label=label)
    
    axs_write[0].set_title('write_empty_chunks=True')
    axs_write[1].set_title('write_empty_chunks=False')
    axs_write[0].legend(loc='upper right')
    for axs in axs_write:
        axs.set_ylabel('Counts')
        ticks = tuple(filter(lambda v: v > 0, [1.0, *axs.get_xticks()]))
        axs.set_xticks(ticks)
        axs.axvline(1.0)
    axs_write[-1].set_xlabel('Normalized write duration')
    return fig_write, axs_write


@click.command()
@click.option('--all_equal_test', type=bool, is_flag=True)
@click.option('--array_write_test', type=bool, is_flag=True)
def main(all_equal_test: bool = False, array_write_test: bool=False):
    figs = {}
    if all_equal_test:
       fig_equal, _ =  all_equal_plot()
       figs['all_equal_test'] = fig_equal
    if array_write_test:
        fig_write, _ = array_write_plot2()
        figs['empty_chunks_write_test'] = fig_write
    plt.tight_layout()
    [value.savefig(f'{key}.svg') for key, value in figs.items()]

if __name__ == '__main__':
    main()
