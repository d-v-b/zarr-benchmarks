import zarr
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

def array_write_plot(compressor: Optional[Codec]):
    chunk_size = 2 ** 26
    num_chunks = 1
    write_value = 'random'
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
@click.option('--compressor', type=str)
def main(compressor: Optional[str]):
    if compressor is not None:
        compressor = numcodecs.registry.get_codec({'id': compressor.lower()})
    fig, _ = array_write_plot(compressor=compressor)
    plt.tight_layout()
    fig.savefig(f'empy_chunks_benchmark_compressor-{compressor}.svg')

if __name__ == '__main__':
    main()