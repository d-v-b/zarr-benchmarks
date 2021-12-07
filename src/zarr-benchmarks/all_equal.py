from zarr.util import all_equal
import numpy as np
import timeit
import matplotlib.pyplot as plt
import click
import os

DTYPE = 'uint8'

def timed_all_equal(size, dtype):
    data = np.random.randint(0,1, dtype=dtype, size=size)
    return timeit.timeit(lambda: all_equal(0, data), number=10)

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


@click.command()
def main():
    fname_root = os.path.basename(__file__).split('.')[0]
    fig, _ =  all_equal_plot()
    plt.tight_layout()
    fig.savefig(f'{fname_root}.svg')

if __name__ == '__main__':
    main()
