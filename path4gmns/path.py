""" Find shortest path given a from node and a to node

Two path engines are provided:
1. C++ engine which is a special implementation of the deque implementation in
   C++ and built into path_engine.dll.
2. Python engine which provides three implementations: FIFO, Deque, and
   heap-Dijkstra. The default is deque.
"""


import collections
import ctypes
import heapq
import platform
from os import path
from time import time

import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import zipfile
import io
import openmatrix as omx
import os

from .consts import MAX_LABEL_COST


__all__ = [
    'single_source_shortest_path',
    'output_path_sequence',
    'find_shortest_path',
    'find_shortest_path_distance'
    'find_shortest_path_network',
    'find_path_for_agents',
    'benchmark_apsp'
]


_os = platform.system()
if _os.startswith('Windows'):
    _dll_file = path.join(path.dirname(__file__), 'bin/path_engine.dll')
elif _os.startswith('Linux'):
    _dll_file = path.join(path.dirname(__file__), 'bin/path_engine.so')
elif _os.startswith('Darwin'):
    # check CPU is Intel or Apple Silicon
    if platform.machine().startswith('x86_64'):
        _dll_file = path.join(path.dirname(__file__), 'bin/path_engine_x86.dylib')
    else:
        _dll_file = path.join(path.dirname(__file__), 'bin/path_engine_arm.dylib')
else:
    raise Exception('Please build the shared library compatible to your OS\
                    using source files in engine_cpp!')

_cdll = ctypes.cdll.LoadLibrary(_dll_file)

# set up the argument types for the shortest path function in dll.
_cdll.shortest_path_n.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_wchar_p),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_wchar_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]


def _optimal_label_correcting_CAPI(G,
                                   origin_node_no,
                                   departure_time=0):
    """ call the deque implementation of MLC written in cpp

    node_label_cost, node_predecessor, and link_predecessor are still
    initialized in shortest_path() even the source node has no outgoing links.
    """
    _cdll.shortest_path_n(origin_node_no,
                          G.get_node_size(),
                          G.get_from_node_no_arr(),
                          G.get_to_node_no_arr(),
                          G.get_first_links(),
                          G.get_last_links(),
                          G.get_sorted_link_no_arr(),
                          G.get_allowed_uses(),
                          G.get_link_costs(),
                          G.get_node_label_costs(),
                          G.get_node_preds(),
                          G.get_link_preds(),
                          G.get_queue_next(),
                          G.get_agent_type_name(),
                          MAX_LABEL_COST,
                          G.get_last_thru_node(),
                          departure_time)


def _single_source_shortest_path_fifo(G, origin_node_no):
    """ FIFO implementation of MLC using built-in list and indicator array

    The caller is responsible for initializing node_label_cost,
    node_predecessor, and link_predecessor.
    """
    G.node_label_cost[origin_node_no] = 0
    # node status array
    status = [0] * G.node_size
    # scan eligible list
    SEList = []
    SEList.append(origin_node_no)

    # label correcting
    while SEList:
        from_node = SEList.pop(0)
        status[from_node] = 0
        for link in G.nodes[from_node].outgoing_links:
            to_node = link.to_node_no
            new_to_node_cost = (G.node_label_cost[from_node]
                                + link.fftt)
            # we only compare cost at the downstream node ToID
            # at the new arrival time t
            if new_to_node_cost < G.node_label_cost[to_node]:
                # update cost label and node/time predecessor
                G.node_label_cost[to_node] = new_to_node_cost
                # pointer to previous physical node index
                # from the current label at current node and time
                G.node_preds[to_node] = from_node
                # pointer to previous physical node index
                # from the current label at current node and time
                G.link_preds[to_node] = link.link_no
                if not status[to_node]:
                    SEList.append(to_node)
                    status[to_node] = 1


def _single_source_shortest_path_deque(G, origin_node_no):
    """ Deque implementation of MLC using deque list and indicator array

    The caller is responsible for initializing node_label_cost,
    node_predecessor, and link_predecessor.

    Adopted and modified from
    https://github.com/jdlph/shortest-path-algorithms
    """
    G.node_label_cost[origin_node_no] = 0
    # node status array
    status = [0] * G.node_size
    # scan eligible list
    SEList = collections.deque()
    SEList.append(origin_node_no)

    # label correcting
    while SEList:
        from_node = SEList.popleft()
        status[from_node] = 2
        for link in G.nodes[from_node].outgoing_links:
            to_node = link.to_node_no
            new_to_node_cost = (G.node_label_cost[from_node]
                                + link.fftt) 
            # we only compare cost at the downstream node ToID
            # at the new arrival time t
            if new_to_node_cost < G.node_label_cost[to_node]:
                # update cost label and node/time predecessor
                G.node_label_cost[to_node] = new_to_node_cost
                # pointer to previous physical node index
                # from the current label at current node and time
                G.node_preds[to_node] = from_node
                # pointer to previous physical node index
                # from the current label at current node and time
                G.link_preds[to_node] = link.link_no
                if status[to_node] != 1:
                    if status[to_node] == 2:
                        SEList.appendleft(to_node)
                    else:
                        SEList.append(to_node)
                    status[to_node] = 1


def _single_source_shortest_path_dijkstra(G, origin_node_no):
    """ Simplified heap-Dijkstra's Algorithm using heapq

    The caller is responsible for initializing node_label_cost,
    node_predecessor, and link_predecessor.

    Adopted and modified from
    https://github.com/jdlph/shortest-path-algorithms
    """
    G.node_label_cost[origin_node_no] = 0
    # node status array
    status = [0] * G.node_size
    # scan eligible list
    SEList = []
    heapq.heapify(SEList)
    heapq.heappush(SEList, (G.node_label_cost[origin_node_no], origin_node_no))

    # label setting
    while SEList:
        (label_cost, from_node) = heapq.heappop(SEList)
        # already scanned, pass it
        if status[from_node] == 1:
            continue
        status[from_node] = 1
        for link in G.nodes[from_node].outgoing_links:
            to_node = link.to_node_no
            new_to_node_cost = label_cost + link.fftt
            # we only compare cost at the downstream node ToID
            # at the new arrival time t
            if new_to_node_cost < G.node_label_cost[to_node]:
                # update cost label and node/time predecessor
                G.node_label_cost[to_node] = new_to_node_cost
                # pointer to previous physical node index
                # from the current label at current node and time
                G.node_preds[to_node] = from_node
                # pointer to previous physical node index
                # from the current label at current node and time
                G.link_preds[to_node] = link.link_no
                heapq.heappush(SEList, (G.node_label_cost[to_node], to_node))

                    
def single_source_shortest_path(G, origin_node_id,
                                engine_type='c', sp_algm='deque'):

    origin_node_no = G.get_node_no(origin_node_id)

    if engine_type.lower() == 'c':
        G.allocate_for_CAPI()
        _optimal_label_correcting_CAPI(G, origin_node_no)  
    else:
        # just in case user uses C++ and Python path engines in a mixed way
        G.has_capi_allocated = False

        # Initialization for all nodes
        G.node_label_cost = [MAX_LABEL_COST] * G.node_size
        # pointer to previous node index from the current label at current node
        G.node_preds = [-1] * G.node_size
        # pointer to previous node index from the current label at current node
        G.link_preds = [-1] * G.node_size

        # make sure node_label_cost, node_predecessor, and link_predecessor
        # are initialized even the source node has no outgoing links
        if not G.nodes[origin_node_no].outgoing_links:
            return

        if sp_algm.lower() == 'fifo':
            _single_source_shortest_path_fifo(G, origin_node_no)
        elif sp_algm.lower() == 'deque':
            _single_source_shortest_path_deque(G, origin_node_no)
        elif sp_algm.lower() == 'dijkstra':
            _single_source_shortest_path_dijkstra(G, origin_node_no)
        else:
            raise Exception('Please choose correct shortest path algorithm: '
                            'fifo or deque or dijkstra')
                        

def output_path_sequence(G, to_node_id, type='node'):
    """ output shortest path in terms of node sequence or link sequence

    Note that this function returns GENERATOR rather than list.
    """
    path = []
    curr_node_no = G.map_id_to_no[to_node_id]

    if type.startswith('node'):
        # retrieve the sequence backwards
        while curr_node_no >= 0:
            path.append(curr_node_no)
            curr_node_no = G.node_preds[curr_node_no]
        # reverse the sequence
        for node_no in reversed(path):
            yield G.map_no_to_id[node_no]
    else:
        # retrieve the sequence backwards
        curr_link_no = G.link_preds[curr_node_no]
        while curr_link_no >= 0:
            path.append(curr_link_no)
            curr_node_no = G.node_preds[curr_node_no]
            curr_link_no = G.link_preds[curr_node_no]
        # reverse the sequence
        for link_no in reversed(path):
            yield G.links[link_no].get_link_id()


def _get_path_cost(G, to_node_id):
    to_node_no = G.map_id_to_no[to_node_id]

    return G.node_label_cost[to_node_no]


def find_shortest_path(G, from_node_id, to_node_id, seq_type='node'):
    if from_node_id not in G.map_id_to_no:
        raise Exception(f'Node ID: {from_node_id} not in the network')
    if to_node_id not in G.map_id_to_no:
        raise Exception(f'Node ID: {to_node_id} not in the network')

    single_source_shortest_path(G, from_node_id, engine_type='c')

    path_cost = _get_path_cost(G, to_node_id)

    if path_cost >= MAX_LABEL_COST:
        return f'distance: infinity | path: '

    path = ';'.join(
        str(x) for x in output_path_sequence(G, to_node_id, seq_type)
    )

    if seq_type.startswith('node'):
        return f'distance: {path_cost:.2f} mi | node path: {path}'
    else:
        return f'distance: {path_cost:.2f} mi | link path: {path}'

def find_shortest_path_distance(G, from_node_id, to_node_id):
    # exceptions
    if from_node_id not in G.map_id_to_no:
        #return None
        raise Exception(f'Node ID: {from_node_id} not in the network')
    if to_node_id not in G.map_id_to_no:
        #return None
        raise Exception(f'Node ID: {to_node_id} not in the network')
    
    single_source_shortest_path(G, from_node_id, engine_type='c')

    path_cost = _get_path_cost(G, to_node_id)
  
    if path_cost >= MAX_LABEL_COST:
        return 9999999
    else:
       return path_cost

def compute_row_distances(G, row_node, nodes):
    """
    Computes the shortest path distances from a single node to all other nodes.

    Parameters:
        row_node (int): The source node ID.
        nodes (np.ndarray): Array of all node IDs.
        network: The loaded network object.

    Returns:
        list: List of shortest path distances.
    """
    return [find_shortest_path_distance(G, row_node, col_node) for col_node in nodes]


def create_numpy_matrix_parallel(G, nodes):
    """
    Creates a shortest path distance matrix using parallel processing.

    Parameters:
        nodes (np.ndarray): Array of all node IDs.
        network: The loaded network object.

    Returns:
        np.ndarray: The shortest path distance matrix.
    """
    start_time = time.time()

    # Use joblib with tqdm for parallel computation
    skim_matrix = Parallel(n_jobs=-1)(
        delayed(compute_row_distances)(G, row_node, nodes)  # Corrected function call
        for row_node in tqdm(nodes, desc="Computing shortest paths"))
    
    elapsed_time = time.time() - start_time
    print(f"Matrix Creation Time: {elapsed_time:.2f} s")

    return np.array(skim_matrix)

def save_as_omx(matrix, nodes, output_path):
    """
    Saves the shortest path matrix as an OMX file.

    Parameters:
        matrix (np.ndarray): The shortest path matrix.
        nodes (np.ndarray): Array of node IDs.
        output_path (str): Path to save the OMX file.
    """
    with omx.open_file(output_path, "w") as omx_out:
        omx_out.create_matrix(
            name="shortest_path_matrix",
            obj=matrix,
            title="Shortest Path Distance Matrix",
            attrs={"Description": "This matrix contains the shortest path distances between nodes"},
        )
        omx_out.create_mapping("nodes", nodes)

    print(f"Matrix saved as OMX file: {output_path}")


def save_as_csv(df_matrix, output_path):
    """
    Saves the shortest path matrix as a CSV file.

    Parameters:
        df_matrix (pd.DataFrame): DataFrame containing the shortest path matrix.
        output_path (str): Path to save the CSV file.
    """
    df_matrix.to_csv(output_path, index=True, header=True, float_format="%.2f")
    print(f"Matrix saved to {output_path}")


def save_as_zip(df_matrix, output_path, csv_filename):
    """
    Saves the shortest path matrix as a zipped CSV file.

    Parameters:
        df_matrix (pd.DataFrame): DataFrame containing the shortest path matrix.
        output_path (str): Path to save the zip file.
        csv_filename (str): Name of the CSV file inside the zip archive.
    """
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        with io.StringIO() as csv_buffer:
            df_matrix.to_csv(csv_buffer, index=True, header=True, float_format="%.2f")
            zipf.writestr(csv_filename, csv_buffer.getvalue())

    print(f"Matrix Zipfile saved to {output_path}")

def find_shortest_path_network(G, output_dir, output_type):
    """
    Reads network files, computes the shortest path distance matrix, and saves it in the desired format.

    Parameters:
        output_dir (str): Directory to save the output files.
        output_type (str): Desired output format (".csv", ".omx", ".zip").
    """
    
    #Convert network lenghts to fftt in minutes
    #This currently doesn't function as it should. 
    for i in range(G.link_size):
        G.links[i].length = G.links[i].fftt

    # Convert node IDs to a NumPy array
    nodes = np.array([G.nodes[i].node_id for i in range(G.node_size)])

    # Compute shortest path matrix in parallel
    skim_matrix = create_numpy_matrix_parallel(G, nodes)

    # Convert matrix to DataFrame
    df_skim_matrix = pd.DataFrame(skim_matrix, index=nodes, columns=nodes)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the matrix in the requested format
    output_path = os.path.join(output_dir, f"{"shortest_path_matrix"}{output_type}")

    if output_type == ".omx":
        save_as_omx(skim_matrix, nodes, output_path)

    elif output_type == ".csv":
        save_as_csv(df_skim_matrix, output_path)

    elif output_type == ".zip":
        csv_filename = "shortest_path_matrix.csv"
        save_as_zip(df_skim_matrix, output_path, csv_filename)

def find_path_for_agents(G, column_pool, engine_type='c'):
    """ find and set up shortest path for each agent

    the internal node and links will be used to set up the node sequence and
    link sequence respectively

    Note that we do not cache the predecessors and label cost even some agents
    may share the same origin and each call of the single-source path algorithm
    will calculate the shortest path tree from the source node.
    """
    if G.get_agent_count() == 0:
        print('setting up individual agents')
        G.setup_agents(column_pool)

    from_node_id_prev = ''
    for agent in G.agents:
        from_node_id = agent.o_node_id
        to_node_id = agent.d_node_id

        # just in case agent has the same origin and destination
        if from_node_id == to_node_id:
            continue

        if from_node_id not in G.map_id_to_no:
            raise Exception(f'Node ID: {from_node_id} not in the network')
        if to_node_id not in G.map_id_to_no:
            raise Exception(f'Node ID: {to_node_id} not in the network')

        # simple caching strategy
        # if the current from_node_id is the same as from_node_id_prev,
        # then there is no need to redo shortest path calculation.
        if from_node_id != from_node_id_prev:
            from_node_id_prev = from_node_id
            single_source_shortest_path(G, from_node_id, engine_type)

        node_path = []
        link_path = []

        curr_node_no = G.map_id_to_no[to_node_id]
        # set up the cost
        agent.path_cost = G.node_label_cost[curr_node_no]

        # retrieve the sequence backwards
        while curr_node_no >= 0:
            node_path.append(curr_node_no)
            curr_link_no = G.link_preds[curr_node_no]
            if curr_link_no >= 0:
                link_path.append(curr_link_no)
            curr_node_no = G.node_preds[curr_node_no]

        # make sure it is a valid path
        if not link_path:
            continue

        agent.node_path = [x for x in node_path]
        agent.link_path = [x for x in link_path]


def benchmark_apsp(G):
    st = time()

    for k in G.map_id_to_no:
        single_source_shortest_path(G, k, 'c')

    print(f'processing time of finding all-pairs shortest paths: {time()-st:.4f} s')