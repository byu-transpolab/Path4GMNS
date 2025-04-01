""" Find shortest path given a from node and a to node

Two path engines are provided:
1. C++ engine which is a special implementation of the deque implementation in
   C++ and built into path_engine.dll.
2. Python engine which provides three implementations: FIFO, Deque, and
   heap-Dijkstra. The default is deque.
"""


import ctypes
import csv
import platform
from os import path
from time import time
from .consts import MAX_LABEL_COST

import numpy as np
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import openmatrix as omx
import os
from datetime import datetime




__all__ = [
    'single_source_shortest_path',
    'output_path_sequence',
    'find_shortest_path',
    'get_shortest_path',
    'find_shortest_path_network',
    'get_shortest_path_tree',
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


# simple caching for _single_source_shortest_path_versatile()
_prev_cost_type = 'time'

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


            
def single_source_shortest_path(G, orig_node_id, cost_type):
    """ use this one with UE, accessibility, and equity """
    G.allocate_for_CAPI()
    
    global _prev_cost_type
    if _prev_cost_type != cost_type:
        G.init_link_costs(cost_type)
        _prev_cost_type = cost_type

    orig_node_no = G.get_node_no(orig_node_id)
    _optimal_label_correcting_CAPI(G, orig_node_no)
                        

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
            yield f"{G.links[link_no].get_link_id()}"


def find_shortest_path(G, from_node_id, to_node_id, seq_type, cost_type):
    if from_node_id not in G.map_id_to_no:
        raise Exception(f'Node ID: {from_node_id} not in the network')
    if to_node_id not in G.map_id_to_no:
        raise Exception(f'Node ID: {to_node_id} not in the network')
        
    single_source_shortest_path(G, from_node_id, cost_type)

    path_cost = G.get_path_cost(to_node_id, cost_type)
    
    if path_cost >= MAX_LABEL_COST:
        return f'path {cost_type}: infinity | path: '

    path = _get_path_sequence_str(G, to_node_id, seq_type)

    unit = 'minutes'
    if cost_type.startswith('dis'):
        unit = G.get_length_unit() + 's'

    if seq_type.startswith('node'):
        return f'path {cost_type}: {path_cost:.4f} {unit} | node path: {path}'
    else:
        return f'path {cost_type}: {path_cost:.4f} {unit} | link path: {path}'


def get_shortest_path(G, from_node_id, to_node_id, cost_type):
    # exceptions
    if from_node_id not in G.map_id_to_no:
        #return None
        raise Exception(f'Node ID: {from_node_id} not in the network')
    if to_node_id not in G.map_id_to_no:
        #return None
        raise Exception(f'Node ID: {to_node_id} not in the network')
    
    single_source_shortest_path(G, from_node_id, cost_type)
    
    path_cost = G.get_path_cost(to_node_id, cost_type)
  
    if path_cost >= MAX_LABEL_COST:
        return 9999999
    else:
       return path_cost

def compute_row_distances(G, row_node, row_nodes, cost_type):
    return [get_shortest_path(G, row_node, col_node, cost_type) for col_node in row_nodes]


def create_numpy_matrix_parallel(G, row_nodes, cost_type):
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
        delayed(compute_row_distances)(G, row_node, row_nodes, cost_type)  # Corrected function call
        for row_node in tqdm(row_nodes, desc="Computing shortest paths"))
    
    elapsed_time = time.time() - start_time
    print(f"Matrix Creation Time: {elapsed_time:.2f} s")

    return np.array(skim_matrix)


def save_to_omx(matrix, row_nodes, output_path, matrix_name, attributes=None):
    """
    Saves the given matrix inside an OMX file, adding multiple matrices into the same file.
    
    Parameters:
        matrix (numpy.ndarray): The matrix to be saved.
        row_nodes (list): List of node identifiers for mapping.
        output_path (str): Path to the OMX file.
        matrix_name (str): Name of the matrix to store inside the OMX file.
        attributes (dict, optional): Dictionary of attributes to store with the matrix.
    """
    with omx.open_file(output_path, "a") as omx_out:  # "a" (append mode) to prevent overwriting
        omx_out[matrix_name] = matrix  # Store the matrix

        # Add attributes if provided
        if attributes:
            for key, value in attributes.items():
                omx_out[matrix_name].attrs[key] = value  # Assign attributes

        # Add node mappings only once
        if "nodes" not in omx_out.list_mappings():
            omx_out.create_mapping("nodes", row_nodes)

    print(f"Matrix '{matrix_name}' saved to: {output_path}")


def save_as_csv(matrix, row_nodes, output_path):
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([""] + row_nodes)  # Write header row
        for i, row in enumerate(matrix):
            writer.writerow([row_nodes[i]] + [f"{val:.2f}" for val in row])  # Write data rows with 2 decimal places
    print(f"Matrix saved to {output_path}")


def find_shortest_path_network(G, output_dir, output_type, cost_type, mode):
    # Check to make sure output_type is an accepted type.
    valid_types = {".omx", ".csv"}
    if output_type not in valid_types:
        raise ValueError(f"Error: Unsupported output type '{output_type}'. Please use one of {valid_types}.")
    
    print(mode.type)
    # Will compute skim for centroids only, checking the zone_id column for nonempty.
    row_nodes = [G.nodes[i].zone_id for i in range(G.node_size) if G.nodes[i].zone_id and G.nodes[i].zone_id.strip().isdigit()]

    # Modify the free flow travel time of link if mode is not all or auto.
    if mode.type == "p":
        for i in range(G.link_size):
            G.links[i].fftt = (G.links[i].length * 5280)/mode.ffs /60 #Convert fftt for walking
    if mode.type == "t":
        for i in range(G.link_size):
            if "p" in G.links[i].allowed_uses: #Convert fftt on pedestrain links only, not transit. 
                G.links[i].fftt = (G.links[i].length * 5280)/mode.ffs /60 
    elif mode.type == "b":
        for i in range(G.link_size):
            G.links[i].fftt = G.links[i].length / mode.ffs * 60 #Convert fftt for biking
    
    
    # Compute shortest path matrix in parallel
    skim_matrix = create_numpy_matrix_parallel(G, row_nodes, cost_type)
    
    #Assign attributes to the matrix
    matrix_attributes = {
    "Description": f"Shortest path travel time matrix for {mode.name}",
    "mode": mode.name,
    "Units": "minutes",
    "Date": datetime.now().date(),
    "time": datetime.now().time(),
    "Calculation_Method": "Dijkstra Algorithm "
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the matrix in the requested format
    output_path = os.path.join(output_dir, f"shortest_path_matrix_{cost_type}{output_type}")

    if output_type == ".csv":
        output_path = os.path.join(output_dir, f"shortest_path_matrix_{cost_type}_{mode.name}{output_type}")
        save_as_csv(skim_matrix, row_nodes, output_path)
    
    elif output_type == ".omx":
        save_to_omx(skim_matrix, row_nodes, output_path, f"{mode.name}", matrix_attributes)
    else:
        raise ValueError(f"Error: Unsupported output type '{output_type}'. Please use one of ['.csv', '.omx'].")

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


def _get_path_sequence_str(G, to_node_id, seq_type):
    return ';'.join(str(x) for x in output_path_sequence(G, to_node_id, seq_type))


def get_shortest_path_tree(G, from_node_id, seq_type, cost_type, integer_node_id):
    """ compute the shortest path tree from the source node (from_node_id)

    it returns a dictionary, where key is to_node_id and value is the
    corresponding shortest path information (path cost and path details).

    Note that the source node itself is excluded from the dictionary keys.
    """
    if from_node_id not in G.map_id_to_no:
        raise Exception(f'Node ID: {from_node_id} not in the network')

    single_source_shortest_path(G, from_node_id, cost_type)
    '''
    if integer_node_id:
        sp_tree = {}
        for to_node_id in G.map_id_to_no:
            if to_node_id == from_node_id:
                continue

            try:
                to_node_id_int = _convert_str_to_int(to_node_id)
            except InvalidRecord:
                to_node_id_int = to_node_id

            sp_tree[to_node_id_int] = (
                G.get_path_cost(to_node_id, cost_type),
                _get_path_sequence_str(G, to_node_id, seq_type)
            )

        return sp_tree
    else:
        return {
            to_node_id : (
                G.get_path_cost(to_node_id, cost_type),
                _get_path_sequence_str(G, to_node_id, seq_type)
            )
            for to_node_id in G.map_id_to_no if to_node_id != from_node_id
        }
        '''


def benchmark_apsp(G):
    st = time()

    for k in G.map_id_to_no:
        single_source_shortest_path(G, k, 'c')

    print(f'processing time of finding all-pairs shortest paths: {time()-st:.4f} s')