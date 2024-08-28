# This module contains functions that create control dependence graph
from ghidra.program.model.symbol import SymbolType
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.program.model.block import BasicBlockModel
from ghidra.graph.jung import JungToGDirectedGraphAdapter
from edu.uci.ics.jung.graph import DirectedSparseGraph
from ghidra.graph import GraphFactory
from ghidra.util.graph import DirectedGraph
from ghidra.util.graph import Edge 
import angr
from ghidra.util.graph import Vertex
from ghidra.util.task import TaskMonitor
from elftools.elf.elffile import ELFFile
from ghidra.graph import GraphAlgorithms
import ghidra
import networkx as nx

class BH_CDG():
    def __init__(self, base, current_program, bin_path, function_name):
        self.current_program = current_program
        self.bin_path = bin_path
        self.base_addr = base
        self.function_name = function_name
        self.cfg = None
        self.cdg = None
        self.base_an = None
        self.monitor = None
        self.control_dependency_levels = {}
        self.create_cdg()
    
    def create_cdg(self):
        self.create_cfg()
        cfg_ghidra = self.convert_to_graphghidra(self.cfg)
        #breakpoint()
        cdg , base = self.create_cdg_alter()
        self.cdg = cdg
        self.base_an = base
        self.control_dependency_levels = self.find_depth_level_for_cdg(cdg, base)
    
    

    def create_cfg(self):
        #breakpoint()
        fm = self.current_program.getFunctionManager()
        funcs = fm.getFunctions(True)
        monitor = ConsoleTaskMonitor()
        for func in funcs: 
            if func.getName() == self.function_name:
                func_cfg , m= self.create_cfg_function(self.current_program, func, monitor)
                self.cfg = func_cfg
                self.monitor = m
                
    def create_cdg_alter(self):
        b = angr.Project(self.bin_path, load_options={"auto_load_libs": False})
        obj = b.loader.main_object
        base_address = b.loader.main_object.min_addr
        symbols = obj.symbols_by_name
        for symbol in symbols:
            if self.function_name in symbol:
                function_symbol = b.loader.find_symbol(symbol)    
                if function_symbol is not None:
                    cfg = b.analyses.CFGEmulated(start=function_symbol.rebased_addr, normalize=True, keep_state=True, 
                                            state_add_options=angr.sim_options.refs,)
                    
                    cdg = b.analyses.CDG(cfg, start =function_symbol.rebased_addr)
                    return cdg, base_address


    def create_cfg_function(self, cu_program, function, mon):
        # create cfg for a single function
        #print("address in cdg")
        block_model_iterator = BasicBlockModel(cu_program)
        function_addresses = function.getBody()
        code_blocks_iterator = block_model_iterator.getCodeBlocksContaining(function_addresses, mon)
        cfg = DirectedSparseGraph()
        while (code_blocks_iterator.hasNext()):
            
            block = code_blocks_iterator.next()
            addr = hex(block.getFirstStartAddress().getOffset())
            #print(addr)
            v = Vertex(block)
            cfg.addVertex(v)
            dstBlocks = block.getDestinations(mon)
            srcBlocks = block.getSources(mon)
            while(srcBlocks.hasNext()):
                source = srcBlocks.next()
                vsrc = Vertex(source)
                cfg.addVertex(vsrc)
                edge1 = Edge(vsrc, v)
                res = cfg.addEdge(edge1, vsrc, v )
                src_addr = hex(source.getSourceAddress().getOffset())
                #print(src_addr)
                
            while (dstBlocks.hasNext()):
                destination = dstBlocks.next()
                des_addr = hex(destination.getDestinationAddress().getOffset())
                vdes = Vertex(destination)
                cfg.addVertex(vdes)
                edge2 = Edge(v, vdes)
                cfg.addEdge(edge2, v, vdes)
                #print(des_addr)
        #breakpoint()
        return cfg, mon

    def convert_to_graphghidra(self, graph):
        
        edges = graph.getEdges()
        nodes = graph.getVertices()
        digraph = nx.DiGraph()

        edge_iter = edges.iterator()
        while edge_iter.hasNext():
            edge = edge_iter.next()
            endpoints = graph.getEndpoints(edge)
            
            source = endpoints.getFirst()
            destination = endpoints.getSecond()
            digraph.add_node(source)
            digraph.add_node(destination)
            digraph.add_edge(source, destination)
            
        return digraph

    def convert_to_sparsegraph(self, g):
        output = DirectedSparseGraph()
        for src, dsc in g.edges():
            output.addVertex(src)
            output.addVertex(dsc)
            e = Edge(src, dsc)
            output.addEdge(e, src, dsc )

        return output

    def create_postdominator_tree(self, node_postdominators):
        G = nx.DiGraph()

        # Add nodes and edges based on the immediate postdominator relationship
        for node, postdominators in node_postdominators.items():
            if postdominators:  # If the node has postdominators
                immediate_postdominator = postdominators[0]  # Assuming the first one is the immediate
                G.add_edge(node, immediate_postdominator)  # Add an edge from node to its immediate postdominator

        return G

    def cfg_reverse(self, cfg):
        new_g = nx.DiGraph()

        new_g.add_nodes_from(cfg.nodes())
        for src, dst, in cfg.edges():
            new_g.add_edge(dst, src)

        return new_g

    def find_exit_nodes(self, cfg):

        exit_nodes = [node for node in cfg.nodes() if cfg.out_degree(node) == 0]
        return exit_nodes

    def create_post_dominator_tree(self, graph , exit_node):
        dominators_reversed = nx.immediate_dominators(graph, exit_node)

        
        post_dominator_tree = nx.DiGraph()
        for node, dom in dominators_reversed.items():
            if node != dom:  
                post_dominator_tree.add_edge(dom, node)

        return post_dominator_tree

    def compute_dominance_frontier(self, graph , pdtree):
        df = {}

        # Perform a post-order search on the dominator tree
        for x in nx.dfs_postorder_nodes(pdtree):
            if x not in graph:
                # Skip nodes that are not in the graph
                continue

            df[x] = set()

            # local set
            for y in graph.successors(x):
                if y in pdtree.nodes():
                    if x not in pdtree.predecessors(y):
                        df[x].add(y)

            # up set
            if x is None:
                continue

            for z in pdtree.successors(x):
                if z is x:
                    continue
                if z not in df:
                    continue
                for y in df[z]:
                    if x not in list(pdtree.predecessors(y)):
                        df[x].add(y)

        return df
    def create_cdg_from_cfg_alter(self, cfg_ghidra):
        cdg, base = self.create_cdg_alter()
        cdg_ghidra = nx.DiGraph()
        if cdg is not None:
            for node1, node2 in cdg.graph.edges():
                if node1.block is not None and node2.block is not None:
                    src_offset = node1.block.addr - base
                    des_offset = node2.block.addr - base
                    mapp = self.mapping_cfgblock_address(cfg_ghidra)
                    if src_offset in list(mapp.keys()) and des_offset in list(mapp.keys()):
                        node1_ghidra = mapp[src_offset]
                        node2_ghidra = mapp[des_offset]
                        cdg_ghidra.add_node(node1_ghidra)
                        cdg_ghidra.add_node(node2_ghidra)
                        cdg_ghidra.add_edge(node1_ghidra, node2_ghidra)

        return cdg_ghidra



    def mapping_cfgblock_address(self, g):
        addr_blcks = {}
        #breakpoint()
        for n in g.nodes():
            block = n.referent()

            if isinstance(block, ghidra.program.model.block.CodeBlockReferenceImpl):
                des_address = int(block.getDestinationAddress().toString(), 16) - int(self.base_addr, 16)
                source_address = int(block.getSourceAddress().toString(), 16) - int(self.base_addr, 16)
                des_block = block.getDestinationBlock()
                src_block = block.getSourceBlock()
                addr_blcks[des_address]= des_block
                addr_blcks[source_address] = src_block

        return addr_blcks



    # this part of code is inspired by angr's cdg source code
    # unfortunately ghidra does not have code example for cdg
    def create_cdg_from_cfg(self, cfg, monit): 
        # create control dependence graph
        cdg = nx.DiGraph()
        #check if the graph is cyclic
        # if the cfg is cyclic remove edges
        if not nx.is_directed_acyclic_graph(cfg):
            cycles = list(nx.simple_cycles(cfg))

            for cycle in cycles:
                # For simplicity, remove the first edge in the cycle
                # You might need a more sophisticated method to choose which edge to remove
                cfg.remove_edge(cycle[0], cycle[1])

        reverse_cfg = self.cfg_reverse(cfg)
        exit_nodes = self.find_exit_nodes(cfg)

        pdt = self.create_post_dominator_tree(reverse_cfg, exit_nodes[0])
        rdf = self.compute_dominance_frontier(cfg, pdt)
        for y in cfg.nodes():
            if y not in rdf:
                continue
            for x in rdf[y]:
                cdg.add_edge(x, y)
        return cdg

    def find_root_node(self, directed_graph):
        root_nodes = []
        for node in directed_graph.nodes():
            # Check if the node has no incoming edges (in-degree is 0)
            #print(directed_graph.in_degree(node))
            if directed_graph.in_degree(node) == 0:
                # Return the first node found with no incoming edges
                root_nodes.append(node)
        # If no root node is found (e.g., graph is not a DAG or is empty), return None
        return root_nodes

    def find_depth_level(self, G, root_node):
        depth_levels = nx.single_source_shortest_path_length(G, root_node)
        return depth_levels

    def find_depth_level_for_cdg(self, cdg, b):
        vertex_depthlevel = {}
        root_node = self.find_root_node(cdg.graph)
            #print(root_node)
        start_node = None
        if len(root_node):
            start_node = root_node
        else:
            start_node = [list(cdg.graph.nodes())[0]]
        #print(start_node)
        #breakpoint()
        for n in start_node:
            depth = self.find_depth_level(cdg.graph, n)
            for node, d_level in depth.items():
                if node.block is not None:
                    
                    vertex_depthlevel[hex(node.block.addr-b)] = d_level
        return vertex_depthlevel

    

            


        





