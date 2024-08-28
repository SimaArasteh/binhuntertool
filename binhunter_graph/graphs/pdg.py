# This module implements Program Dependence Graph using  Ghidra
from ghidra.util.task import TaskMonitor
import networkx as nx
from binhunter_graph.graphs.ddg import BH_DDG
from binhunter_graph.graphs.cdg import BH_CDG
from binhunter_graph.slicing.debug import DebugInfo
from collections import defaultdict
from collections import Counter
import ghidra
import json
import joblib

class BH_PDG():
    def __init__(self, dataset='juliet'):
        self.dataset= 'juliet'
        self.current_program = getCurrentProgram()
        self.bin_base_address = hex(int(self.current_program.getImageBase().toString(), 16))
        self.bin_file = self.current_program.getExecutablePath()
        self.function_name = None
        self.cdg = None
        self.cdg_levels = {}
        self.ddg = None
        self.ddg_obj= None
        self.cdg_obj = None
        self.pdg = None
        self.call_graph = self.create_call_graph()
        self.create_pdg()
        
    
    def create_pdg(self):
        #breakpoint()
        target_function = self.find_target_function()
        self.function_name = target_function
        bh_ddg = BH_DDG(self.bin_base_address, self.current_program, self.bin_file, target_function)
        self.ddg = bh_ddg.ddg
        self.ddg_obj = bh_ddg
        bh_cdg = BH_CDG(self.bin_base_address, self.current_program, self.bin_file, target_function)
        self.cdg = bh_cdg.cdg
        self.cdg_obj = bh_cdg
        self.cdg_levels = bh_cdg.control_dependency_levels
        #print(self.cdg_levels)
        

    def create_call_graph(self):
        cg = nx.DiGraph()
        fm = self.current_program.getFunctionManager()
        functions = fm.getFunctions(True)
        for func in functions:
            func_name = func.getName()
            callees = self.find_callee_func(func)
            cg.add_node(func_name)
            for f in callees:
                callee_name = f.getName()
                cg.add_node(callee_name)
                cg.add_edge(func_name, callee_name)
        return cg


    def find_target_function(self):
        #find the target function
        # in juliet test suite dataset, there is a main function
        # main function calls a function and this function is a target function.
        #self.create_call_graph()
        if self.dataset == "juliet":
            succ = list(self.call_graph.successors('main'))
            if "juliet_bad" in self.bin_file:
                for item in succ:
                    if 'bad' in item:
                        pre = list(self.call_graph.predecessors(item))
                        if len(pre) == 1 and pre[0] == 'main':
                            
                            return item
            elif "juliet_good" in self.bin_file:
                for item in succ:
                    if 'good' in item:
                        pre = list(self.call_graph.predecessors(item))
                        if len(pre) == 1 and pre[0] == 'main': # the predessecor of the target function should be only main
                            #  item here is a caller function but sometimes this caller function only calls 
                            # patch functions that patch the vulnerability in different ways
                            good_callees =  list(self.call_graph.successors(item))
                            #breakpoint()
                            for entry in good_callees:
                                if 'good' in entry:
                                    return entry

                            return item
                            
    
        else:
            pass
    
    def find_callee_func(self, caller_func):
        monitor = TaskMonitor.DUMMY
        callees = caller_func.getCalledFunctions(monitor)
        return callees

    def add_control_dependency_levels(self):
        # adding control dependency levels for each node
        #breakpoint()
        pdg_graph = nx.DiGraph()
        pdg_graph.graph['base_addr'] = self.ddg.graph['base_addr']
        pdg_attributes = defaultdict(lambda: {'label':None, 'node_type':None, 'ins_addr':set(), 'bb_addr' :set()})
        for node1, node2 , data in self.ddg.edges(data=True):
            #print(data)
            #breakpoint()
            node1_attrs = self.ddg.nodes[node1]
            
            
            bb_addr = node1_attrs['bb_addr']
            
            basic_block_addr = list(bb_addr)[0]
            bb_offset = hex(int(basic_block_addr, 16)-int(self.bin_base_address, 16))
            
            #mapping = self.map_dependency_levels()
            new_node_val1= None
            new_node_val2 = None
            ls =list( self.cdg_levels.values())
            c = Counter(ls)
            m, a = c.most_common(1)[0]

            if bb_offset in list(self.cdg_levels.keys()):
                
                bb_level = self.cdg_levels[bb_offset] #dependency level for each node
                #print(bb_level)
                new_node_val1 = node1+"_"+str(bb_level)
                #print(new_node_val)
                
                pdg_graph.add_node(new_node_val1)
                pdg_attributes[new_node_val1]['label'] = node1_attrs['label']
                pdg_attributes[new_node_val1]['node_type'] = node1_attrs['node_type']
                pdg_attributes[new_node_val1]['ins_addr'] = node1_attrs['ins_addr']
                pdg_attributes[new_node_val1]['bb_addr']= node1_attrs['bb_addr']
                #print(pdg_attributes)
            node2_attrs = self.ddg.nodes[node2]
            
            bb_addr2 = node2_attrs['bb_addr']
            basic_block_addr2 = list(bb_addr2)[0]
            bb_offset2 = hex(int(basic_block_addr2, 16)-int(self.bin_base_address, 16))
            #mapping1 = self.map_dependency_levels()
            
            if bb_offset2 in list(self.cdg_levels.keys()):
                
                bb_level2 = self.cdg_levels[bb_offset2] #dependency level for each node
                #print(bb_level2)
                new_node_val2 = node2+"_"+str(bb_level2)
                #print(new_node_val)
                
                pdg_graph.add_node(new_node_val2)
                pdg_attributes[new_node_val2]['label'] = node2_attrs['label']
                pdg_attributes[new_node_val2]['node_type'] = node2_attrs['node_type']
                pdg_attributes[new_node_val2]['ins_addr'] = node2_attrs['ins_addr']
                pdg_attributes[new_node_val2]['bb_addr']= node2_attrs['bb_addr']
            if new_node_val1 is not None and new_node_val2 is not None:
                pdg_graph.add_edge(new_node_val1, new_node_val2, edge_type= data['edge_type'])
            else:
                
                new_node_val1 = node1+"_"+str(m)
                new_node_val2 =  node2+"_"+str(m)
                pdg_graph.add_node(new_node_val1)
                pdg_attributes[new_node_val1]['label'] = node1_attrs['label']
                pdg_attributes[new_node_val1]['node_type'] = node1_attrs['node_type']
                pdg_attributes[new_node_val1]['ins_addr'] = node1_attrs['ins_addr']
                pdg_attributes[new_node_val1]['bb_addr']= node1_attrs['bb_addr']
                pdg_graph.add_node(new_node_val2)
                pdg_attributes[new_node_val2]['label'] = node2_attrs['label']
                pdg_attributes[new_node_val2]['node_type'] = node2_attrs['node_type']
                pdg_attributes[new_node_val2]['ins_addr'] = node2_attrs['ins_addr']
                pdg_attributes[new_node_val2]['bb_addr']= node2_attrs['bb_addr']
                pdg_graph.add_edge(new_node_val1, new_node_val2, edge_type= data['edge_type'])

        nx.set_node_attributes(pdg_graph, pdg_attributes)
        #for edge_start, edge_end, attrs in self.ddg.edges(data=True):
            #print(attrs)
            #pdg_graph.add_edge(edge_start, edge_end, **attrs)
        #breakpoint()
        #print(pdg_graph.nodes(data=True))
        #breakpoint()
        self.pdg= pdg_graph.copy()



    
    def map_dependency_levels(self):
        address_level = {}
        for addr in self.cdg_levels:
            level = self.cdg_levels[v]
            addr_str = str(hex(addr))
            address_level[addr_str] = level
        return address_level
               


    def extend_target_locations(self, ins_addres, data_dependency_graph, target_locations, base_address):
        #breakpoint()
        #print("this is target locations")
        #print(target_locations)
        if len(target_locations):
            ins_addres.sort()
            target_addresses = set()
            border_addresses = set()
            final_addresses = []
            legit_node_types = ['Operation', 'Register', 'Const', 'Unique']
            
            for target in target_locations:
                #breakpoint()
                for node, node_feature in data_dependency_graph.nodes(data=True):
                    if node_feature['node_type'] in legit_node_types:
                        full_adr = hex(int(target, 16)+int(base_address, 16))
                        if full_adr in node_feature['ins_addr']:
                            target_addresses.update(node_feature['ins_addr'])
                #breakpoint()
            target_list = list(target_addresses)
            target_list.sort()
            #print(target_list)

            target_locations.sort()
            location_start = target_locations[0]
            full_start =  hex(int(location_start, 16)+int(base_address, 16))
            if full_start in target_list:
                return list(set(target_list[target_list.index(full_start):]))
            else:
                return list(set(target_list))

        else:
            return None

    def Slice_PDG(self, portion_addresses, base):
        #breakpoint()
        if portion_addresses is not None:
            subgraph_nodes = set()
            subgraph_edges = [] # this is the problem
            for addr in portion_addresses:
                #print(addr)
                subgraph_edges = subgraph_edges + self.ddg_obj.permitted_edges[addr]
                for n , n_attr in self.pdg.nodes(data=True):
                    if addr in n_attr['ins_addr'] and n is not None:
                        subgraph_nodes.add(n)

            
            ddg_subgraph = self.pdg.subgraph(subgraph_nodes)
            #print("length of ddg_subgrpah edges before removing")
            #print(len(ddg_subgraph.edges()))
            unwanted_edges = []
            for edge in ddg_subgraph.edges():
                node1 = edge[0]
                node2 = edge[1]
                node1_new = node1.rsplit('_', 1)[0]
                node2_new = node2.rsplit('_', 1)[0]

                if (node1_new, node2_new) not in subgraph_edges:
                    unwanted_edges.append(edge)
            H = ddg_subgraph.copy()
            H.remove_edges_from(unwanted_edges)

            if len(H.nodes()) == len(H):
                return H 
            else:

                set_dif = set(ddg_subgraph.nodes()).symmetric_difference(set(subgraph_nodes))
                not_conncted_nodes = list(set_dif)
                not_concted_nodes_2 = ddg_subgraph.nodes()
                for s in not_conncted_nodes:
                    for d in not_concted_nodes_2:
                        path = nx.shortest_path(self.pdg, source=s, target=d)
                        if len(path):
                            right_nodes = set(path+subgraph_nodes)
                            ddg_subgraph = self.pdg.subgraph(right_nodes)
                            return ddg_subgraph

        else:
            return None

    def compute_PDG_subgraph(self):
        #breakpoint()
        #bincdg = self.compute_CDG()
        #find debug info for all cus
        main_path = self.bin_file.replace(self.bin_file.split("/")[-1], '')
        dl = DebugInfo(self.bin_file)
        ml = dl.find_memory_offsets()
        self.map_ins_offset = ml
        vul_offsets = []
        for source_file in dl.cus:
            source_path = main_path+source_file
            #print(source_path)
            variant_num = source_path.split("/")[-1].split(".")[0].split("_")[-1]
            #if variant_num.isdigit(): # we set this condition to avoid having long range data dependency samples
            #ignore vulnerability across function for now
            lines = self.extract_source_line_number(source_path)
            #print("lines:")
            #print(lines)
            for l in lines:
                vul_offsets+=ml[source_file][l]
        
        if len(vul_offsets):
            all_addresses = []
            #print(len(self.ddg.nodes()))
            #print(len(self.pdg.nodes()))
            graph_address = self.pdg.graph['base_addr']
            for node, node_feature in self.pdg.nodes(data= True):
                #print(node)
                #print(node_feature)
                for addr in node_feature['ins_addr']:
                    all_addresses.append(hex(abs(int(addr, 16)-int(graph_address, 16))))

            #print(len(vul_offsets))
            

            unique_addresses = list(set(all_addresses))
            #print(len(unique_addresses))
            wanted_addresses = self.extend_target_locations(unique_addresses,self.pdg, vul_offsets,graph_address )
            #wanted_addresses.sort()
            if wanted_addresses is not None:
                #print("here is the length of ")
                #print(len(wanted_addresses))
                pdg_sub = self.Slice_PDG(wanted_addresses, graph_address)
                #print(len(ddg_sub.nodes()))
                attrs_g = {'range_of_address': wanted_addresses , 'debug_line_addressess':vul_offsets}
                pdg_sub.graph.update(attrs_g)
                return pdg_sub
            else:
                return None
        else:
            return None


    def compute_PDG_subgraph_decompiled(self):
        #breakpoint()
        # compute function cdg 
        bincdg = self.compute_CDG()
        #find debug info for all cus
        main_path = self.bin_path.replace(self.bin_path.split("/")[-1], '')
        dl = DebugInfo(self.bin_path)
        ml = dl.find_memory_offsets()
        self.map_ins_offset = ml
        vul_offsets = []
        for source_file in dl.cus:
            source_path = main_path+source_file
            #print(source_path)
            variant_num = source_path.split("/")[-1].split(".")[0].split("_")[-1]

            if variant_num.isdigit():
                #ignore vulnerability across function now
                lines = self.extract_source_line_number(source_path)
                #print("lines:")
                #print(lines)
                for l in lines:
                    vul_offsets+=ml[source_file][l]
        all_addresses = []
        graph_address = self.Bin_DGG.graph['base_addr']
        for node, node_feature in self.Bin_DGG.nodes(data= True):
            for addr in node_feature['ins_addr']:
                all_addresses.append(hex(abs(int(addr, 16)-int(graph_address, 16))))


        final_memory_offsets = self.find_target_memory_offsets(list(set(all_addresses)), vul_offsets )
        #print(vul_offsets)
        #print(final_memory_offsets)
        ################################################### slice ddg based on final memory offsets 
    
        # from the vul location or patch location go one level up and one level down
        pdg_subgraph = nx.DiGraph()
        #print(memory_offs)
        pdg_attributes = defaultdict(lambda: {'label':None, 'node_type':None, 'ins_addr':set(), 'bb_addr' :set(), 'cdg_level':[]})
        if nx.is_weakly_connected(self.Bin_DGG):
            # if there is no seperated graph 
            notint = set()
            bin_base_address = self.Bin_DGG.graph['base_addr']
            for node_value , node_attr in self.Bin_DGG.nodes(data=True):
                #breakpoint()
                #print(node_value)
                #print(node_attr)
                pdg_node_attribute = {}
                if node_value is not None:
                    #print(node_value)
                    for addr in node_attr['ins_addr']:

                        node_offset = int(addr,16) - int(bin_base_address, 16)
                        if hex(node_offset) in final_memory_offsets:
                            #breakpoint()
                            pdg_subgraph.add_node(node_value)
                            pdg_attributes[node_value]['label'] = node_attr['label']
                            pdg_attributes[node_value]['node_type'] = node_attr['node_type']
                            pdg_attributes[node_value]['ins_addr'] = node_attr['ins_addr']
                            pdg_attributes[node_value]['bb_addr'] = node_attr['bb_addr']
                            pre_nodes = list(self.Bin_DGG.predecessors(node_value))
                            succ_nodes = list(self.Bin_DGG.successors(node_value))
                            for n in pre_nodes:
                                
                                #n_arr = nx.get_node_attributes(self.Bin_DGG, n)
                                
                                if n is not None:
                                   
                                    #breakpoint()
                                    pdg_subgraph.add_node(n)
                                    pdg_attributes[n]['label'] = self.Bin_DGG.nodes()[n]['label']
                                    pdg_attributes[n]['node_type'] = self.Bin_DGG.nodes()[n]['node_type']
                                    pdg_attributes[n]['ins_addr'] =  self.Bin_DGG.nodes()[n]['ins_addr']
                                    pdg_attributes[n]['bb_addr'] = self.Bin_DGG.nodes()[n]['bb_addr']
                                    pdg_subgraph.add_edge(n,node_value, edge_type = 'data')

                            
                            for no in succ_nodes:
                                
                                #no_arr = nx.get_node_attributes(self.Bin_DGG, no)
                                if no is not None:
                                    #print(no)
                                    
                                    #breakpoint()
                                    pdg_subgraph.add_node(no)
                                    pdg_attributes[no]['label'] = self.Bin_DGG.nodes()[no]['label']
                                    pdg_attributes[no]['node_type'] = self.Bin_DGG.nodes()[no]['node_type']
                                    pdg_attributes[no]['ins_addr'] =  self.Bin_DGG.nodes()[no]['ins_addr']
                                    pdg_attributes[no]['bb_addr'] = self.Bin_DGG.nodes()[no]['bb_addr']
                                    pdg_subgraph.add_edge(node_value, no, edge_type = 'data')
                                 

            if nx.is_weakly_connected(pdg_subgraph):
                cdg_priority = self.compute_cdgnode_priority(self.Bin_CDG)
                #print(cdg_priority)
                for node in pdg_subgraph.nodes():
                    #print(node)
                    pdg_ins = pdg_attributes[node]['ins_addr']
                    #print(pdg_ins)
                    cdg_levels = []
                    for inspdg in pdg_ins:
                        off = int(inspdg, 16) - int(graph_address, 16)
                        if hex(off) in list(self.map_ins_block.keys()):
                            target = self.map_ins_block[hex(off)]
                            if target in list(cdg_priority.keys()):
                                cdg_levels.append(cdg_priority[target])
                    pdg_attributes[node]['cdg_level'] = list(set(cdg_levels))
                nx.set_node_attributes(pdg_subgraph, pdg_attributes)
                pdg_subgraph.graph['number_bbs'] = self.compute_number_basicblock(pdg_subgraph)
                return pdg_subgraph

            else :
                return None
                

        else:
            return None

    def compute_cdgnode_priority(self, cdg):
        block_priority = {}
        for node in cdg.graph.nodes():
            #print(node)
            node_pre = list(cdg.graph.predecessors(node))
            node_post = list(cdg.graph.successors(node))
            if not len(node_pre):
                depth_levels = nx.shortest_path_length(cdg.graph,node)
                for node in depth_levels:
                    if node.block is not None:
                        block_priority[hex(node.block.addr-self.angr_base)] = depth_levels[node]


        return block_priority

    def find_target_memory_offsets(self, pcode_address, memory_targets):
        new_targets = []
        pcode_address.sort()
        for addr in memory_targets:
            if addr in pcode_address:
                new_targets.append(addr)
            else:
                for idx in range(len(pcode_address)):
                    if idx+1 < len(pcode_address):
                        if addr > pcode_address[idx] and addr < pcode_address[idx+1]:
                            new_targets.append(pcode_address[idx])


        return new_targets

    def compute_number_basicblock(self, pdg_sub):
        different_basic_blocks = set()

        for node, node_attr in pdg_sub.nodes(data=True):
            for bb in node_attr['bb_addr']:
                different_basic_blocks.add(bb)

        return len(different_basic_blocks)
    
    

    def extract_source_line_number(self,source_file):
        #extract line numbers from all source files that created a binary
        #return line numbers for each source file
        source_target_lines = {}

        with open(source_file) as f:
            lines = f.readlines()
        
        index_start=0 
        index_finish=0
        files  = []
        if 'juliet_good' in source_file and '#ifndef OMITGOOD\n' in lines : 
            #print(source_file)
            index_start = lines.index('#ifndef OMITGOOD\n')+1
            index_finish = lines.index('#endif /* OMITGOOD */\n')+1
            for j in range(index_start, index_finish):
                if 'FIX:' in lines[j] or 'FLAW:' in lines[j]:
                    patch_line_num = j+1
                    if '*/' in lines[patch_line_num]:
                        patch_line_num+=1

                    if 'if' in lines[patch_line_num] or  'while' in lines[patch_line_num] or 'for' in lines[patch_line_num]:
                        start_flow = patch_line_num
                        while '}\n' not in lines[patch_line_num]:
                            files.append(patch_line_num+1)
                            patch_line_num+=1
                    else:
                        files.append(patch_line_num+1)
            

        if 'juliet_bad' in source_file and '#ifndef OMITBAD\n' in lines:
            
            index_start = lines.index('#ifndef OMITBAD\n') +1
            index_finish = lines.index('#endif /* OMITBAD */\n')+1
            slice = lines[index_start: index_finish]
            #print(slice)
            for i in range(index_start, index_finish):
                
                if 'FLAW:' in lines[i]:
                    
                    vul_line_num = i+1
                    if '*/' in lines[vul_line_num]:
                        vul_line_num+=1

                    if 'if' in lines[vul_line_num] or  'while' in lines[vul_line_num] or 'for' in lines[vul_line_num]:
                        start_flow = vul_line_num
                        while '}\n' not in lines[vul_line_num]:
                            files.append(vul_line_num+1)
                            vul_line_num+=1 
                    else:
                        files.append(vul_line_num+1)
                    
                
        #source_target_lines[source_file] = files    

        return files

    
#breakpoint()

if __name__ == "__main__":
    
    bhpdg = BH_PDG()
    bhpdg.add_control_dependency_levels()
    pdg_sub = bhpdg.compute_PDG_subgraph()
    #breakpoint()
    print(pdg_sub.edges())
    print(len(pdg_sub.nodes(data=True)))
    
    joblib.dump((pdg_sub, bhpdg.pdg, bhpdg.ddg, bhpdg.cdg, bhpdg.bin_file, bhpdg.function_name, bhpdg.ddg_obj.permitted_edges)  , 'pdg_sub.pkl')



    
    