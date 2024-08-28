# This module creates data dependency graph using Ghidra
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor 
from ghidra.program.model.pcode import HighFunction
from ghidra.program.model.symbol import RefType
from ghidra.app.decompiler import DecompileOptions
from collections import defaultdict
import networkx as nx
import ghidra

class BH_DDG():
    def __init__(self,base_address , current_program, bin_path, function_name):
        self.base_address = base_address
        self.current_program = current_program
        self.bin_file = bin_path
        self.function_name = function_name
        self.permitted_edges = None
        self.ddg = None
        self.create_ddg()
    
    def create_ddg(self):
        ddg_graph = nx.DiGraph()
        ddg_graph.graph['base_addr'] = self.base_address
        ddg_attributes = defaultdict(lambda: {'label':None, 'node_type':None, 'ins_addr':set(), 'bb_addr' :set()})
        allowed_edges = defaultdict(list)
        list_operations = []
        for fn in self.current_program.getFunctionManager().getFunctions(True):
            if fn.getName() == self.function_name:
                print(self.function_name)
                mapping = self.find_call_arguments_decompiled_ddg(fn)
                options = DecompileOptions()
                monitor = ConsoleTaskMonitor()
                ifc = DecompInterface()
                ifc.setOptions(options)
                ifc.openProgram(self.current_program)
                ifc.setSimplificationStyle("firstpass") # It should be firstpass otherwise some offsets wont showup
                res = ifc.decompileFunction(fn, 60, monitor)
                high = res.getHighFunction()
                opiter = high.getPcodeOps()
                index = 0
                while opiter.hasNext():
                    op = opiter.next()
                    basicblock_address = op.getParent().getStart().toString()
                    instruction_address = op.getSeqnum().getTarget().toString()
                    #print(instruction_address+': '+op.toString())
                    bb_address = hex(int(basicblock_address,16))
                    instr_address = hex(int(instruction_address, 16))
                    #print(bb_address)
                    #print(instr_address+':'+op.toString())
                    #print(instr_address)
                    inputs_op = list(op.getInputs())
                    output_op = op.getOutput()
                    mnemonic = op.getMnemonic()
                    
                    if mnemonic == 'CALL':
                        # add an edge for call to create a weakly connected graph
                        #print("hello")
                        if instr_address in list(mapping.keys()):
                            inpu = mapping[instr_address]
                            for entry in inpu:
                                if isinstance(entry, ghidra.program.model.pcode.VarnodeAST):
                                    if not entry.isAddress():
                                        inputs_op.append(entry)
                        #print(instr_address)
                        #print(inputs_op)
                    mnemonic_type , mnemonic_value, mnemonic_label = self.compute_node_value(mnemonic, list_operations)
                    #if 'Operation_COPY' in mnemonic_value :
                        #breakpoint()
                    ddg_graph.add_node(mnemonic_value)
                    ddg_attributes[mnemonic_value]['label'] = mnemonic_label
                    ddg_attributes[mnemonic_value]['node_type'] = mnemonic_type
                    ddg_attributes[mnemonic_value]['ins_addr'].add(instr_address)
                    ddg_attributes[mnemonic_value]['bb_addr'].add(bb_address)
                    
                    if output_op is not None:
                        output_dependent_address = set()
                        output_items = output_op.getDescendants()
                        #print(output_items)
                        for entry in output_items:
                            output_dependent_address.add(entry.getSeqnum().getTarget().toString())

                        output_type , output_value , output_label= self.compute_node_value(output_op, list_operations)
                        ddg_graph.add_node(output_value )
                        ddg_attributes[output_value]['label'] = output_label
                        ddg_attributes[output_value]['node_type'] = output_type
                        ddg_attributes[output_value]['ins_addr'].add(instr_address)
                        ddg_attributes[output_value]['bb_addr'].add(bb_address)
                        ddg_graph.add_edge(mnemonic_value, output_value, edge_type = 'data')
                        allowed_edges[instr_address].append((mnemonic_value, output_value))

            

                    for inp in inputs_op:
                        
                        input_dependent_address = set()
                        input_items = inp.getDescendants()
                        for each in input_items:
                            input_dependent_address.add(each.getSeqnum().getTarget().toString())

                        
                        input_type , input_value, input_label = self.compute_node_value(inp, list_operations)
                        if input_value is not None:
                            ddg_graph.add_node(input_value)
                            ddg_attributes[input_value]['label'] = input_label
                            ddg_attributes[input_value]['node_type'] = input_type
                            ddg_attributes[input_value]['ins_addr'].add(instr_address)
                            ddg_attributes[input_value]['bb_addr'].add(bb_address)
                            ddg_attributes[input_value]['data_dependent_state'] = input_dependent_address
                            ddg_graph.add_edge(input_value, mnemonic_value , edge_type = 'data')
                            allowed_edges[instr_address].append((input_value, mnemonic_value))
                    index+=1
        nx.set_node_attributes(ddg_graph, ddg_attributes)
        #breakpoint()
        self.permitted_edges= allowed_edges
        self.ddg = ddg_graph

    def compute_node_value(self, node, ls_ops):
        node_value = None
        node_type = None
        label= None
        
        if isinstance(node, ghidra.program.model.pcode.VarnodeAST):
            if node.isConstant():
                node_type = 'Const'
            elif node.isRegister():
                node_type = 'Register'
            elif node.isUnique():
                node_type = 'Unique'
            elif node.isAddress():
                node_type = 'Address'
            

            if node_type is not None:
                node_value = node_type+'_'+hex(node.getOffset()).rstrip("L")
                if node_type == 'Const':
                    node_size = node.getSize()
                    node_value = node_value+'_'+str(node_size)
                    label = node_type+'_'+str(node_size)
                else:
                    label = node_value
        else:
        
            node_type = 'Operation'
            node_value = node_type+"_"+node+"_"+str(ls_ops.count(node))
            label = node_value
            ls_ops.append(node)
        
        return node_type, node_value , label
    

    def find_call_arguments_decompiled_ddg(self, func):
        map_call_arguments = {}

        options_ = DecompileOptions()
        monitor_ = ConsoleTaskMonitor()
        ifc_ = DecompInterface()
        ifc_.setOptions(options_)
        ifc_.openProgram(getCurrentProgram())
        res_ = ifc_.decompileFunction(func, 60, monitor_)
        high_ = res_.getHighFunction()
        opiter_ = high_.getPcodeOps()
        
        
        while opiter_.hasNext():
            op_ = opiter_.next()
            #print(op_)
            
            basicblock_address_ = op_.getParent().getStart().toString()
            instruction_address_ = op_.getSeqnum().getTarget().toString()
            #print(instruction_address_+': '+op_.toString())
            bb_address_ = hex(int(basicblock_address_,16))
            instr_address_ = hex(int(instruction_address_, 16))
            #print(instr_address)
            inputs_op_ = list(op_.getInputs())
            output_op_ = op_.getOutput()
            mnemonic_ = op_.getMnemonic()
            if str(mnemonic_) == 'CALL':
                map_call_arguments[instr_address_] = inputs_op_
        
        return map_call_arguments  