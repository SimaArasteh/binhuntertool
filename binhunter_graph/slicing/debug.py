
from elftools.elf.elffile import ELFFile
from collections import defaultdict
import elftools


class DebugInfo():
    def __init__(self, binary_file):
        self.binary = binary_file
        self.elffile = None
        self.debug_info = None
        self.cus = []
        self.dwarf_exists()
        self.compute_cus()
    
    def dwarf_exists(self):
        with open(self.binary, 'rb') as file:
            elffile = ELFFile(file)
            if not elffile.has_dwarf_info():
                print('  file has no DWARF info')
                return 
            else:
                self.debug_info = elffile.get_dwarf_info()

    def compute_cus(self):
        if self.debug_info is not None:

            for CU in self.debug_info.iter_CUs():
                
                cu_die = CU.get_top_DIE()
                cu_name = ""
                for att_cu in cu_die.attributes:
                    if att_cu == 'DW_AT_name':
                        cu_name = cu_die.attributes[att_cu].value.decode("utf-8")
                        self.cus.append(cu_name)

    
    def find_memory_offsets(self):

        debugline_section = {}
        if self.debug_info is not None:

            for CU in self.debug_info.iter_CUs():
                
                cu_die = CU.get_top_DIE()
                cu_name = ""
                for att_cu in cu_die.attributes:
                    if att_cu == 'DW_AT_name':
                        cu_name = cu_die.attributes[att_cu].value.decode("utf-8")
            
                lines = self.debug_info.line_program_for_CU(CU)
                
                debugsec_lines = lines.get_entries()
                lines_offset = defaultdict(list)
                for entry in debugsec_lines:
                    if isinstance(entry, elftools.dwarf.lineprogram.LineProgramEntry):   
                        if entry.state is not None:
                            memory_offset = hex(entry.state.address)
                            source_line = entry.state.line
                            lines_offset[source_line].append(memory_offset)
    
                debugline_section[cu_name] = lines_offset
        
        return debugline_section 