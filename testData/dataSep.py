import re

name_pattern = '^>rno'

seq_math = '^[A-Z]'
fd = open("hairpin_mirbase_release22.fa","r")

while True:
    line_name = fd.readline().strip()
    if re.match(name_pattern, line_name) is not None:
        print (line_name)
        merged_seq = ""
        while True:
            # the location is for read again to match the line_seq
            location = fd.tell()
            line_seq = fd.readline()
            if re.match(seq_math, line_seq) is not None:
                merged_seq += line_seq.strip()
            else:
                fd.seek(location,0)
                print(merged_seq)
                break
    elif not line_name:
        break 
        
fd.close()
