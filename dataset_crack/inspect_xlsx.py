from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import sys

sys.stdout.reconfigure(encoding='utf-8')

path = Path('BFY_ModelComp.xlsx')
ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(path) as z:
    root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    strings_root = ET.fromstring(z.read('xl/sharedStrings.xml'))
    strings = [''.join(si.itertext()) for si in strings_root.findall('m:si', ns)]
    dimension = root.find('m:dimension', ns)
    print(dimension.attrib if dimension is not None else 'no dimension')
    for row in root.findall('.//m:row', ns):
        vals=[]
        for cell in row.findall('m:c', ns):
            value = cell.find('m:v', ns)
            text = '' if value is None else value.text
            if cell.attrib.get('t') == 's' and text:
                text = strings[int(text)]
            vals.append(f"{cell.attrib.get('r')}={text}")
        print(' | '.join(vals))
