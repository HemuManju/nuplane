from lxml import etree


class BaseWriter:
    def __init__(self) -> None:
        self.id = 1
        self.id_counter_incr = 1
        return None

    def _get_new_id(self):
        self.id += self.id_counter_incr
        return self.id

    def file_open(self, filename):
        self.f = open(filename, 'w', buffering=-1, encoding='utf-8')

    def node_to_xml(self, nodes):
        raise NotImplementedError

    def edge_to_xml(self, edges):
        raise NotImplementedError

    def file_close(self):
        self.f.write('</osm>')
        if self.f:
            self.f.close()
            self.f = None

    def write_header(self):
        self.f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        self.f.write('<osm version="0.6" generator="OSM">\n')

    def write(self, file_name, nodes, edges):
        # Open then file
        self.file_open(file_name)

        # Write the header
        self.write_header()

        self.id = len(nodes)

        # Write nodes and edges
        for node in nodes:
            self.f.write(self.node_to_xml(node))
            self.f.write('\n')

        for edge in edges:
            self.f.write(self.edge_to_xml(edge))
            self.f.write('\n')


class OSMDataWriter(BaseWriter):
    def __init__(self) -> None:
        super().__init__()

    def node_to_xml(self, node):
        xmlattrs = {
            'visible': 'true',
            'id': ('%d' % node[0]),
            'lat': str(node[1]['lat']),
            'lon': str(node[1]['lon']),
        }

        xmlobject = etree.Element('node', xmlattrs)

        return etree.tostring(xmlobject, encoding='unicode')

    def edge_to_xml(self, edge):
        xmlattrs = {'visible': 'true', 'id': (f'{self._get_new_id()}')}
        xmlattrs.update(edge[2])

        xmlobject = etree.Element('way', xmlattrs)
        # Start
        nd = etree.Element('nd', {'ref': ('%d' % edge[0])})
        xmlobject.append(nd)

        # Stop
        nd = etree.Element('nd', {'ref': ('%d' % edge[1])})
        xmlobject.append(nd)

        return etree.tostring(xmlobject, encoding='unicode')
