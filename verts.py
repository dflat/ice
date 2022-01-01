from collections import defaultdict

def parse_obj(path):
    '''
    Read an .obj file, return a list of vertices.
    '''
    vert_frames = []
    link_frames = []
    verts = []
    links = []
    with open(path) as f:
        while True: # parse all objects
            line = f.readline()
            if not line: # end of file
                vert_frames = [i[1] for i in sorted(vert_frames)]
                link_frames = [i[1] for i in sorted(link_frames)]
                return clockwise_order(vert_frames, link_frames)
            items = line.split()
            if items[0] == 'o': # begin object
                index = int(items[1].split('_')[0])
                while True:
                    last_line_start = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    what, *data = line.split()
                    if what == 'v':
                        verts.append([float(c) for c in data])
                    elif what == 'l': # links
                        links.append([int(i) for i in data])
                    else:
                        break
                vert_frames.append((index, list(flip_y_strip_z(verts)))) # end object
                link_frames.append((index, links))
                verts = []
                links = []
                f.seek(last_line_start) 

def parse_obj_as_dict(path):
    '''
    Read an .obj file, return a dict of named vertex groups.
    '''
    vert_groups = defaultdict(list)
    verts = []
    links = []
    with open(path) as f:
        while True: # parse all objects
            line = f.readline()
            if not line: # end of file
                return vert_groups
            items = line.split()
            if items[0] == 'o': # begin object
                name = items[1].split('_')[0]
                while True:
                    last_line_start = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    what, *data = line.split()
                    if what == 'v':
                        verts.append([float(c) for c in data])
                    elif what == 'l': # links
                        links.append([int(i) for i in data])
                    else:
                        break
                clockwise_verts = [verts[i] for i in get_order(links)]
                vert_groups[name].append(list(flip_y_strip_z(clockwise_verts))) # endobject
                verts = []
                links = []
                f.seek(last_line_start) 

def flip_y_strip_z(verts):
    for v in verts:
        yield [v[0], -v[1]]

def clockwise_order(vert_frames, link_frames):
    result = []
    n = len(vert_frames)
    print('frames', n)
    for i in range(n):
        verts = vert_frames[i]
        links = link_frames[i]
        order = get_order(links)
        clockwise_verts = [verts[i] for i in order]
        result.append(clockwise_verts) 
    return result

def get_order(links):
    order = []
    start, seek = links.pop(0)
    order.extend([start,seek])
    while seek != start:
        for i, link in enumerate(links):
            if seek in link:
                break 
        a, b = links.pop(i)
        if a == seek:
            order.append(b)
            seek = b
        elif b == seek:
            order.append(a)
            seek = a
    order.pop() # remove final closing link back to start
    mn = min(order)
    return [x - mn for x in order] 
    
    
